"""Generate age-level simplification data with OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def build_prompt(original_text: str) -> str:
    return f"""
You are an expert in educational text simplification and literacy development.

Rewrite the original text into THREE reading levels.

Follow ALL constraints strictly.

--------------------------------
LEVEL 1: Kindergarten (age 3-5)
--------------------------------

Vocabulary level:
- Use only very common everyday words.
- Target vocabulary difficulty: CEFR Pre-A1 / A1.
- Prefer concrete words that represent visible things (dog, tree, house).
- Avoid abstract words, technical terms, and rare vocabulary.
- Prefer short words (1-2 syllables)

Sentence structure:
- Each sentence MUST contain 3-8 words.
- Use only simple sentences.
- No subordinate clauses.

Length:
- Maximum of 2 sentences.

Style:
- Can use simple storytelling or metaphors like a children's book.
- Use a friendly tone that is easy to understand for a kindergarten child.

--------------------------------
LEVEL 2: Primary School (age 6-11)
--------------------------------

Vocabulary level:
- Target vocabulary difficulty: CEFR A1-A2.
- Use common everyday vocabulary.
- Avoid rare or technical words.
- Words can have 1-3 syllables.

Sentence structure:
- Each sentence can contain at most 15 words.
- Simple conjunctions are allowed (and, because, so).

Length:
- Depends on the length of the original text, but no more than 4 sentences.

Content:
- Keep the main facts from the original text.

--------------------------------
LEVEL 3: Secondary School (age 12-16)
--------------------------------

Vocabulary level:
- Target vocabulary difficulty: CEFR A2-B2.
- More descriptive or abstract words are allowed.
- Avoid highly technical or domain-specific terminology.

Sentence structure:
- Each sentence can contain at most 25 words.
- Subordinate clauses are allowed.

Length:
- Depends on the length of the original text, but no more than 8 sentences.

Content:
- Preserve most of the key information from the original text.

--------------------------------
OUTPUT FORMAT
--------------------------------

Return ONLY valid JSON.

{{
"kindergarten": "...",
"primary_school": "...",
"secondary_school": "..."
}}

--------------------------------
ORIGINAL TEXT
--------------------------------

{original_text}
"""


def parse_model_json(content: str) -> dict[str, str]:
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else content
        content = content.replace("json", "", 1).strip()
    parsed = json.loads(content)
    required = ("kindergarten", "primary_school", "secondary_school")
    missing = [field for field in required if field not in parsed]
    if missing:
        raise ValueError(f"Model response missing fields: {missing}")
    return {field: str(parsed[field]) for field in required}


def call_llm(
    original_text: str,
    *,
    api_key: str,
    model: str,
    temperature: float,
    retry_limit: int,
) -> dict[str, str] | None:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(original_text)}],
        "temperature": temperature,
    }

    for attempt in range(1, retry_limit + 1):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return parse_model_json(content)
        except Exception as exc:  # noqa: BLE001
            print(f"Retry {attempt}/{retry_limit}: {exc}")
            time.sleep(2)

    return None


def count_existing_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def generate_dataset(
    *,
    input_path: Path,
    output_path: Path,
    api_key: str,
    model: str,
    temperature: float,
    retry_limit: int,
    limit: int | None,
) -> None:
    processed = count_existing_rows(output_path)
    print(f"Already processed: {processed}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with input_path.open("r", encoding="utf-8") as input_file:
        with output_path.open("a", encoding="utf-8") as output_file:
            for idx, line in enumerate(input_file):
                if idx < processed:
                    continue
                if limit is not None and written >= limit:
                    break

                item = json.loads(line)
                original_text = item["original_text"]
                result = call_llm(
                    original_text,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    retry_limit=retry_limit,
                )
                if result is None:
                    continue

                output_item = {
                    "original_text": original_text,
                    "simplified_kindergarten": result["kindergarten"],
                    "simplified_primary_school": result["primary_school"],
                    "simplified_secondary_school": result["secondary_school"],
                    "original_word_count": word_count(original_text),
                    "kindergarten_word_count": word_count(result["kindergarten"]),
                    "primary_school_word_count": word_count(result["primary_school"]),
                    "secondary_school_word_count": word_count(result["secondary_school"]),
                }
                output_file.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                output_file.flush()
                written += 1
                print(f"Processed: {idx + 1}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate age-level simplification data.")
    parser.add_argument("--input", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output_simplified.jsonl"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--retry-limit", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing OpenRouter API key. Set OPENROUTER_API_KEY or pass --api-key.")
    generate_dataset(
        input_path=args.input,
        output_path=args.output,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        retry_limit=args.retry_limit,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
