"""Round-trip translation baseline using English/French Helsinki-NLP models."""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import pipeline


MODELS = {
    "en-fr": "weights/opus_mt_en_fr",
    "fr-en": "weights/opus_mt_fr_en",
}

DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 512


def build_pipelines(batch_size: int, max_length: int) -> dict:
    """Load the two translation models on GPU when one is available."""
    device = 0 if torch.cuda.is_available() else -1
    device_label = f"cuda:{device}" if device >= 0 else "cpu"
    print(f"  Device : {device_label}", flush=True)

    pipes = {}
    for key, model_id in MODELS.items():
        print(f"  [load] {key} -> {model_id}", flush=True)
        pipes[key] = pipeline(
            "translation",
            model=model_id,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    return pipes


def translate_batch(texts: list[str], pipe) -> list[str]:
    """Translate a list of texts through a Hugging Face pipeline."""
    results = pipe(texts)
    return [r["translation_text"] for r in results]


def build_all_outputs(
    originals: list[str],
    pipes: dict,
    verbose: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Run one, two, and three round trips for the three reading levels."""
    n = len(originals)

    def step(texts: list[str], key: str, label: str) -> list[str]:
        if verbose:
            print(f"  [{label}] {key} - {n} texts ...", flush=True)
        return translate_batch(texts, pipes[key])

    fr_1 = step(originals, "en-fr", "pass 1/6")
    en_1 = step(fr_1, "fr-en", "pass 2/6")

    fr_2 = step(en_1, "en-fr", "pass 3/6")
    en_2 = step(fr_2, "fr-en", "pass 4/6")

    fr_3 = step(en_2, "en-fr", "pass 5/6")
    en_3 = step(fr_3, "fr-en", "pass 6/6")

    return en_1, en_2, en_3


def load_jsonl(path: str, text_col: str) -> list[str]:
    """Read the requested text column from a JSONL file."""
    texts = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [warn] line {lineno}: invalid JSON; skipping ({exc})")
                continue
            if text_col not in record:
                print(f"  [warn] line {lineno}: missing column '{text_col}'; skipping")
                continue
            texts.append(str(record[text_col]))
    return texts


def write_csv(
    originals: list[str],
    secondary: list[str],
    primary: list[str],
    kindergarten: list[str],
    output_path: str,
) -> None:
    fieldnames = [
        "original_text",
        "simplified_kindergarten",
        "simplified_primary_school",
        "simplified_secondary_school",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for orig, sec, pri, kin in zip(originals, secondary, primary, kindergarten):
            writer.writerow(
                {
                    "original_text": orig,
                    "simplified_kindergarten": kin,
                    "simplified_primary_school": pri,
                    "simplified_secondary_school": sec,
                }
            )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round-trip translation text simplifier (HuggingFace, batched)"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input JSONL file.")
    parser.add_argument(
        "--output",
        "-o",
        default="output.csv",
        help="Path to output CSV file (default: output.csv).",
    )
    parser.add_argument(
        "--text_col",
        "-c",
        default="text",
        help="Text column name in JSONL (default: 'text').",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Pipeline batch size (default: {DEFAULT_BATCH_SIZE}). Increase for more GPU VRAM.",
    )
    parser.add_argument(
        "--max_length",
        "-m",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Max tokens per translation (default: {DEFAULT_MAX_LENGTH}).",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Process only the first N rows (for testing).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    if not Path(args.input).exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Round-trip translation simplifier [batched]")
    print(f"{'=' * 60}")
    print(f"Input      : {args.input}")
    print(f"Output     : {args.output}")
    print(f"Column     : {args.text_col}")
    print(f"Batch size : {args.batch_size}")
    print()

    print("Loading JSONL ...")
    texts = load_jsonl(args.input, args.text_col)
    if args.limit:
        texts = texts[: args.limit]
    print(f"  -> {len(texts)} text(s) loaded.\n")

    if not texts:
        print("No texts to process. Exiting.")
        sys.exit(0)

    print("Loading translation models ...")
    pipes = build_pipelines(args.batch_size, args.max_length)
    print()

    print("Translating ...")
    secondary, primary, kindergarten = build_all_outputs(texts, pipes)
    print()

    write_csv(texts, secondary, primary, kindergarten, args.output)
    print(f"Done. Results written to: {args.output}")


if __name__ == "__main__":
    main()
