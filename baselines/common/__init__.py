from __future__ import annotations

from typing import Any, Dict

AGE_PROMPTS: Dict[str, str] = {
    "kindergarten": (
        "You are simplifying text for a kindergarten reader.\n"
        "- Use extremely simple vocabulary and very short sentences.\n"
        "- Do NOT change any facts or properties described in the text.\n"
        "- Do NOT flip meanings.\n"
        "- Only rephrase, split, or reorder to make it easier to understand.\n"
        "- Output ONLY the simplified text. Do not add explanations, notes, or headings.\n\n"
        "Original text:\n"
        "{text}\n\n"
        "Simplified (same meaning, much simpler words):"
    ),
    "primary": (
        "You are simplifying text for a primary school reader.\n"
        "- Use simple vocabulary and clear sentences.\n"
        "- Keep all important details and do NOT change any factual content.\n"
        "- Do NOT add or remove information; only rephrase and clarify.\n"
        "- Output ONLY the simplified text. Do not add explanations, notes, or headings.\n\n"
        "Original text:\n"
        "{text}\n\n"
        "Simplified (same meaning, clearer wording):"
    ),
    "secondary": (
        "You are simplifying text for a secondary school reader.\n"
        "- Improve clarity and structure while preserving nuance and technical content.\n"
        "- Do NOT change any factual statements.\n"
        "- Do NOT oversimplify by deleting important information.\n"
        "- Output ONLY the simplified text. Do not add explanations, notes, or headings.\n\n"
        "Original text:\n"
        "{text}\n\n"
        "Simplified (same meaning, clearer):"
    ),
}


def _extract_text(rec: Dict[str, Any]) -> str:
    text = rec.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Each record must have a non-empty 'text' field.")
    return text


def _get_age_band(rec: Dict[str, Any], *, options: object | None = None) -> str:
    v = rec.get("age_band")
    if not isinstance(v, str) or not v.strip():
        raise ValueError("Each record must have a non-empty 'age_band' field.")
    band = v.strip().lower()
    allowed = {"kindergarten", "primary", "secondary"}
    if band not in allowed:
        raise ValueError(f"Unsupported age_band: {band!r}. Expected one of {sorted(allowed)!r}.")
    return band


def _build_prompt(text: str, age_band: str) -> str:
    key = (age_band or "kindergarten").lower()
    template = AGE_PROMPTS.get(key, AGE_PROMPTS["kindergarten"])
    return template.format(text=text)
