from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import google.generativeai as genai

from baselines.common import _build_prompt, _extract_text, _get_age_band
from baselines.wrapper.router import RunOptions

def _configure_client() -> None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
    genai.configure(api_key=api_key)


def _call_gemini(prompt: str, *, model_name: str, temperature: float, max_tokens: int) -> str:
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    text = getattr(resp, "text", None)
    if not isinstance(text, str):
        raise RuntimeError("Gemini returned empty response text.")
    return text.strip()


def run_gemini(records: List[Dict[str, Any]], *, options: RunOptions) -> List[Dict[str, Any]]:
    _configure_client()
    model_name = options.model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    out: List[Dict[str, Any]] = []
    for rec in records:
        text = _extract_text(rec)
        band = _get_age_band(rec, options=options)
        prompt = _build_prompt(text, band)

        # Minimal retry for transient errors
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                simpl = _call_gemini(
                    prompt,
                    model_name=model_name,
                    temperature=options.temperature,
                    max_tokens=options.max_tokens,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.8 * (2**attempt))
        if last_err is not None:
            raise RuntimeError(f"Gemini call failed after retries: {last_err}") from last_err

        new_rec = dict(rec)
        new_rec.setdefault("source", text)
        new_rec["simplified_gemini"] = simpl
        out.append(new_rec)
    return out
