from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from baselines.common import _build_prompt, _extract_text, _get_age_band
from baselines.wrapper.router import RunOptions


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

def _ollama_generate(
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int = 300,
) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama generate failed ({r.status_code}): {r.text}")
    data = r.json()
    resp = data.get("response")
    if not isinstance(resp, str):
        raise RuntimeError(f"Unexpected Ollama response: {json.dumps(data)[:500]}")
    return resp.strip()


def _ollama_pull(model: str, timeout_s: int = 3600) -> None:
    url = f"{OLLAMA_BASE_URL}/api/pull"
    payload = {"name": model, "stream": False}
    r = requests.post(url, json=payload, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama pull failed ({r.status_code}): {r.text}")


def run_llama3(records: List[Dict[str, Any]], *, options: RunOptions) -> List[Dict[str, Any]]:
    # Default to a small model that fits typical Docker Desktop memory limits.
    # Override with --model or LLAMA3_MODEL if you have more memory.
    model = options.model or os.environ.get("LLAMA3_MODEL", "llama3.2:1b")

    # Best-effort: pull model if missing (no-op if already present)
    try:
        _ollama_pull(model)
    except Exception:
        # If pull fails (e.g. no network), still attempt generate; caller gets clear error then.
        pass

    out: List[Dict[str, Any]] = []
    for rec in records:
        text = _extract_text(rec)
        band = _get_age_band(rec, options=options)
        prompt = _build_prompt(text, band)
        simpl = _ollama_generate(
            model=model,
            prompt=prompt,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
        )
        new_rec = dict(rec)
        new_rec.setdefault("source", text)
        new_rec["simplified_llama3"] = simpl
        out.append(new_rec)
    return out


def wait_for_ollama(max_wait_s: int = 30) -> None:
    url = f"{OLLAMA_BASE_URL}/api/tags"
    start = time.time()
    while time.time() - start < max_wait_s:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Ollama not reachable at {OLLAMA_BASE_URL}. Set OLLAMA_BASE_URL or start Ollama.")
