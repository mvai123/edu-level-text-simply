from __future__ import annotations

from typing import Any, Callable, Dict, List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from baselines.common import _extract_text
from baselines.wrapper.router import RunOptions


def _build_summarizer(model_name: str, max_new_tokens: int) -> Callable[[str], str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(text: str) -> str:
        enc = tokenizer("summarize: " + text, return_tensors="pt", truncation=True)
        out = model.generate(**enc, max_new_tokens=max_new_tokens)
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()

    return summarize


def run_summarization(records: List[Dict[str, Any]], *, options: RunOptions) -> List[Dict[str, Any]]:
    model_name = options.model or "t5-small"
    summarizer = _build_summarizer(model_name, max_new_tokens=options.max_tokens)

    out: List[Dict[str, Any]] = []
    for rec in records:
        text = _extract_text(rec)
        summary = summarizer(text)

        new_rec = dict(rec)
        new_rec.setdefault("source", text)
        new_rec["simplified_summarization"] = summary
        out.append(new_rec)
    return out
