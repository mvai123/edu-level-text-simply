from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


Backend = Literal["llama3", "gemini", "identity", "summarization"]


@dataclass(frozen=True)
class RunOptions:
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 256


def route_and_run(
    backend: Backend,
    records: List[Dict[str, Any]],
    *,
    options: RunOptions,
) -> List[Dict[str, Any]]:
    if backend == "identity":
        from baselines.identity.runner import run_identity

        return run_identity(records, options=options)
    if backend == "summarization":
        from baselines.summarization.runner import run_summarization

        return run_summarization(records, options=options)
    if backend == "llama3":
        from baselines.llama3.runner import run_llama3

        return run_llama3(records, options=options)
    if backend == "gemini":
        from baselines.gemini.runner import run_gemini

        return run_gemini(records, options=options)
    raise ValueError(
        f"Unsupported backend: {backend!r}. "
        "Expected 'identity', 'summarization', 'llama3', or 'gemini'."
    )
