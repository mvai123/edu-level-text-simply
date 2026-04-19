from __future__ import annotations

from typing import Any, Dict, List

from baselines.common import _extract_text
from baselines.wrapper.router import RunOptions

def run_identity(records: List[Dict[str, Any]], *, options: RunOptions) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        text = _extract_text(rec)
        new_rec = dict(rec)
        new_rec.setdefault("source", text)
        new_rec["simplified_identity"] = text
        out.append(new_rec)
    return out
