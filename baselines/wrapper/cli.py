from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from baselines.wrapper.router import RunOptions, route_and_run


def _load_records(path: Optional[str]) -> List[Dict[str, Any]]:
    if path is None or path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise ValueError(f"Input record {i} is not an object.")
    return data  # type: ignore[return-value]


def _write_records(path: Optional[str], records: List[Dict[str, Any]]) -> None:
    out = json.dumps(records, ensure_ascii=False, indent=2)
    if path is None or path == "-":
        sys.stdout.write(out)
        sys.stdout.write("\n")
        return
    Path(path).write_text(out + "\n", encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wrapper CLI routing to identity, summarization, llama3 or gemini.")
    p.add_argument("--backend", required=True, choices=["identity", "summarization", "llama3", "gemini"])
    p.add_argument("--input", default="-", help="Input JSON path (or '-' for stdin).")
    p.add_argument("--output", default="-", help="Output JSON path (or '-' for stdout).")

    p.add_argument("--model", default=None, help="Backend model override (optional).")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=256)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    records = _load_records(args.input)

    options = RunOptions(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    out_records = route_and_run(args.backend, records, options=options)
    _write_records(args.output, out_records)


if __name__ == "__main__":
    main()
