import argparse
import csv
import json
import random
import sys
from pathlib import Path


FIELDS = [
    "original_text",
    "simplified_kindergarten",
    "simplified_primary_school",
    "simplified_secondary_school",
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_sample(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def sample_predictions(input_path: Path, output_path: Path, sample_count: int) -> None:
    records = load_jsonl(input_path)
    selected = random.sample(records, min(sample_count, len(records)))
    write_sample(selected, output_path)
    print(f"Saved {len(selected)} samples to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample JSONL predictions and write a smaller CSV for human evaluation."
    )
    parser.add_argument("input", type=Path, help="Path to the source JSONL file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sampled_output.csv"),
        help="Path for the output CSV file.",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=200,
        help="Number of random samples to extract.",
    )
    parser.add_argument("-s", "--seed", type=int, help="Optional seed for reproducible sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    try:
        sample_predictions(args.input, args.output, args.count)
    except FileNotFoundError:
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
