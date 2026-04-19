import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_jsonl(
    input_file: Path,
    train_output: Path,
    validation_output: Path,
    validation_size: int,
    seed: int,
) -> None:
    rows = read_jsonl(input_file)
    rng = random.Random(seed)
    rng.shuffle(rows)

    validation_rows = rows[:validation_size]
    train_rows = rows[validation_size:]

    write_jsonl(validation_output, validation_rows)
    write_jsonl(train_output, train_rows)
    print(
        f"Saved {len(train_rows)} training rows to {train_output} "
        f"and {len(validation_rows)} validation rows to {validation_output}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a JSONL dataset into train and validation files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output_simplified.jsonl"),
        help="Input JSONL file.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("dataset_train.jsonl"),
        help="Output path for training rows.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("dataset_validation.jsonl"),
        help="Output path for validation rows.",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=1000,
        help="Number of rows to put in validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_jsonl(
        input_file=args.input,
        train_output=args.train_output,
        validation_output=args.validation_output,
        validation_size=args.validation_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
