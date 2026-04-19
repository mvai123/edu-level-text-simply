"""Generate predictions from a fine-tuned FLAN-T5 simplification model."""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from train_model import (
    AGE_LEVELS,
    DataConfig,
    FullConfig,
    GenerationConfig,
    ModelConfig,
    _apply_overrides,
    _load_jsonl,
    _populate_dataclass,
    _transform_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("infer_t5")


@dataclass
class InferConfig:
    """Options used only while generating predictions."""
    batch_size: int = 16
    decode_strategy: str = "beam"
    device: str = "auto"
    output_dir: str = "predictions"


def load_config_with_infer(
    config_path: str,
    overrides: Optional[List[str]] = None,
) -> tuple[FullConfig, InferConfig]:
    """Load the shared training config plus the optional infer section."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    if overrides:
        raw = _apply_overrides(raw, overrides)

    cfg = FullConfig()
    section_map = {
        "model": cfg.model,
        "data": cfg.data,
        "training": cfg.training,
        "evaluation": cfg.evaluation,
        "logging": cfg.logging,
        "generation": cfg.generation,
    }
    for name, dc in section_map.items():
        if name in raw:
            _populate_dataclass(dc, raw[name])

    infer_cfg = InferConfig()
    if "infer" in raw:
        _populate_dataclass(infer_cfg, raw["infer"])

    log.info("Config loaded from %s", config_path)
    return cfg, infer_cfg


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def load_val_records(cfg_data: DataConfig) -> List[Dict[str, Any]]:
    """Load validation records and apply the same prompt expansion as training."""
    if cfg_data.val_file:
        records = _load_jsonl(cfg_data.val_file)
        if cfg_data.transformation_fn is not None:
            records = _transform_data(records, cfg_data)
        log.info("Loaded %d validation records from %s", len(records), cfg_data.val_file)
    else:
        log.warning(
            "data.val_file is not set; splitting %.0f%% of train_file (%s) for validation.",
            cfg_data.val_split_ratio * 100,
            cfg_data.train_file,
        )
        all_records = _load_jsonl(cfg_data.train_file)
        if cfg_data.transformation_fn is not None:
            all_records = _transform_data(all_records, cfg_data)
        split_idx = int(len(all_records) * (1 - cfg_data.val_split_ratio))
        records = all_records[split_idx:]
        log.info("Using %d records as the validation split.", len(records))
    return records


def tokenize_batch(
    batch_texts: List[str],
    tokenizer,
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        batch_texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def build_generate_kwargs(
    gen_cfg: GenerationConfig,
    infer_cfg: InferConfig,
    tokenizer,
) -> Dict[str, Any]:
    """Build the arguments passed to model.generate."""
    strategy = infer_cfg.decode_strategy.lower()
    if strategy == "greedy":
        num_beams = 1
        do_sample = False
        log.info("Decode strategy: greedy (num_beams=1)")
    elif strategy == "beam":
        num_beams = max(gen_cfg.num_beams, 1)
        do_sample = False
        log.info("Decode strategy: beam search (num_beams=%d)", num_beams)
    else:
        raise ValueError(
            f"Unknown decode_strategy '{strategy}'. Choose 'greedy' or 'beam'."
        )

    kwargs: Dict[str, Any] = {
        "num_beams": num_beams,
        "do_sample": do_sample,
        "num_return_sequences": 1,
        "early_stopping": gen_cfg.early_stopping if num_beams > 1 else False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    return kwargs


@torch.inference_mode()
def generate_responses(
    records: List[Dict[str, Any]],
    model,
    tokenizer,
    cfg_model: ModelConfig,
    cfg_data: DataConfig,
    gen_cfg: GenerationConfig,
    infer_cfg: InferConfig,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Run batched generation over all records and return enriched dicts."""

    prefix = cfg_data.input_prefix or ""
    input_col = cfg_data.input_column
    output_col = cfg_data.output_column
    generate_kwargs = build_generate_kwargs(gen_cfg, infer_cfg, tokenizer)

    results: List[Dict[str, Any]] = []
    total_batches = (len(records) + infer_cfg.batch_size - 1) // infer_cfg.batch_size
    t0 = time.perf_counter()

    for batch_idx in tqdm(
        range(0, len(records), infer_cfg.batch_size),
        total=total_batches,
        desc="Generating",
        unit="batch",
    ):
        batch = records[batch_idx : batch_idx + infer_cfg.batch_size]

        raw_inputs = [str(r[input_col]) for r in batch]
        prefixed = [prefix + t for t in raw_inputs]
        enc = tokenize_batch(prefixed, tokenizer, cfg_model.max_input_length, device)

        output_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=cfg_model.max_target_length,
            **generate_kwargs,
        )

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for record, raw_in, prediction in zip(batch, raw_inputs, decoded):
            out: Dict[str, Any] = {
                input_col: raw_in,
                "prediction": prediction.strip(),
            }
            if output_col in record:
                out[output_col] = record[output_col]
            for key, val in record.items():
                if key not in out:
                    out[key] = val
            results.append(out)

    elapsed = time.perf_counter() - t0
    log.info(
        "Generated %d responses in %.1fs (%.1f samples/s)",
        len(results),
        elapsed,
        len(results) / elapsed if elapsed else 0.0,
    )
    return results


def _source_from_prompt(prompt: str, level: str, cfg_data: DataConfig) -> str:
    """Recover the source text from a prompt when metadata is unavailable."""
    template = cfg_data.transformation_fn or ""
    if "[input]" not in template:
        return prompt

    before_input, after_input = template.split("[input]", 1)
    before_input = before_input.replace("[age]", level)
    after_input = after_input.replace("[age]", level)

    source = prompt
    if source.startswith(before_input):
        source = source[len(before_input) :]
    if after_input and source.endswith(after_input):
        source = source[: -len(after_input)]
    return source


def _reverse_transform(records: List[Dict[str, Any]], cfg_data: DataConfig) -> List[Dict[str, Any]]:
    """Collapse one row per level back into the evaluator's wide JSONL format."""
    rows_by_source: Dict[str, Dict[str, Any]] = {}
    source_order: List[str] = []

    for idx, record in enumerate(records):
        level = str(record.get("target_level") or AGE_LEVELS[idx % len(AGE_LEVELS)])
        source = record.get("original_text")
        if source is None:
            prompt = str(record.get(cfg_data.input_column, ""))
            source = _source_from_prompt(prompt, level, cfg_data)

        if source not in rows_by_source:
            rows_by_source[source] = {
                "original_text": source,
                "simplified_kindergarten": "",
                "simplified_primary_school": "",
                "simplified_secondary_school": "",
            }
            source_order.append(source)

        rows_by_source[source][f"simplified_{level}"] = record.get("prediction", "")

    return [rows_by_source[source] for source in source_order]


def save_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Saved %d records to %s", len(records), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained T5 model using the training config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file (same file used during training).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override any config value using dotted key paths. Examples:\n"
            "  data.val_file=data/my_val.jsonl\n"
            "  infer.batch_size=32\n"
            "  infer.decode_strategy=greedy\n"
            "  generation.num_beams=6"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg, infer_cfg = load_config_with_infer(args.config, overrides=args.override)

    log.info("=== Inference Configuration ===")
    for name in ("model", "data", "training", "generation", "logging"):
        log.info("[%s] %s", name, vars(getattr(cfg, name)))
    log.info("[infer] %s", vars(infer_cfg))

    device = resolve_device(infer_cfg.device)
    log.info("Using device: %s", device)

    model_path = cfg.training.output_dir
    log.info("Loading model from %s...", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("Model loaded - %.1fM parameters", num_params)

    val_records = load_val_records(cfg.data)

    if not val_records:
        log.error("No validation records found. Check data.val_file in your config.")
        sys.exit(1)

    results = generate_responses(
        records=val_records,
        model=model,
        tokenizer=tokenizer,
        cfg_model=cfg.model,
        cfg_data=cfg.data,
        gen_cfg=cfg.generation,
        infer_cfg=infer_cfg,
        device=device,
    )

    run_name = cfg.logging.run_name
    output_path = Path(infer_cfg.output_dir) / f"{run_name}.jsonl"
    final_results = _reverse_transform(results, cfg.data)
    save_jsonl(final_results, output_path)


if __name__ == "__main__":
    main()
