"""Fine-tune a FLAN-T5 model for age-specific text simplification."""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import yaml
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("train_t5")

AGE_LEVELS = ("kindergarten", "primary_school", "secondary_school")


@dataclass
class ModelConfig:
    name_or_path: str = "google/flan-t5-base"
    max_input_length: int = 512
    max_target_length: int = 128


@dataclass
class DataConfig:
    train_file: str = "data/train.jsonl"
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    transformation_fn: Optional[str] = None
    input_column: str = "input"
    output_column: str = "output"
    input_prefix: str = ""
    val_split_ratio: float = 0.1


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/t5-finetune"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    seed: int = 42


@dataclass
class EvaluationConfig:
    strategy: str = "epoch"
    eval_steps: int = 500
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


@dataclass
class LoggingConfig:
    report_to: str = "none"
    logging_steps: int = 50
    run_name: str = "t5-run"


@dataclass
class GenerationConfig:
    num_beams: int = 4
    early_stopping: bool = True
    compute_metrics: bool = True


@dataclass
class FullConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def _apply_overrides(raw: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply key=value overrides to the raw config dict (dotted keys supported).

    Example: training.learning_rate=1e-4 sets raw["training"]["learning_rate"].
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must be in key=value format.")
        key_path, value_str = override.split("=", 1)
        keys = key_path.strip().split(".")
        node = raw
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        parsed_value = yaml.safe_load(value_str)
        node[keys[-1]] = parsed_value
        log.info("Override applied: %s = %r", key_path, parsed_value)
    return raw


def _populate_dataclass(dc_instance, raw: Dict[str, Any]) -> None:
    """Fill a dataclass in-place from a dict, ignoring unknown keys."""
    for key, value in raw.items():
        if hasattr(dc_instance, key):
            expected_type = type(getattr(dc_instance, key))
            try:
                if expected_type in (int, float, bool, str) and not isinstance(value, expected_type):
                    value = expected_type(value)
            except (ValueError, TypeError):
                pass
            setattr(dc_instance, key, value)
        else:
            log.warning("Unknown config key ignored: %s", key)


def load_config(config_path: str, overrides: Optional[List[str]] = None) -> FullConfig:
    """Parse YAML config, apply CLI overrides, and return a typed FullConfig."""
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
    for section_name, dc_instance in section_map.items():
        if section_name in raw:
            _populate_dataclass(dc_instance, raw[section_name])

    log.info("Config loaded from %s", config_path)
    return cfg


def _load_jsonl(path: str) -> List[Dict[str, str]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _transform_data(records: List[Dict[str, Any]], cfg: DataConfig) -> List[Dict[str, Any]]:
    """Expand each source row into one example per target level."""
    if cfg.transformation_fn is None:
        return records

    data: List[Dict[str, Any]] = []
    for record in records:
        source = record["original_text"]
        for age in AGE_LEVELS:
            prompt = cfg.transformation_fn.replace("[age]", age).replace("[input]", source)
            data.append(
                {
                    cfg.input_column: prompt,
                    cfg.output_column: record[f"simplified_{age}"],
                    "original_text": source,
                    "target_level": age,
                }
            )
    return data


def load_datasets(cfg: DataConfig) -> DatasetDict:
    """Load train / val (/ test) splits from JSONL files."""
    train_records = _load_jsonl(cfg.train_file)
    if cfg.transformation_fn is not None:
        train_records = _transform_data(train_records, cfg)
    train_ds = Dataset.from_list(train_records)

    if cfg.val_file:
        val_records = _load_jsonl(cfg.val_file)
        if cfg.transformation_fn is not None:
            val_records = _transform_data(val_records, cfg)
        val_ds = Dataset.from_list(val_records)
    else:
        log.info(
            "No val_file specified; splitting %.0f%% of train for validation.",
            cfg.val_split_ratio * 100,
        )
        split = train_ds.train_test_split(test_size=cfg.val_split_ratio, seed=42)
        train_ds, val_ds = split["train"], split["test"]

    ds_dict: Dict[str, Dataset] = {"train": train_ds, "validation": val_ds}

    if cfg.test_file:
        test_records = _load_jsonl(cfg.test_file)
        if cfg.transformation_fn is not None:
            test_records = _transform_data(test_records, cfg)
        ds_dict["test"] = Dataset.from_list(test_records)

    log.info(
        "Dataset sizes - train: %d | val: %d%s",
        len(ds_dict["train"]),
        len(ds_dict["validation"]),
        f" | test: {len(ds_dict['test'])}" if "test" in ds_dict else "",
    )
    return DatasetDict(ds_dict)


def build_preprocess_fn(tokenizer, cfg_model: ModelConfig, cfg_data: DataConfig):
    """Return a batched map function that tokenizes inputs and targets."""

    prefix = cfg_data.input_prefix or ""

    def preprocess(batch):
        inputs = [prefix + str(x) for x in batch[cfg_data.input_column]]
        targets = [str(x) for x in batch[cfg_data.output_column]]

        model_inputs = tokenizer(
            inputs,
            max_length=cfg_model.max_input_length,
            padding=False,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=cfg_model.max_target_length,
                padding=False,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def build_compute_metrics(tokenizer, gen_cfg: GenerationConfig):
    """Return compute_metrics function for Seq2SeqTrainer."""
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        rouge_result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        bleu_result = bleu.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels],
        )

        metrics = {
            "rouge1": round(rouge_result["rouge1"], 4),
            "rouge2": round(rouge_result["rouge2"], 4),
            "rougeL": round(rouge_result["rougeL"], 4),
            "bleu": round(bleu_result["score"], 4),
        }
        pred_lengths = [np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]
        metrics["gen_len"] = round(np.mean(pred_lengths), 2)
        return metrics

    return compute_metrics


def build_training_args(
    cfg_train: TrainingConfig,
    cfg_eval: EvaluationConfig,
    cfg_log: LoggingConfig,
    cfg_gen: GenerationConfig,
) -> Seq2SeqTrainingArguments:
    """Combine all config sections into Seq2SeqTrainingArguments."""
    return Seq2SeqTrainingArguments(
        output_dir=cfg_train.output_dir,
        num_train_epochs=cfg_train.num_train_epochs,
        per_device_train_batch_size=cfg_train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg_train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg_train.gradient_accumulation_steps,
        learning_rate=cfg_train.learning_rate,
        lr_scheduler_type=cfg_train.lr_scheduler_type,
        warmup_ratio=cfg_train.warmup_ratio,
        weight_decay=cfg_train.weight_decay,
        max_grad_norm=cfg_train.max_grad_norm,
        fp16=cfg_train.fp16,
        bf16=cfg_train.bf16,
        eval_strategy=cfg_eval.strategy,
        eval_steps=cfg_eval.eval_steps if cfg_eval.strategy == "steps" else None,
        save_strategy=cfg_eval.save_strategy,
        save_total_limit=cfg_eval.save_total_limit,
        load_best_model_at_end=cfg_eval.load_best_model_at_end,
        metric_for_best_model=cfg_eval.metric_for_best_model,
        greater_is_better=cfg_eval.greater_is_better,
        logging_steps=cfg_log.logging_steps,
        report_to=cfg_log.report_to,
        run_name=cfg_log.run_name,
        predict_with_generate=cfg_gen.compute_metrics,
        generation_num_beams=cfg_gen.num_beams,
        seed=cfg_train.seed,
        dataloader_pin_memory=True,
        group_by_length=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a T5-style seq2seq model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. training.learning_rate=5e-5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg.training.seed)
    os.makedirs(cfg.training.output_dir, exist_ok=True)

    log.info("=== Configuration ===")
    for section_name in ("model", "data", "training", "evaluation", "logging", "generation"):
        section = getattr(cfg, section_name)
        log.info("[%s] %s", section_name, vars(section))

    log.info("Loading tokenizer and model: %s", cfg.model.name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name_or_path)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("Model loaded - %.1fM parameters", num_params)

    raw_datasets = load_datasets(cfg.data)

    preprocess_fn = build_preprocess_fn(tokenizer, cfg.model, cfg.data)
    tokenized_datasets = raw_datasets.map(
        preprocess_fn,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    compute_metrics = (
        build_compute_metrics(tokenizer, cfg.generation)
        if cfg.generation.compute_metrics
        else None
    )

    training_args = build_training_args(
        cfg.training, cfg.evaluation, cfg.logging, cfg.generation
    )

    callbacks = []
    if cfg.evaluation.load_best_model_at_end:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    log.info("Starting training...")
    train_result = trainer.train()

    trainer.save_model()
    trainer.save_state()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    log.info("Training complete. Model saved to %s", cfg.training.output_dir)

    if "test" in tokenized_datasets:
        log.info("Running evaluation on test set...")
        test_metrics = trainer.evaluate(
            eval_dataset=tokenized_datasets["test"],
            metric_key_prefix="test",
        )
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    resolved_cfg_path = Path(cfg.training.output_dir) / "resolved_config.yaml"
    resolved: Dict[str, Any] = {}
    for section_name in ("model", "data", "training", "evaluation", "logging", "generation"):
        resolved[section_name] = vars(getattr(cfg, section_name))
    with open(resolved_cfg_path, "w") as f:
        yaml.dump(resolved, f, default_flow_style=False, sort_keys=False)
    log.info("Resolved config saved to %s", resolved_cfg_path)


if __name__ == "__main__":
    main()
