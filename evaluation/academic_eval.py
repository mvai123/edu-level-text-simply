from __future__ import annotations

import argparse
import csv
import json
import math
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
import textstat
import torch
import torch.nn.functional as F
from bert_score import BERTScorer
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)


PREDICTION_COLUMNS = {
    "kindergarten": "simplified_kindergarten",
    "primary": "simplified_primary_school",
    "secondary": "simplified_secondary_school",
}

PREDICTION_COLUMN_ALIASES = {
    "simplified_kindergarten": ["simplified_kindergarten"],
    "simplified_primary_school": ["simplified_primary_school"],
    "simplified_secondary_school": ["simplified_secondary_school"],
}

FKGL_TARGET_BANDS = {
    "kindergarten": (-math.inf, 1.0),
    "primary": (1.0, 5.0),
    "secondary": (5.0, 9.0),
}

SUMMARY_METRIC_COLUMNS = [
    "empty_prediction",
    "adequacy_nonref_nonllm_entailment",
    "adequacy_ref_nonllm_bertscore",
    "simplicity_nonref_nonllm_fkgl",
    "simplicity_nonref_nonllm_target_band_hit",
    "simplicity_ref_nonllm_sari",
    "fluency_nonref_nonllm_ppl",
    "fluency_ref_nonllm_ppl_gap",
]


@dataclass
class ExampleRecord:
    system: str
    row_id: int
    level: str
    source: str
    prediction: str
    reference: str


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def sentence_sari(source: str, prediction: str, reference: str, max_n: int = 4) -> float:
    source_tokens = _normalize_text(source).lower().split()
    pred_tokens = _normalize_text(prediction).lower().split()
    ref_tokens = _normalize_text(reference).lower().split()

    add_scores: List[float] = []
    keep_scores: List[float] = []
    del_scores: List[float] = []

    for n in range(1, max_n + 1):
        src = _ngrams(source_tokens, n)
        pred = _ngrams(pred_tokens, n)
        ref = _ngrams(ref_tokens, n)

        pred_add = pred - src
        ref_add = ref - src
        add_overlap = len(pred_add & ref_add)
        add_precision = add_overlap / len(pred_add) if pred_add else 1.0
        add_recall = add_overlap / len(ref_add) if ref_add else 1.0
        add_f1 = (
            2 * add_precision * add_recall / (add_precision + add_recall)
            if add_precision + add_recall > 0
            else 0.0
        )
        add_scores.append(add_f1)

        pred_keep = pred & src
        ref_keep = ref & src
        keep_overlap = len(pred_keep & ref_keep)
        keep_precision = keep_overlap / len(pred_keep) if pred_keep else 1.0
        keep_recall = keep_overlap / len(ref_keep) if ref_keep else 1.0
        keep_f1 = (
            2 * keep_precision * keep_recall / (keep_precision + keep_recall)
            if keep_precision + keep_recall > 0
            else 0.0
        )
        keep_scores.append(keep_f1)

        pred_del = src - pred
        ref_del = src - ref
        del_overlap = len(pred_del & ref_del)
        del_precision = del_overlap / len(pred_del) if pred_del else 1.0
        del_scores.append(del_precision)

    sari = (sum(add_scores) / max_n + sum(keep_scores) / max_n + sum(del_scores) / max_n) / 3
    return sari * 100.0


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def get_cuda_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    return float(props.total_memory) / (1024**3)


def resolve_auto_batch_sizes(device: torch.device) -> Dict[str, int]:
    if device.type != "cuda":
        return {
            "entailment": 8,
            "perplexity": 8,
            "bertscore": 16,
        }

    memory_gb = get_cuda_memory_gb(device)
    if memory_gb >= 80:
        return {
            "entailment": 64,
            "perplexity": 64,
            "bertscore": 128,
        }
    if memory_gb >= 40:
        return {
            "entailment": 32,
            "perplexity": 32,
            "bertscore": 64,
        }
    if memory_gb >= 20:
        return {
            "entailment": 16,
            "perplexity": 16,
            "bertscore": 32,
        }
    return {
        "entailment": 8,
        "perplexity": 8,
        "bertscore": 16,
    }


def resolve_cuda_amp_dtype(name: str, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda" or name == "none":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "auto":
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        return torch.bfloat16 if bf16_supported else torch.float16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def get_autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]) -> Any:
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def maybe_reduce_batch_size_on_oom(desc: str, device: torch.device, batch_size: int) -> int:
    if device.type != "cuda" or batch_size <= 1:
        raise
    new_batch_size = max(1, batch_size // 2)
    torch.cuda.empty_cache()
    tqdm.write(f"{desc}: CUDA OOM at batch_size={batch_size}; retrying with batch_size={new_batch_size}.")
    return new_batch_size


def score_unique_texts(
    texts: Sequence[str],
    *,
    score_many_fn: Callable[..., List[float]],
    batch_size: int,
) -> List[float]:
    unique_texts = list(dict.fromkeys(texts))
    unique_scores = score_many_fn(unique_texts, batch_size=batch_size)
    score_lookup = dict(zip(unique_texts, unique_scores))
    return [score_lookup[text] for text in texts]


def score_unique_pairs(
    left: Sequence[str],
    right: Sequence[str],
    *,
    score_many_fn: Callable[..., List[float]],
    batch_size: int,
) -> List[float]:
    if len(left) != len(right):
        raise ValueError("Pairwise score inputs must have the same length.")
    unique_pairs = list(dict.fromkeys(zip(left, right)))
    unique_left = [pair[0] for pair in unique_pairs]
    unique_right = [pair[1] for pair in unique_pairs]
    unique_scores = score_many_fn(unique_left, unique_right, batch_size=batch_size)
    score_lookup = dict(zip(unique_pairs, unique_scores))
    return [score_lookup[(lhs, rhs)] for lhs, rhs in zip(left, right)]


class GPT2PerplexityScorer:
    def __init__(
        self,
        model_name: str = "gpt2",
        *,
        device: Optional[torch.device] = None,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.device = device or get_torch_device()
        self.amp_dtype = amp_dtype if self.device.type == "cuda" else None
        self.model.to(self.device)

    def score(self, text: str) -> float:
        text = _normalize_text(text)
        if not text:
            return float("inf")
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)
        with torch.inference_mode():
            with get_autocast_context(self.device, self.amp_dtype):
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return float(torch.exp(loss).item())

    def score_many(self, texts: Sequence[str], batch_size: int = 8) -> List[float]:
        scores: List[float] = []
        effective_batch_size = max(1, batch_size)
        progress = tqdm(total=len(texts), desc="Perplexity", leave=False)
        try:
            start = 0
            while start < len(texts):
                stop = min(start + effective_batch_size, len(texts))
                batch_texts = [_normalize_text(text) or self.tokenizer.eos_token for text in texts[start:stop]]
                try:
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024,
                        padding=True,
                    ).to(self.device)
                    labels = inputs["input_ids"].clone()
                    labels[inputs["attention_mask"] == 0] = -100

                    with torch.inference_mode():
                        with get_autocast_context(self.device, self.amp_dtype):
                            outputs = self.model(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                            )

                    shift_logits = outputs.logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    token_losses = F.cross_entropy(
                        shift_logits.float().view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view(shift_labels.size())

                    valid_tokens = (shift_labels != -100).sum(dim=1).clamp_min(1)
                    loss_per_sequence = token_losses.sum(dim=1) / valid_tokens
                    scores.extend(float(torch.exp(loss).item()) for loss in loss_per_sequence.cpu())
                    progress.update(stop - start)
                    start = stop
                except torch.OutOfMemoryError:
                    effective_batch_size = maybe_reduce_batch_size_on_oom(
                        "Perplexity",
                        self.device,
                        effective_batch_size,
                    )
        finally:
            progress.close()
        return scores


class SourceEntailmentScorer:
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        *,
        device: Optional[torch.device] = None,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = device or get_torch_device()
        self.amp_dtype = amp_dtype if self.device.type == "cuda" else None
        self.model.to(self.device)

    def score(self, premise: str, hypothesis: str) -> float:
        premise = _normalize_text(premise)
        hypothesis = _normalize_text(hypothesis)
        if not premise or not hypothesis:
            return 0.0

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)
        with torch.inference_mode():
            with get_autocast_context(self.device, self.amp_dtype):
                logits = self.model(**inputs).logits
        probs = torch.softmax(logits.float(), dim=-1)[0].detach().cpu().tolist()

        label_map = {
            self.model.config.id2label[i].lower(): probs[i]
            for i in range(len(probs))
        }
        return float(label_map.get("entailment", probs[-1]))

    def score_many(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        batch_size: int = 8,
    ) -> List[float]:
        if len(premises) != len(hypotheses):
            raise ValueError("Premises and hypotheses must have the same length.")

        scores: List[float] = []
        effective_batch_size = max(1, batch_size)
        progress = tqdm(total=len(premises), desc="Entailment", leave=False)
        try:
            start = 0
            while start < len(premises):
                stop = min(start + effective_batch_size, len(premises))
                batch_premises = [_normalize_text(x) or " " for x in premises[start:stop]]
                batch_hypotheses = [_normalize_text(x) or " " for x in hypotheses[start:stop]]
                try:
                    inputs = self.tokenizer(
                        batch_premises,
                        batch_hypotheses,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024,
                        padding=True,
                    ).to(self.device)
                    with torch.inference_mode():
                        with get_autocast_context(self.device, self.amp_dtype):
                            logits = self.model(**inputs).logits

                    probs = torch.softmax(logits.float(), dim=-1).detach().cpu()
                    for row in probs:
                        label_map = {
                            self.model.config.id2label[i].lower(): float(row[i].item())
                            for i in range(len(row))
                        }
                        scores.append(float(label_map.get("entailment", row[-1].item())))
                    progress.update(stop - start)
                    start = stop
                except torch.OutOfMemoryError:
                    effective_batch_size = maybe_reduce_batch_size_on_oom(
                        "Entailment",
                        self.device,
                        effective_batch_size,
                    )
        finally:
            progress.close()
        return scores


class BatchedBERTScore:
    def __init__(self, model_type: str = "bert-base-uncased", *, device: Optional[torch.device] = None) -> None:
        self.device = str(device or get_torch_device())
        self.scorer = BERTScorer(lang="en", model_type=model_type, device=self.device)

    def score_pairs(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        batch_size: int = 16,
    ) -> List[float]:
        scores: List[float] = []
        effective_batch_size = max(1, batch_size)
        progress = tqdm(total=len(predictions), desc="BERTScore", leave=False)
        try:
            start = 0
            while start < len(predictions):
                stop = min(start + effective_batch_size, len(predictions))
                batch_preds = [str(x) for x in predictions[start:stop]]
                batch_refs = [str(x) for x in references[start:stop]]
                try:
                    with torch.inference_mode():
                        _, _, f1 = self.scorer.score(
                            batch_preds,
                            batch_refs,
                            batch_size=effective_batch_size,
                            verbose=False,
                        )
                    scores.extend(float(x) for x in f1.detach().cpu().tolist())
                    progress.update(stop - start)
                    start = stop
                except torch.OutOfMemoryError:
                    effective_batch_size = maybe_reduce_batch_size_on_oom(
                        "BERTScore",
                        torch.device(self.device),
                        effective_batch_size,
                    )
        finally:
            progress.close()
        return scores


def load_reference_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def build_reference_lookup(references: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(references):
        key = _normalize_text(row["original_text"])
        if key in lookup:
            raise ValueError(f"Duplicate source text found in validation references at row {idx}.")
        lookup[key] = row
    return lookup


def strip_bom_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in row.items():
        clean[key.lstrip("\ufeff")] = value
    return clean


def detect_prediction_files(paths: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for raw in paths:
        candidate = Path(raw)
        if candidate.is_dir():
            out.extend(sorted(candidate.glob("*.csv")))
            out.extend(sorted(candidate.glob("*.jsonl")))
        else:
            out.append(candidate)
    return sorted({p.resolve() for p in out})


def iter_prediction_rows(prediction_file: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if prediction_file.suffix.lower() == ".csv":
        with prediction_file.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            rows.extend(strip_bom_keys(raw_row) for raw_row in reader)
        return rows

    if prediction_file.suffix.lower() == ".jsonl":
        with prediction_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows.append(strip_bom_keys(json.loads(line)))
        return rows

    raise ValueError(f"Unsupported prediction file format: {prediction_file}")


def get_prediction_value(row: Dict[str, Any], canonical_column: str) -> str:
    for candidate in PREDICTION_COLUMN_ALIASES[canonical_column]:
        if candidate in row:
            return _normalize_text(row.get(candidate, ""))
    raise KeyError(f"Missing prediction column {canonical_column} (aliases: {PREDICTION_COLUMN_ALIASES[canonical_column]})")


def build_examples(
    *,
    prediction_files: Sequence[Path],
    references: Sequence[Dict[str, Any]],
) -> List[ExampleRecord]:
    examples: List[ExampleRecord] = []
    reference_lookup = build_reference_lookup(references)
    for prediction_file in prediction_files:
        system_name = prediction_file.stem
        rows = iter_prediction_rows(prediction_file)
        for idx, row in enumerate(rows):
            source = _normalize_text(row["original_text"])
            ref_row = reference_lookup.get(source)
            if ref_row is None:
                raise ValueError(
                    f"Could not find source text from {prediction_file.name} row {idx} in validation.jsonl"
                )

            for level, pred_col in PREDICTION_COLUMNS.items():
                examples.append(
                    ExampleRecord(
                        system=system_name,
                        row_id=idx,
                        level=level,
                        source=source,
                        prediction=get_prediction_value(row, pred_col),
                        reference=_normalize_text(ref_row[pred_col]),
                    )
                )
    return examples


def compute_target_band_hit(level: str, fkgl: float) -> int:
    lower_bound, upper_bound = FKGL_TARGET_BANDS[level]
    return int(lower_bound < fkgl <= upper_bound)


def compute_non_llm_metrics(
    examples: Sequence[ExampleRecord],
    *,
    bertscore_model: str,
    device: torch.device,
    entailment_batch_size: int,
    perplexity_batch_size: int,
    bertscore_batch_size: int,
    cuda_amp_dtype: Optional[torch.dtype],
) -> pd.DataFrame:
    entailment = SourceEntailmentScorer(device=device, amp_dtype=cuda_amp_dtype)
    bertscore = BatchedBERTScore(model_type=bertscore_model, device=device)
    ppl = GPT2PerplexityScorer(device=device, amp_dtype=cuda_amp_dtype)

    sources = [ex.source for ex in examples]
    predictions = [ex.prediction for ex in examples]
    references = [ex.reference for ex in examples]

    entailment_scores = score_unique_pairs(
        sources,
        predictions,
        score_many_fn=entailment.score_many,
        batch_size=entailment_batch_size,
    )
    prediction_ppl_scores = score_unique_texts(
        predictions,
        score_many_fn=ppl.score_many,
        batch_size=perplexity_batch_size,
    )
    reference_ppl_scores = score_unique_texts(
        references,
        score_many_fn=ppl.score_many,
        batch_size=perplexity_batch_size,
    )
    bertscore_scores = score_unique_pairs(
        predictions,
        references,
        score_many_fn=bertscore.score_pairs,
        batch_size=bertscore_batch_size,
    )

    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        is_empty_prediction = int(not ex.prediction)
        fkgl = float(textstat.flesch_kincaid_grade(ex.prediction)) if ex.prediction else float("nan")
        reference_ppl = reference_ppl_scores[idx]
        prediction_ppl = prediction_ppl_scores[idx]
        rows.append(
            {
                "system": ex.system,
                "row_id": ex.row_id,
                "level": ex.level,
                "source": ex.source,
                "prediction": ex.prediction,
                "reference": ex.reference,
                "empty_prediction": is_empty_prediction,
                "adequacy_nonref_nonllm_entailment": 0.0 if is_empty_prediction else entailment_scores[idx],
                "adequacy_ref_nonllm_bertscore": bertscore_scores[idx],
                "simplicity_nonref_nonllm_fkgl": fkgl,
                "simplicity_nonref_nonllm_target_band_hit": compute_target_band_hit(ex.level, fkgl),
                "simplicity_ref_nonllm_sari": sentence_sari(ex.source, ex.prediction, ex.reference),
                "fluency_nonref_nonllm_ppl": float("nan") if is_empty_prediction else prediction_ppl,
                "fluency_ref_nonllm_ppl_gap": (
                    float("nan")
                    if is_empty_prediction
                    else abs(math.log(max(prediction_ppl, 1e-8)) - math.log(max(reference_ppl, 1e-8)))
                ),
                "fluency_reference_ppl": reference_ppl,
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_metrics = [col for col in SUMMARY_METRIC_COLUMNS if col in df.columns]

    by_level = (
        df.groupby(["system", "level"], dropna=False)[available_metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    macro = (
        df.groupby(["system"], dropna=False)[available_metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    if "empty_prediction" in by_level.columns:
        by_level = by_level.rename(columns={"empty_prediction": "empty_prediction_rate"})
    if "empty_prediction" in macro.columns:
        macro = macro.rename(columns={"empty_prediction": "empty_prediction_rate"})
    return by_level, macro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Academic text simplification evaluation with local automatic metrics."
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        default=["outputs"],
        help="Prediction CSV files or directories. Default: outputs",
    )
    parser.add_argument(
        "--references",
        default="validation.jsonl",
        help="Validation JSONL containing gold simplifications.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Directory to write example-level and summary outputs.",
    )
    parser.add_argument(
        "--bertscore-model",
        default="bert-base-uncased",
        help="Backbone used for BERTScore.",
    )
    parser.add_argument(
        "--entailment-batch-size",
        type=int,
        default=None,
        help="Batch size for source entailment scoring. Default: auto-tuned from device memory.",
    )
    parser.add_argument(
        "--perplexity-batch-size",
        type=int,
        default=None,
        help="Batch size for GPT-2 perplexity scoring. Default: auto-tuned from device memory.",
    )
    parser.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=None,
        help="Batch size for BERTScore. Default: auto-tuned from device memory.",
    )
    parser.add_argument(
        "--cuda-amp-dtype",
        choices=["auto", "none", "float16", "bfloat16"],
        default="auto",
        help="CUDA autocast dtype for non-LLM metrics. Default: auto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_torch_device()
    configure_torch_runtime(device)
    auto_batch_sizes = resolve_auto_batch_sizes(device)
    entailment_batch_size = args.entailment_batch_size or auto_batch_sizes["entailment"]
    perplexity_batch_size = args.perplexity_batch_size or auto_batch_sizes["perplexity"]
    bertscore_batch_size = args.bertscore_batch_size or auto_batch_sizes["bertscore"]
    cuda_amp_dtype = resolve_cuda_amp_dtype(args.cuda_amp_dtype, device)
    amp_label = "disabled" if cuda_amp_dtype is None else str(cuda_amp_dtype).replace("torch.", "")
    print(
        "Runtime config: "
        f"device={device}, "
        f"entailment_batch_size={entailment_batch_size}, "
        f"perplexity_batch_size={perplexity_batch_size}, "
        f"bertscore_batch_size={bertscore_batch_size}, "
        f"cuda_amp_dtype={amp_label}"
    )

    references = load_reference_rows(Path(args.references))
    prediction_files = detect_prediction_files(args.predictions)
    examples = build_examples(prediction_files=prediction_files, references=references)

    metrics_df = compute_non_llm_metrics(
        examples,
        bertscore_model=args.bertscore_model,
        device=device,
        entailment_batch_size=entailment_batch_size,
        perplexity_batch_size=perplexity_batch_size,
        bertscore_batch_size=bertscore_batch_size,
        cuda_amp_dtype=cuda_amp_dtype,
    )
    metrics_df.to_csv(output_dir / "example_metrics_non_llm.csv", index=False)

    by_level, macro = summarize_metrics(metrics_df)
    by_level.to_csv(output_dir / "summary_by_system_level.csv", index=False)
    macro.to_csv(output_dir / "summary_macro.csv", index=False)

    print(f"Saved example metrics to {output_dir}")


if __name__ == "__main__":
    main()
