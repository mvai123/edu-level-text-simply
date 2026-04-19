# Towards Efficient Educational-Level-Aware Text Simplification

## How to set up the environment

```bash
uv python install 3.10
uv sync --all-groups
```

```bash
export OPENROUTER_API_KEY="..."
export GEMINI_API_KEY="..."
```

## Our method

Generate data.

```bash
uv run python method/generate_data.py \
  --input data/source_train.jsonl \
  --output data/output_simplified.jsonl

uv run python method/split_data.py \
  --input data/output_simplified.jsonl \
  --train-output data/train.jsonl \
  --validation-output data/validation.jsonl \
  --validation-size 1000
```

Train.

```bash
uv run hf download google/flan-t5-base \
  --local-dir weights/flan_t5_base \
  --include "*.json" "*.safetensors" "*.model"

uv run python method/train_model.py \
  --config method/configs/simple.yaml
```

Infer.

```bash
uv run python method/infer_model.py \
  --config method/configs/simple.yaml
```

By default, inference writes `outputs/predictions/t5-finetune-run.jsonl`. Each
line contains the source text and one simplification for each target reading
level.

```json
{
  "original_text": "I/O may refer to: Input/output, a system of communication...",
  "simplified_kindergarten": "One name has many meanings. It can be music or computers.",
  "simplified_primary_school": "I/O can mean many different things. It often means how computers send and get information.",
  "simplified_secondary_school": "The term I/O has several meanings across different fields. In computing, it refers to input and output, or how systems communicate."
}
```

## Baselines

The wrapper baselines use JSON array input with `text` and `age_band` fields.

```bash
# Identity
uv run python -m baselines.wrapper.cli \
  --backend identity \
  --input data/validation.jsonl \
  --output outputs/identity.json

# Summarization
uv run python -m baselines.wrapper.cli \
  --backend summarization \
  --input data/validation.jsonl \
  --output outputs/summarization.json

# Llama 3: start Ollama separately with `ollama serve`
uv run python -m baselines.wrapper.cli \
  --backend llama3 \
  --input data/validation.jsonl \
  --output outputs/llama3.json

# Gemini
uv run python -m baselines.wrapper.cli \
  --backend gemini \
  --input data/validation.jsonl \
  --output outputs/gemini.json

# Readability
uv run python baselines/readability/readability_pipeline.py \
  --input data/validation.jsonl \
  --output outputs/readability.csv \
  --text_col original_text

# Round-trip
uv run hf download Helsinki-NLP/opus-mt-en-fr \
  --local-dir weights/opus_mt_en_fr \
  --include "pytorch_model.*" "*.json" "*.spm"

uv run hf download Helsinki-NLP/opus-mt-fr-en \
  --local-dir weights/opus_mt_fr_en \
  --include "pytorch_model.*" "*.json" "*.spm"

uv run python baselines/round_trip/round_trip.py \
  --input data/validation.jsonl \
  --output outputs/round_trip.csv \
  --text_col original_text
```

Baseline outputs are written under `outputs/`. The wrapper baselines write JSON
for the requested `age_band`; readability and round-trip write CSV files in the
wide prediction format used by evaluation.

```csv
original_text,simplified_kindergarten,simplified_primary_school,simplified_secondary_school
"I/O may refer to: Input/output, a system of communication...","I/O is a name for many things.","I/O can mean many different things.","The term I/O has several meanings across different fields."
```

## Evaluation

```bash
uv run python evaluation/academic_eval.py \
  --predictions outputs/predictions/t5-finetune-run.jsonl outputs/readability.csv outputs/round_trip.csv \
  --references data/validation.jsonl \
  --output-dir outputs/evaluation
```

The evaluation directory will contain:

- example_metrics_non_llm.csv
- summary_by_system_level.csv
- summary_macro.csv

`summary_by_system_level.csv` reports each system by target level, while
`summary_macro.csv` reports the macro-average scores for each system.
