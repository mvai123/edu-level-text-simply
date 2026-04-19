# Towards Efficient Educational-Level-Aware Text Simplification

## Setup environment

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

## Baselines

```bash
# Identity
uv run python -m baselines.wrapper.cli \
  --backend identity \
  --input examples/dummy_input.json \
  --output outputs/identity.json

# Summarization
uv run python -m baselines.wrapper.cli \
  --backend summarization \
  --input examples/dummy_input.json \
  --output outputs/summarization.json

# Llama 3: start Ollama separately with `ollama serve`
uv run python -m baselines.wrapper.cli \
  --backend llama3 \
  --input examples/dummy_input.json \
  --output outputs/llama3.json

# Gemini
uv run python -m baselines.wrapper.cli \
  --backend gemini \
  --input examples/dummy_input.json \
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

## Evaluation

```bash
uv run python evaluation/academic_eval.py \
  --predictions outputs/predictions/t5-finetune-run.jsonl outputs/readability.csv outputs/round_trip.csv \
  --references data/validation.jsonl \
  --output-dir outputs/evaluation
```
