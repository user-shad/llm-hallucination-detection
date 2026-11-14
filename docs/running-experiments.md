# Running the Experiments

Run commands from the repository root.

## Environment

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Windows Command Prompt: `.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Transformer checkpoints are downloaded on first use. They are not committed.

## 1. Classical baselines

```bash
python experiments/run_classical_baselines.py \
  --data data/llm_hallucination_detection_dataset.jsonl \
  --output-dir outputs/classical
```

On Windows Command Prompt, put the command on one line instead of using `\`.

Outputs:

- `classical_fold_metrics.csv`
- `classical_oof_predictions.csv`
- `classical_summary.csv`
- `classical_run_config.json`

## 2. Error analysis

```bash
python experiments/analyze_errors.py \
  --predictions outputs/classical/classical_oof_predictions.csv \
  --output-dir outputs/classical_error_analysis
```

Review `incorrect_predictions.csv` manually. Generated reports are working artifacts and are ignored by Git.

## 3. Pretrained NLI baseline

This model is used without project-specific fine-tuning:

```bash
python experiments/run_nli_baseline.py \
  --output-dir outputs/nli_baseline
```

The decision rule is:

- entailment → faithful;
- contradiction or neutral → hallucinated.

This is the closest inexpensive first test of the SciHal NLI framing.

## 4. Fine-tuned transformers

Run one model at a time:

```bash
python experiments/run_transformer_cv.py --model bert --output-dir outputs/bert
python experiments/run_transformer_cv.py --model roberta --output-dir outputs/roberta
python experiments/run_transformer_cv.py --model deberta --output-dir outputs/deberta
python experiments/run_transformer_cv.py --model scibert --output-dir outputs/scibert
```

Each command trains six models, one per grouped fold, so CPU execution may take several hours. DeBERTa is likely the slowest. Use a GPU if available.

Default checkpoints:

| Name | Checkpoint |
| --- | --- |
| BERT | `bert-base-uncased` |
| RoBERTa | `roberta-base` |
| DeBERTa | `microsoft/deberta-v3-base` |
| SciBERT | `allenai/scibert_scivocab_uncased` |

Override a checkpoint with `--checkpoint`.

## 5. Ablations

The transformer runner supports:

```text
context_response
prompt_context_response
prompt_response
response_only
```

Example:

```bash
python experiments/run_transformer_cv.py \
  --model bert \
  --condition response_only \
  --output-dir outputs/bert_response_only
```

Use `context_response` for the primary result. Ablations test whether a model is using the evidence or exploiting response-only shortcuts.

## Before adding a result

1. Confirm that all six folds completed.
2. Check `run_config.json`.
3. Review the fold standard deviation.
4. Run the error-analysis script.
5. Add one entry to `docs/experiment-log.md`.
6. Add the aggregate row to `docs/model-results.md`.
7. State limitations; do not compare grouped and random-split scores as if they were equivalent.
