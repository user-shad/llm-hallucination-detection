# Model Results

All primary results use the protocol in [experimental-protocol.md](experimental-protocol.md): exact duplicates removed and six-fold GroupKFold by context.

## EXP-001 — Classical baselines

- **Date:** 14 November 2026
- **Input:** context + response
- **Unique examples:** 192
- **Context groups:** 24
- **Test coverage:** every example predicted once out of fold
- **Seed:** 42 where applicable

| Model | Accuracy | Weighted F1 | Macro F1 | Hallucinated precision | Hallucinated recall | Hallucinated F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority class | 0.500 | 0.333 | 0.333 | 0.000 | 0.000 | 0.000 |
| TF-IDF + linear SVM | 0.562 | 0.533 | 0.533 | 0.540 | 0.708 | 0.593 |
| TF-IDF + logistic regression | 0.604 | 0.569 | 0.569 | 0.568 | 0.792 | 0.642 |
| Word overlap | 0.896 | 0.885 | 0.885 | 1.000 | 0.792 | 0.852 |

The table reports the mean of the six fold metrics. Weighted-F1 standard deviations were:

- majority: 0.000;
- linear SVM: 0.105;
- logistic regression: 0.101;
- word overlap: 0.169.

### Interpretation

Word overlap remains the strongest classical method, but its fold variation is large. It misses hallucinations that retain most source words while changing one important number, entity, date, or relation. The grouped result is lower and more informative than the previous favorable single split.

Mixing context and response into one TF-IDF document performs poorly because the representation measures topic vocabulary rather than the logical relationship between evidence and claim. This supports the next experiment: pair encoders and NLI-pretrained models.

### Reproduction

```bash
python experiments/run_classical_baselines.py \
  --data data/llm_hallucination_detection_dataset.jsonl \
  --output-dir outputs/classical
```

The command writes fold metrics, out-of-fold predictions, an aggregate summary, and a run configuration. The generated `outputs/` directory is intentionally excluded from Git.

## Planned transformer results

| Experiment | Model | Status |
| --- | --- | --- |
| EXP-002 | BERT pair classifier | Runner ready; training pending |
| EXP-003 | RoBERTa pair classifier | Runner ready; training pending |
| EXP-004 | DeBERTa-v3 NLI | Runner ready; training pending |
| EXP-005 | SciBERT pair classifier | Runner ready; training pending |

Example:

```bash
python experiments/run_transformer_cv.py \
  --model deberta \
  --output-dir outputs/deberta
```

Transformer scores will be added only after all six context-grouped folds finish successfully.
