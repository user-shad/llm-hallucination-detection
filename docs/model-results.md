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

## EXP-002 — Pretrained DeBERTa-v3 NLI

The pretrained `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` checkpoint was evaluated without project-specific training. Context was the NLI premise and response was the hypothesis.

| Accuracy | Weighted F1 | Macro F1 | Correct examples | Inference time |
| ---: | ---: | ---: | ---: | ---: |
| 1.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 192/192 | 1.379 seconds |

All six context groups scored 1.000. Faithful examples had entailment probabilities between 0.680 and 0.998, while hallucinated examples ranged from 0.00006 to 0.398.

The binary decision rule combines contradiction and neutral as hallucinated: entailment probability at least 0.5 is faithful; anything below 0.5 is hallucinated.

### Interpretation

This result supports the paper-guided NLI approach. It also shows that the current synthetic dataset is easy for a strong pretrained NLI checkpoint. The perfect score must not be described as perfect real-world hallucination detection. A second, independently collected dataset is needed to test generalization.

### Reproduction

```bash
python experiments/run_nli_baseline.py --output-dir outputs/nli_baseline
```

## Planned transformer results

| Experiment | Model | Status |
| --- | --- | --- |
| EXP-003 | BERT pair classifier | Runner ready; training pending |
| EXP-004 | RoBERTa pair classifier | Runner ready; training pending |
| EXP-005 | Fine-tuned DeBERTa-v3 | Runner ready; training pending |
| EXP-006 | SciBERT pair classifier | Runner ready; training pending |

Example:

```bash
python experiments/run_transformer_cv.py \
  --model deberta \
  --output-dir outputs/deberta
```

Transformer scores will be added only after all six context-grouped folds finish successfully.
