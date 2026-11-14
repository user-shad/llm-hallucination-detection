# Experiment Log

Record every completed, failed, or abandoned experiment. Do not delete an entry because its result is poor.

## Status

| ID | Date | Model | Input | Evaluation | Status | Weighted F1 | Notes |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| EXP-001 | TBD | Classical baselines | context + response | 6-fold grouped CV | Planned | — | Majority, overlap, TF-IDF LR, linear SVM |
| EXP-002 | TBD | BERT | context + response | 6-fold grouped CV | Planned | — | Pair classification |
| EXP-003 | TBD | RoBERTa | context + response | 6-fold grouped CV | Planned | — | Pair classification |
| EXP-004 | TBD | DeBERTa-v3 NLI | context + response | 6-fold grouped CV | Planned | — | Zero-shot, then fine-tuned |
| EXP-005 | TBD | SciBERT | context + response | 6-fold grouped CV | Planned | — | Domain-specific comparison |

## Entry template

Copy this section for each run.

### EXP-XXX — Short model name

- **Date:**
- **Research question:**
- **Hypothesis:**
- **Dataset version or commit:**
- **Rows before deduplication:**
- **Unique examples:**
- **Context groups:**
- **Input condition:**
- **Model/checkpoint:**
- **Package versions:**
- **Hardware:**
- **Fold definition:**
- **Random seed(s):**
- **Hyperparameters:**
- **Configurations attempted:**
- **Training time:**
- **Inference time:**
- **Weighted F1, mean ± standard deviation:**
- **Macro F1, mean ± standard deviation:**
- **Hallucinated precision/recall/F1:**
- **Faithful precision/recall/F1:**
- **Accuracy:**
- **Output files:**
- **Main errors:**
- **Interpretation:**
- **Limitations:**
- **Decision and next step:**

## Rules

1. Record the hypothesis before interpreting the result.
2. Include failed runs and the error that caused them.
3. Do not choose a model from one favorable fold.
4. Do not compare scores produced by different split strategies as if they used the same test data.
5. Link every final result table row to an experiment ID.
