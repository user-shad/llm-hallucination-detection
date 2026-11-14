# Experiment Log

Record every completed, failed, or abandoned experiment. Do not delete an entry because its result is poor.

## Status

| ID | Date | Model | Input | Evaluation | Status | Weighted F1 | Notes |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| EXP-001 | 14 Nov 2026 | Classical baselines | context + response | 6-fold grouped CV | Complete | 0.885 best | Word overlap was strongest |
| EXP-002 | 14 Nov 2026 | Pretrained DeBERTa-v3 NLI | context + response | 6-fold grouped reporting | Complete | 1.000 | No project-specific training |
| EXP-003 | TBD | BERT | context + response | 6-fold grouped CV | Planned | — | Pair classification |
| EXP-004 | TBD | RoBERTa | context + response | 6-fold grouped CV | Planned | — | Pair classification |
| EXP-005 | TBD | DeBERTa-v3 | context + response | 6-fold grouped CV | Planned | — | Fine-tuned pair classifier |
| EXP-006 | TBD | SciBERT | context + response | 6-fold grouped CV | Planned | — | Domain-specific comparison |

## EXP-001 — Classical baselines

- **Date:** 14 November 2026
- **Research question:** How do simple detectors perform when every test context is unseen during training?
- **Hypothesis:** Word overlap will beat topic-based TF-IDF because the synthetic hallucinations usually add or replace visible words.
- **Rows before deduplication:** 5,000
- **Unique examples:** 192
- **Context groups:** 24
- **Input condition:** context + response
- **Models:** majority class, word overlap at 0.60, TF-IDF logistic regression, TF-IDF linear SVM
- **Fold definition:** six-fold GroupKFold by context; four test contexts per fold
- **Random seed:** 42 where applicable
- **Configurations attempted:** one documented configuration per model
- **Weighted F1, mean ± standard deviation:** majority 0.333 ± 0.000; word overlap 0.885 ± 0.169; logistic regression 0.569 ± 0.101; linear SVM 0.533 ± 0.105
- **Best-model accuracy:** 0.896
- **Best-model hallucinated precision/recall/F1:** 1.000 / 0.792 / 0.852
- **Output files:** generated locally by `experiments/run_classical_baselines.py`; aggregate values recorded in `docs/model-results.md`
- **Main errors:** word overlap missed all date and location errors in this dataset and some factual substitutions
- **Interpretation:** lexical overlap is a strong synthetic-data baseline, but the large fold variation and substitution errors justify testing NLI pair encoders
- **Limitations:** only 24 synthetic contexts; no claim of real-world performance
- **Decision and next step:** run the pretrained NLI baseline, then BERT, RoBERTa, DeBERTa-v3, and SciBERT with the same folds

## EXP-002 — Pretrained DeBERTa-v3 NLI

- **Date:** 14 November 2026
- **Research question:** Can a pretrained NLI model detect unsupported or contradictory responses without training on this project dataset?
- **Hypothesis:** An NLI model will outperform word overlap because it compares the evidence and claim relationship.
- **Unique examples:** 192
- **Context groups:** 24
- **Input condition:** context as premise and response as hypothesis
- **Model/checkpoint:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- **Training:** none; pretrained zero-shot inference
- **Fold definition:** predictions reported in the same six context groups used by other experiments
- **Batch size:** 8
- **Maximum length:** 256 tokens
- **Inference time:** 1.379 seconds on the Colab run
- **Decision rule:** entailment probability at least 0.5 means faithful; otherwise hallucinated
- **Weighted F1, mean ± standard deviation:** 1.000 ± 0.000
- **Macro F1, mean ± standard deviation:** 1.000 ± 0.000
- **Accuracy:** 1.000
- **Correct predictions:** 192 of 192
- **Probability separation:** faithful entailment probabilities ranged from 0.680 to 0.998; hallucinated examples ranged from 0.00006 to 0.398
- **Main errors:** none on this dataset
- **Interpretation:** the pretrained NLI formulation fits the simple evidence-response relationships in this synthetic dataset much better than the lexical baselines
- **Limitations:** a perfect result on 24 synthetic contexts is strong evidence that the benchmark is easy for this checkpoint, not evidence of perfect real-world hallucination detection
- **Decision and next step:** run BERT and RoBERTa under grouped fine-tuning, then test generalization on a second dataset

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
