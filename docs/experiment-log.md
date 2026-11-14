# Experiment Log

Record every completed, failed, or abandoned experiment. Do not delete an entry because its result is poor.

## Status

| ID | Date | Model | Input | Evaluation | Status | Weighted F1 | Notes |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| EXP-001 | 14 Nov 2026 | Classical baselines | context + response | 6-fold grouped CV | Complete | 0.885 best | Word overlap was strongest |
| EXP-002 | 14 Nov 2026 | Pretrained DeBERTa-v3 NLI | context + response | 6-fold grouped reporting | Complete | 1.000 | No project-specific training |
| EXP-003 | 14 Nov 2026 | BERT | context + response | 6-fold grouped CV | Complete | 0.739 | High fold variation |
| EXP-004 | 14 Nov 2026 | RoBERTa | context + response | 6-fold grouped CV | Complete | 0.892 | Zero false alarms |
| EXP-005 | 14 Nov 2026 | DeBERTa-v3 | context + response | 6-fold grouped CV | Complete | 0.937 | Strongest fine-tuned model |
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

## EXP-003 — Fine-tuned BERT

- **Date:** 14 November 2026
- **Research question:** Can BERT learn evidence-response consistency from the small project dataset?
- **Hypothesis:** Pair fine-tuning will improve on TF-IDF, but may be unstable because only 24 contexts are available.
- **Unique examples:** 192
- **Context groups:** 24
- **Input condition:** context + response
- **Model/checkpoint:** `bert-base-uncased`
- **Fold definition:** six-fold GroupKFold by context, with grouped validation inside each training fold
- **Train/validation/test per fold:** 128 / 32 / 32 examples
- **Hyperparameters:** learning rate 2e-5, batch size 16, maximum 5 epochs, weight decay 0.01, maximum length 256, seed 42
- **Total training time:** 335.403 seconds (about 5 minutes 35 seconds)
- **Total inference time:** 0.724 seconds
- **Weighted F1, mean ± standard deviation:** 0.739 ± 0.209
- **Macro F1, mean ± standard deviation:** 0.739 ± 0.209
- **Accuracy:** 0.771
- **Hallucinated precision/recall/F1:** 0.792 / 0.583 / 0.665
- **Faithful precision/recall/F1:** 0.719 / 0.958 / 0.814
- **Aggregate confusion counts:** 92 faithful correct, 4 faithful false alarms, 56 hallucinations detected, 40 hallucinations missed
- **Main errors:** factual errors (16), definition errors (8), irrelevant claims (8), contradictions (4), date errors (4), and four faithful examples
- **Fold stability:** weighted F1 ranged from 0.333 to 0.873; fold 3 predicted all 16 hallucinations as faithful
- **Interpretation:** BERT outperformed TF-IDF but underperformed word overlap and pretrained NLI; the small dataset does not support stable BERT fine-tuning across unseen contexts
- **Limitations:** one seed and a small synthetic dataset; results may change with initialization and validation grouping
- **Decision and next step:** run RoBERTa with the same protocol, then compare whether NLI pretraining is more useful than general language-model pretraining

## EXP-004 — Fine-tuned RoBERTa

- **Date:** 14 November 2026
- **Research question:** Does RoBERTa generalize across unseen contexts better than BERT?
- **Hypothesis:** RoBERTa will be more stable than BERT but may still be limited by the small number of context groups.
- **Unique examples:** 192
- **Context groups:** 24
- **Input condition:** context + response
- **Model/checkpoint:** `roberta-base`
- **Fold definition:** six-fold GroupKFold by context, with grouped validation inside each training fold
- **Train/validation/test per fold:** 128 / 32 / 32 examples
- **Hyperparameters:** learning rate 2e-5, batch size 16, maximum 5 epochs, weight decay 0.01, maximum length 256, seed 42
- **Total training time:** 336.796 seconds (about 5 minutes 37 seconds)
- **Total inference time:** 0.883 seconds
- **Weighted F1, mean ± standard deviation:** 0.892 ± 0.100
- **Macro F1, mean ± standard deviation:** 0.892 ± 0.100
- **Accuracy:** 0.896
- **Hallucinated precision/recall/F1:** 1.000 / 0.792 / 0.873
- **Faithful precision/recall/F1:** 0.844 / 1.000 / 0.911
- **Aggregate confusion counts:** 96 faithful correct, 0 faithful false alarms, 76 hallucinations detected, 20 hallucinations missed
- **Main errors:** date errors (8), factual errors (8), and contradictions (4)
- **Fold stability:** weighted F1 ranged from 0.733 to 1.000; two folds scored 1.000
- **Interpretation:** RoBERTa was substantially more stable than BERT and slightly exceeded word overlap in weighted F1, but pretrained NLI remained clearly strongest
- **Limitations:** one seed, 24 synthetic contexts, and no external test dataset
- **Decision and next step:** run fine-tuned DeBERTa-v3 and SciBERT, then evaluate the strongest approaches on a second dataset

## EXP-005 — Fine-tuned DeBERTa-v3

- **Date:** 14 November 2026
- **Research question:** Does fine-tuned DeBERTa-v3 provide better evidence-response reasoning than BERT and RoBERTa?
- **Hypothesis:** DeBERTa-v3's stronger representations will improve hallucination recall and fold stability.
- **Unique examples:** 192
- **Context groups:** 24
- **Input condition:** context + response
- **Model/checkpoint:** `microsoft/deberta-v3-base`
- **Fold definition:** six-fold GroupKFold by context, with grouped validation inside each training fold
- **Train/validation/test per fold:** 128 / 32 / 32 examples
- **Hyperparameters:** learning rate 2e-5, batch size 8, maximum 5 epochs, weight decay 0.01, maximum length 256, seed 42
- **Total training time:** 793.098 seconds (about 13 minutes 13 seconds)
- **Total inference time:** 1.229 seconds
- **Weighted F1, mean ± standard deviation:** 0.937 ± 0.070
- **Macro F1, mean ± standard deviation:** 0.937 ± 0.070
- **Accuracy:** 0.938
- **Hallucinated precision/recall/F1:** 1.000 / 0.875 / 0.929
- **Faithful precision/recall/F1:** 0.900 / 1.000 / 0.944
- **Aggregate confusion counts:** 96 faithful correct, 0 faithful false alarms, 84 hallucinations detected, 12 hallucinations missed
- **Main errors:** factual errors (8) and date errors (4)
- **Fold stability:** weighted F1 ranged from 0.873 to 1.000; the final three folds scored 1.000
- **Interpretation:** DeBERTa-v3 was the strongest fine-tuned classifier, but the pretrained NLI checkpoint remained best; this suggests that task-relevant NLI pretraining is more valuable here than fine-tuning a general encoder on 192 examples.
- **Limitations:** one seed, 24 synthetic contexts, and no external test dataset
- **Decision and next step:** run SciBERT under the same protocol, then evaluate the strongest approaches on a second dataset

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
