# Final Model Analysis

## Project objective

The project evaluates whether an AI response is faithful to a supplied context or contains a hallucination. The detector receives the context and candidate response as a pair. The main reported measure is weighted F1, supported by macro F1, accuracy, per-class measures, and error analysis.

## Evaluation design

The source file contains 5,000 rows, but only 192 unique prompt-context-response examples remain after exact deduplication. These examples cover 24 distinct contexts. To prevent copies or closely related examples from appearing in both training and testing, all results use six-fold GroupKFold with context as the grouping variable. Each example receives one out-of-fold prediction.

The four main transformer approaches are:

1. a pretrained DeBERTa-v3 NLI model used without project-specific training;
2. fine-tuned BERT;
3. fine-tuned RoBERTa; and
4. fine-tuned DeBERTa-v3.

The fine-tuned models use the same grouped protocol. BERT and RoBERTa use batch size 16; DeBERTa-v3 uses batch size 8. All use a learning rate of 2e-5, weight decay of 0.01, a maximum sequence length of 256, up to five epochs, and seed 42.

## Final comparison

| Approach | Project training | Accuracy | Weighted F1 | Hallucinated precision | Hallucinated recall | Hallucinated F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained DeBERTa-v3 NLI | None | 1.000 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 |
| Fine-tuned DeBERTa-v3 | Six grouped folds | 0.938 | 0.937 ± 0.070 | 1.000 | 0.875 | 0.929 |
| Fine-tuned RoBERTa | Six grouped folds | 0.896 | 0.892 ± 0.100 | 1.000 | 0.792 | 0.873 |
| Fine-tuned BERT | Six grouped folds | 0.771 | 0.739 ± 0.209 | 0.792 | 0.583 | 0.665 |

For context, the strongest non-transformer baseline was word overlap, with accuracy 0.896 and weighted F1 0.885 ± 0.169. TF-IDF logistic regression and linear SVM reached weighted F1 scores of 0.569 and 0.533 respectively.

## Main findings

### 1. The NLI formulation was the best fit

The pretrained NLI model classified all 192 unique examples correctly. Its task is already to determine whether a hypothesis is supported by a premise, which closely matches checking whether a response is supported by its context. It therefore needed no training on this small dataset.

This perfect score should be interpreted as evidence that the current synthetic dataset is easy for this checkpoint. It is not evidence of perfect hallucination detection on unrestricted real-world text.

### 2. Fine-tuned DeBERTa-v3 was the strongest trained classifier

DeBERTa-v3 achieved weighted F1 of 0.937 and correctly classified 180 of 192 examples. It produced no false alarms on faithful responses and detected 84 of 96 hallucinations. Its fold scores ranged from 0.873 to 1.000, making it more stable than BERT and RoBERTa.

### 3. RoBERTa improved substantially over BERT

RoBERTa improved weighted F1 from BERT's 0.739 to 0.892. BERT was unstable across unseen contexts: one fold predicted every hallucinated response as faithful and scored weighted F1 of 0.333. RoBERTa's lowest fold score was 0.733.

### 4. Simple overlap remained a meaningful baseline

Word overlap slightly trailed RoBERTa in weighted F1 and matched its accuracy. This happened because many synthetic hallucinations visibly replace or add words. However, overlap missed subtle substitutions involving dates, locations, and facts, so it is not a sufficient general detector.

## Error analysis

| Approach | Total errors | Faithful responses flagged | Hallucinations missed | Main error types |
| --- | ---: | ---: | ---: | --- |
| Pretrained NLI | 0 | 0 | 0 | None on this dataset |
| Fine-tuned DeBERTa-v3 | 12 | 0 | 12 | 8 factual, 4 date |
| Fine-tuned RoBERTa | 20 | 0 | 20 | 8 factual, 8 date, 4 contradiction |
| Fine-tuned BERT | 44 | 4 | 40 | factual, definition, irrelevant, contradiction, and date errors |

The repeated difficulty with factual and date substitutions is important. These responses often preserve nearly all of the original sentence and change only one critical detail. Detecting them requires comparing the relationship between the context and response rather than relying only on topic or vocabulary similarity.

## Limitations

- The dataset is synthetic and has only 192 unique examples across 24 contexts after deduplication.
- The results may reflect repeated templates and simple transformations used to generate the dataset.
- Only one random seed was used for the fine-tuned models.
- No independent dataset was evaluated because of the available project time.
- A perfect NLI result on this dataset cannot be generalized to open-domain or real-world hallucinations.
- Runtime measurements came from a single Colab environment and are not hardware-independent benchmarks.

These limitations should be stated with the results rather than hidden. Grouped cross-validation and deduplication reduce leakage, but they cannot make a small synthetic dataset representative of real deployments.

## Scope decision

SciBERT was considered but not run. The completed approaches already compare a general BERT encoder, RoBERTa, DeBERTa-v3, and an NLI-specialized checkpoint. SciBERT would mainly add a domain-pretraining variation, while the larger remaining weakness is external validity. Given the available time, completing the analysis and documentation is more useful than training another model.

## Conclusion

The pretrained DeBERTa-v3 NLI model is the best detector for the current dataset, reaching weighted F1 of 1.000 without project-specific training. Among models trained on the project data, DeBERTa-v3 is strongest at 0.937 weighted F1, followed by RoBERTa at 0.892 and BERT at 0.739.

The central result is that task formulation matters: treating hallucination detection as evidence-response entailment worked better than fine-tuning a general text classifier on a very small dataset. For a prototype built from this work, the pretrained NLI method is the recommended approach, while fine-tuned DeBERTa-v3 provides the strongest supervised comparison.

## Future work

If more time becomes available, the first priority should be evaluation on an independently collected dataset, not another model on the same synthetic data. Further work could also repeat training with multiple seeds, calibrate confidence scores, and add explanations that identify the unsupported part of a response.

Detailed configurations and individual experiment records are available in [experiment-log.md](experiment-log.md), with numerical results in [model-results.md](model-results.md).
