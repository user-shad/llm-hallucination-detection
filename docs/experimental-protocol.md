# Experimental Protocol

This document defines the evaluation rules for every detector in this repository. A model result should not be added to the final comparison unless it follows this protocol or clearly states why it differs.

## Dataset unit

The source file has 5,000 rows but only 192 unique combinations of:

- `prompt`
- `context`
- `response`

Exact duplicate triples are removed before splitting. The target is converted to:

- `faithful = 0`
- `hallucinated = 1`

The following fields are never model inputs because they reveal the label or would not normally be available during inference:

- `label`
- `hallucination_type`
- `factuality_score`
- `confidence_score`

Metadata such as domain and difficulty may be used only for error analysis.

## Grouped evaluation

All examples with the same `context` stay in the same fold. This prevents the same fact, faithful response, and altered response from appearing in both training and testing under different prompt wording.

The primary evaluation is **six-fold GroupKFold by context**:

- 24 context groups;
- four held-out contexts per fold;
- every unique example appears in a test fold exactly once;
- all models use the same folds.

Six folds are used because 24 contexts divide evenly into six groups. Fold IDs must be saved with the predictions.

A simple random split may be reported only as a leakage demonstration. It must not be presented as the main result.

## Validation and tuning

Classical models use a small fixed parameter grid. Parameter selection happens using grouped folds within the training portion, never the held-out test fold.

Transformer experiments use a documented, limited search:

- learning rate: one or two stated values;
- epochs: maximum five with early stopping;
- batch size: selected for available hardware;
- random seeds: 42, and additional seeds when runtime allows.

The number of attempted configurations must be recorded. Test results must not be used to choose a configuration.

## Inputs

Every model must state which input condition it uses:

1. `context + response` — primary grounded condition;
2. `prompt + context + response` — secondary condition;
3. `prompt + response` — no-context ablation;
4. `response only` — shortcut-detection ablation.

For pair encoders, context is the premise/evidence and response is the hypothesis/claim.

## Metrics

The primary metric is **weighted F1**, matching SciHal25.

Also report:

- macro F1;
- hallucinated-class precision, recall, and F1;
- faithful-class precision, recall, and F1;
- accuracy;
- confusion matrix;
- mean and standard deviation across folds;
- out-of-fold predictions for all 192 unique examples.

When feasible, also record:

- training time;
- inference time;
- model parameter count or disk size;
- hardware.

## Required artifacts

Each completed experiment produces:

- one configuration record;
- per-fold metrics;
- one out-of-fold prediction file;
- an aggregate result row;
- a short error-analysis note.

Generated model weights and large caches are not committed. Small CSV or JSON results may be committed when they are necessary to reproduce a table.

## Error analysis

After prediction, errors may be grouped using metadata that was excluded from training:

- hallucination type;
- domain;
- task type;
- difficulty.

At minimum, inspect:

- numeric changes;
- entity, date, and location changes;
- negation and contradiction;
- unsupported additions;
- overgeneralization;
- false alarms on faithful responses.

The analysis must distinguish a model limitation from an ambiguous or low-quality example.

## Reproducibility checklist

Before reporting a result:

- [ ] Exact duplicates were removed.
- [ ] Context groups do not cross folds.
- [ ] No target-leaking field was used.
- [ ] Random seeds were recorded.
- [ ] Model checkpoint and package versions were recorded.
- [ ] Hyperparameters were selected without test-fold feedback.
- [ ] All requested metrics were saved.
- [ ] Failed runs and deviations were noted.
- [ ] Predictions were retained for error analysis.

## Result interpretation

The dataset has only 24 context groups and is fully synthetic. Results may vary substantially by fold and cannot establish real-world performance. Claims in the report must therefore be limited to this dataset until a second, independently collected benchmark is evaluated.
