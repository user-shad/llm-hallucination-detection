# LLM Hallucination Detection

COSC 497 Senior Design Project I

## Team

- Shahad Aljasmi
- Wadha Msaiqer
- Mouza Alshehhi
- Maram Alkatheeri

## Supervisor

Dr. U Zeyar Aung, Computer Science, Khalifa University

## Project

This project evaluates whether an AI-generated response is faithful to a supplied context or contains a hallucination. It compares classical baselines, fine-tuned transformer classifiers, and a pretrained natural-language-inference model.

## Dataset and evaluation

Although the source dataset contains 5,000 rows, exact deduplication leaves 192 unique examples across 24 contexts. The project therefore uses six-fold grouped cross-validation so that each test context is unseen during training.

| Main approach | Accuracy | Weighted F1 |
| --- | ---: | ---: |
| Pretrained DeBERTa-v3 NLI | 1.000 | 1.000 ± 0.000 |
| Fine-tuned DeBERTa-v3 | 0.938 | 0.937 ± 0.070 |
| Fine-tuned RoBERTa | 0.896 | 0.892 ± 0.100 |
| Fine-tuned BERT | 0.771 | 0.739 ± 0.209 |

The perfect NLI result applies only to this small synthetic dataset and must not be interpreted as perfect real-world hallucination detection.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

The dataset is stored at `data/llm_hallucination_detection_dataset.jsonl`.

## Run experiments

Classical baselines:

```bash
python experiments/run_classical_baselines.py --output-dir outputs/classical
```

Pretrained NLI:

```bash
python experiments/run_nli_baseline.py --output-dir outputs/nli_baseline
```

Fine-tuned transformers:

```bash
python experiments/run_transformer_cv.py --model bert --output-dir outputs/bert
python experiments/run_transformer_cv.py --model roberta --output-dir outputs/roberta
python experiments/run_transformer_cv.py --model deberta --output-dir outputs/deberta
```

For Google Colab GPU runs, use `experiments/colab_transformer_experiments.ipynb`. Generated outputs and downloaded model weights are excluded from Git.

## Documentation

- [Final model analysis](docs/final-analysis.md)
- [Detailed model results](docs/model-results.md)
- [Experiment log](docs/experiment-log.md)
- [Evaluation protocol](docs/experimental-protocol.md)
- [Paper comparison matrix](docs/paper-matrix.md)
- [Experiment instructions](docs/running-experiments.md)

SciBERT and validation on an independent dataset are recorded as future work because of the available project time.
