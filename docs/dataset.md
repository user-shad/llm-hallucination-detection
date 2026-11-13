# Dataset Description

## Source

This project uses the [LLM Hallucination Detection dataset on Kaggle](https://www.kaggle.com/datasets/mmumairkhattak/llm-hallucination-detection-nlp-and-ai-safety).

A copy is stored in this repository at
`data/llm_hallucination_detection_dataset.jsonl`. It is a JSON Lines file,
with one record per line.

## Dataset Size

The file contains 5,000 records. After removing repeated combinations of
`prompt`, `context`, and `response`, there are 192 unique examples:

- 96 labeled `faithful`
- 96 labeled `hallucinated`

The remaining 4,808 records repeat a prompt, context, and response combination
already present. Other metadata in those records may differ.

The 192 unique examples contain 96 distinct prompt–context pairs and
24 distinct context strings.

## Dataset Fields

| Field | Description |
| --- | --- |
| `prompt` | Question or instruction given to the language model |
| `context` | Passage used to evaluate the response |
| `response` | Answer being classified |
| `label` | Dataset label: `faithful` or `hallucinated` |
| `hallucination_type` | Metadata describing a hallucination category |
| `domain`, `task_type` | Metadata describing the domain and task |
| `factuality_score`, `confidence_score` | Scores included in the dataset |
| `id`, `evidence_available`, `difficulty`, `language`, `source_type`, `created_year` | Additional record metadata |

The domains are history, technology, science, geography, general medicine,
general law, business, and general finance.

All records are marked as English and have `source_type` set to `synthetic`.
The dataset file does not explain exactly how the examples were generated.

## Use in This Project

The current context-based experiments use the prompt, context, and response.
The `label` field is the target the models are trained or evaluated against.

The fields `hallucination_type`, `factuality_score`, and `confidence_score`
are not used as model inputs.

## Limitations

The dataset has relatively few unique examples and contexts. As a result,
strong performance on this dataset does not necessarily mean a model will
perform as well on new, real-world answers.

Related examples can also appear in both training and test sets if the data
is split randomly. The project’s `data_leakage_analysis.ipynb` examines this
risk and evaluates a grouped split.
