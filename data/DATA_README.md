# LLM Hallucination Detection Dataset

## Overview

The LLM Hallucination Detection Dataset is a synthetic English dataset designed for hallucination detection, factuality classification, grounded response evaluation, and NLP research.

The dataset contains **5,000** prompt-context-response examples labeled as either `faithful` or `hallucinated`.

## Files

- `llm_hallucination_detection_dataset.csv`
- `llm_hallucination_detection_dataset.jsonl`
- `data_dictionary.csv`
- `dataset-metadata.json`
- `LICENSE`

## Main Target

The primary target column is:

- `label`

Possible values:

- `faithful`
- `hallucinated`

## Columns

| Column | Description |
|---|---|
| id | Unique row identifier |
| prompt | User question or instruction |
| context | Grounding evidence/context |
| response | Candidate AI response |
| label | Target label: faithful or hallucinated |
| hallucination_type | Type of hallucination or none |
| domain | Knowledge domain |
| task_type | NLP task category |
| evidence_available | Whether context is available |
| factuality_score | Synthetic factuality score |
| confidence_score | Synthetic confidence score |
| difficulty | Estimated difficulty |
| language | Record language |
| source_type | Synthetic data marker |
| created_year | Dataset creation year |

## Use Cases

- LLM hallucination detection
- Factuality classification
- Grounded response evaluation
- Prompt-response verification
- NLP model training
- AI safety research
- Chatbot response quality analysis
- Instruction-following evaluation

## Domains

- Science
- History
- Technology
- General medicine
- Business
- Geography
- General law
- General finance

## License

MIT License.

## Notes

This is a synthetic dataset created for research, education, and prototyping. It should be reviewed and expanded before use in production AI safety systems.
