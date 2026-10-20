# Progress report

COSC 497 Senior Design Project I
Project: LLM Hallucination Detection
Supervisor: Dr. U Zeyar Aung
Date: 20 October 2026

## What the group has done

The dataset has 5,000 rows, but only 192 unique prompt, context, and answer examples. The copies were removed before the final test.

Three checks were run on the same 39 unique test rows:

- Word check: an answer is called hallucinated when fewer than 60% of its words appear in the context. Accuracy 97%. For hallucinated answers, precision 100% and recall 95%.
- A model trained on the prompt, context, and answer mixed together. Accuracy 54%.
- A model trained only on the words that are in the answer and not in the context. Accuracy 97%.

The missed answers copy most of the context and change one fact, such as a city, a date, or a number.

## Decision

The useful signal is the extra words in the answer. The word check is the current method.

## Repository

https://github.com/user-shad/llm-hallucination-detection