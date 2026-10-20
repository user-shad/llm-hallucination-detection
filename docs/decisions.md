# Decisions

COSC 497 Senior Design Project I
Project: LLM Hallucination Detection
Date: 20 October 2026

## Topic

On 9 September 2026 the group accepted Dr. Zeyar's topic, LLM Hallucination Detection, after AI-powered Smart Campus Assistant was withdrawn. He asked Dr. Jamal to set that title. The paper he sent is https://aclanthology.org/2025.sdp-1.29/.

## Dataset

On 21 September 2026 Dr. Zeyar wrote that the competition dataset was hard to download. The project uses the Kaggle dataset he named: mmumairkhattak/llm-hallucination-detection-nlp-and-ai-safety. That file is in this repository.

## Testing

The file has 5,000 rows and 192 unique prompt, context, and answer examples. Copies are removed before scoring. The test is 20% of the unique rows, 39 rows, with random_state 42.

The columns hallucination_type, factuality_score, and confidence_score are not used as inputs.

## Method

The current method calls an answer hallucinated when fewer than 60% of its words appear in the context. On the 39 unique test rows this scores 97%.

A model trained on the full prompt, context, and answer scores 54%. A model trained only on words that are in the answer and missing from the context scores 97%. The useful signal is the extra words in the answer, so the word check stays the current method.
