# Requirements and specification

COSC 497 Senior Design Project I
Project: LLM Hallucination Detection
Supervisor: Dr. U Zeyar Aung
Date: 8 November 2026

## Requirements

The detector reads one example and returns one label: faithful or hallucinated.

The current system reads three fields: prompt, context, and response. The context is the passage the answer should follow. The response is the candidate answer.

These fields are not inputs: hallucination_type, factuality_score, and confidence_score.

The score is reported on the 39 unique test rows, using random_state 42. Duplicate rows are removed before scoring.

A later version must work from the question and the answer only, with no context given. That version is not the current system.

## Specification

The current method is the word check. An answer is hallucinated when fewer than 60% of its words appear in the context. On the 39 test rows this scores 0.974 accuracy.

BERT and RoBERTa, trained with the context, score 0.872. A logistic regression model with no context scores 0.846. RoBERTa with no context scores 0.615. The word check stays the current method.

The system does not call ChatGPT or Gemini. It does not build a missing context with another model. It does not use Taxonomy 2.

The word check fails when an answer copies the context and changes one fact. The 0.974 score is for this dataset only.

This note still needs Dr. Zeyar's confirmation.
