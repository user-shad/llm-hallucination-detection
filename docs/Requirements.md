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

## Research and evaluation questions
How well can each method identify hallucinated and faithful answers?
Compare the word-overlap rule, logistic regression, BERT, and RoBERTa using the same test examples. Report accuracy, precision, recall, F1 score, and a confusion matrix.

Does providing the context improve detection?
Compare models that receive the prompt, context, and response with models that receive only the prompt and response. Use the same data splits for both.

Can the models handle questions and facts they have not seen during training?
Keep related examples based on the same passage or fact in the same split. This tests whether the models can generalize to new material and reduces the risk of data leakage.

Which types of hallucinations are hardest to detect?
Examine errors involving changed names, numbers, dates, negation, and unsupported claims. Also check whether the models incorrectly flag faithful paraphrases.

Which method offers the best balance between performance and complexity?
Compare the simple baselines with BERT and RoBERTa on detection quality, training time, and computational requirements. A simpler model may be preferable if its performance is comparable on a reliable test set.

## Specification

The current method is the word check. An answer is hallucinated when fewer than 60% of its words appear in the context. On the 39 test rows this scores 0.974 accuracy.

BERT and RoBERTa, trained with the context, score 0.872. A logistic regression model with no context scores 0.846. RoBERTa with no context scores 0.615. The word check stays the current method.

The system does not call ChatGPT or Gemini. It does not build a missing context with another model. It does not use Taxonomy 2.

The word check fails when an answer copies the context and changes one fact. The 0.974 score is for this dataset only.

This note still needs Dr. Zeyar's confirmation.

## Problem statement
Large language models can produce answers that sound convincing but contain information that is unsupported by the source material. The aim of this project is to develop and evaluate methods that determine whether an AI-generated answer is supported by a given context.

For each example, the system receives:
- Prompt: the question or instruction given to the language model.
- Context: the passage containing the information the answer should rely on.
- Response: the answer to be evaluated.
The system predicts one of two labels:
- Faithful: the response’s substantive claims are supported by the context.
- Hallucinated: the response contains at least one substantive claim that contradicts the context or is not supported by it.
For example, if the context says “Tokyo is the capital of Japan,” the response “Japan’s capital is Tokyo” is faithful. The response “Japan’s capital is Kyoto” is hallucinated because it contradicts the context.
## Current scope
The current project evaluates answers against a supplied context. It does not establish whether the context itself is factually correct, and it does not detect every possible kind of hallucination.

The current word-overlap baseline compares the response with the context and predicts “hallucinated” when fewer than 60% of the response’s words appear in the context. This is a prediction rule, not the definition of a hallucination: an answer can use many of the same words as the context while changing an important fact.

The project also experiments with logistic regression, BERT, and RoBERTa. Some experiments omit the context, but a system that works from only the question and answer remains a future project requirement. The fields hallucination_type, factuality_score, and confidence_score are excluded as model inputs.

## Limitations
The current dataset is small after duplicate removal, so the results should be described as performance on the evaluated dataset rather than proof of general performance on real-world answers. Ambiguous cases and answers that are only partly supported will need clear labeling guidance as the dataset is expanded.
