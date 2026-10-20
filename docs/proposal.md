# Proposal

COSC 497 Senior Design Project I
Project: LLM Hallucination Detection
Supervisor: Dr. U Zeyar Aung
Date: 20 October 2026

## Problem

An AI answer can repeat a source and still change one fact. The project detects whether an answer stays faithful to a context passage or is hallucinated.

## Data

The dataset is the Kaggle set named by Dr. Zeyar. It has 5,000 rows and 192 unique prompt, context, and answer examples. Repeated rows are removed before testing.

## Method

Compare the answer with the context. The current method calls an answer hallucinated when fewer than 60% of its words appear in the context. On 39 unique test rows this scores 97% accuracy. A model that uses only the words present in the answer and absent from the context also scores 97%. A model trained on the full mixed text scores 54%.

## Work still to complete

Literature and patent review, design and cost, standards and ethics, project plan, meeting minutes, final report, and presentation.