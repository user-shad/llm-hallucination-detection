# Patent search

COSC 497 Senior Design Project I
Project: LLM Hallucination Detection
Date of search: 8 November 2026
Source: Google Patents
Query: hallucination detection large language model

This is a record of published applications found on that date. It is not a legal opinion, and it is not a claim that no other patents exist.

## Applications found

US20250077777A1, also granted as US12387049B2. Intuit Inc. Semantic aware hallucination detection for large language models. Priority date 30 August 2023. A first model answers a question and rewrites that question in several ways. A second, different model answers those rewrites. A consistency score decides whether the first answer is a hallucination. https://patents.google.com/patent/US20250077777A1

US20250238457A1. LivePerson Inc. Techniques for detecting hallucination in machine-generated responses. Priority date 23 January 2025. The answer is produced with retrieval-augmented generation. A classifier then labels whether the answer contradicts the retrieved knowledge. https://patents.google.com/patent/US20250238457A1

US20250225337A1. Google LLC. Detection of hallucinations in large language model responses. Priority date 9 January 2024. The application was pending on the search date. If a response is judged to contain a hallucination, the model generates another response and the check is repeated. https://patents.google.com/patent/US20250225337A1

US20250103818A1. Bitvore Corp. Hallucination detection as a metric for determining accuracy of results for large language models in machine learning. The method compares an ordinary answer with an answer from an augmented system and turns the difference into a hallucination score. https://patents.google.com/patent/US20250103818A1

## Relation to this project

These filings detect hallucinations by asking a second model, by checking a retrieved knowledge base, by generating the answer again, or by comparing two versions of an answer. This project uses a word check against a context that is already in the dataset. The search did not return a filing that describes that word check. The current method does not copy these patented systems. Asking ChatGPT or Gemini to judge an answer, and building a missing context with another model, remain later steps.
