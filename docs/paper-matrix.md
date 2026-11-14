# Hallucination Detection Paper Matrix

This matrix links published methods to experiments that can be reproduced in this project. The project focuses first on **grounded hallucination detection**: given a context passage and a response, decide whether the response is supported.

## Comparison

| Work | Evidence | Method | Evaluation contribution | Relevance to this project | Main limitation for reproduction |
| --- | --- | --- | --- | --- | --- |
| SciHal25 overview (Li et al., 2025) | Scientific abstracts | Shared task with coarse and fine labels | Weighted F1; expert-labelled scientific claims | Defines the project's NLI framing and primary metric | Current project data are synthetic and binary |
| Schopf et al. (2025) | Scientific abstracts | Fine-tuned ModernBERT and DeBERTa-v3-large ensemble | Best SciHal25 coarse test result | Closest published model design to reproduce | Ensemble training is relatively expensive |
| Galimzianova et al. (2025) | Scientific abstracts | NLI-pretrained DeBERTa-v3 fine-tuning | Shows direct NLI fine-tuning can beat more complex pipelines | Strong single-model experiment | Needs a pretrained checkpoint and transformer training |
| Carla and Uban (2025) | Scientific abstracts | SciBERT with contrastive learning | Tests domain-specific representations | Motivates a SciBERT comparison | Current dataset is not limited to science |
| SummaC (Laban et al., 2022) | Source documents | Sentence-level NLI consistency scoring | Aggregates evidence/claim consistency | Useful pretrained NLI baseline | Designed for summarization |
| TRUE (Honovich et al., 2022) | Grounding documents | Standardized factual-consistency evaluation | Emphasizes example-level evaluation across datasets | Supports cross-dataset testing and common metrics | It is a benchmark collection, not one detector |
| SelfCheckGPT (Manakul et al., 2023) | Multiple sampled responses | Black-box self-consistency | Reference-free hallucination detection | Future comparison when no context is available | Repeated agreement does not guarantee truth |
| AlignScore (Zha et al., 2023) | Arbitrary source and output text | Unified alignment model trained across seven tasks | Evaluated across many factual-consistency datasets | Candidate pretrained scoring baseline | Larger than the project's classical baselines |
| RAGTruth (Niu et al., 2024) | Retrieved passages | Manually annotated RAG responses, including spans | Nearly 18,000 naturally generated responses | Strong candidate external dataset | Label format differs from the current binary dataset |
| MiniCheck (Tang et al., 2024) | Grounding documents | Small fact-checking model trained on synthetic errors | Efficient factuality checking and LLM-AggreFact | Candidate efficient pretrained detector | Reproduction may require more compute than a baseline |

## Research questions

1. Does an evidence-aware NLI model outperform lexical baselines on unseen contexts?
2. How much does performance change between random and context-grouped evaluation?
3. Do pretrained factual-consistency models transfer better than fine-tuning on 192 examples?
4. Which error types remain difficult: numbers, entities, dates, negation, unsupported claims, or overgeneralization?
5. Are accuracy gains worth the training time, inference time, and model size?

## Model order

1. Majority, word-overlap, TF-IDF logistic regression, and linear SVM baselines.
2. Pretrained NLI scoring without project-specific fine-tuning.
3. BERT and RoBERTa pair classification.
4. DeBERTa-v3 NLI fine-tuning.
5. SciBERT comparison.
6. Optional AlignScore or MiniCheck transfer experiment.
7. Reference-free methods only after the grounded experiments are complete.

## Reading notes required for each paper

Each literature entry must record:

- problem and hallucination definition;
- dataset and labels;
- evidence available to the detector;
- model and training objective;
- split strategy and leakage risks;
- metrics and main results;
- limitations;
- exact experiment, if any, that this project will reproduce.

## Primary sources

- Li et al. (2025), SciHal25 overview: https://aclanthology.org/2025.sdp-1.29/
- Schopf et al. (2025), NLI fine-tuning: https://aclanthology.org/2025.sdp-1.33/
- Galimzianova et al. (2025), DeBERTa NLI: https://aclanthology.org/2025.sdp-1.34/
- Manakul et al. (2023), SelfCheckGPT: https://aclanthology.org/2023.emnlp-main.557/
- Laban et al. (2022), SummaC: https://doi.org/10.1162/tacl_a_00453
- Honovich et al. (2022), TRUE: https://aclanthology.org/2022.naacl-main.287/
- Zha et al. (2023), AlignScore: https://aclanthology.org/2023.acl-long.634/
- Niu et al. (2024), RAGTruth: https://aclanthology.org/2024.acl-long.585/
- Tang et al. (2024), MiniCheck: https://arxiv.org/abs/2404.10774
