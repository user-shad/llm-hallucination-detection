# Literature review

COSC 497 Senior Design Project I
Project: LLM Hallucination Detection
Supervisor: Dr. U Zeyar Aung
Date: 3 November 2026

Dr. Zeyar asked the group, on 14 September and again on 26 October, to read how other people detect hallucinations and to follow those methods before changing them. The paper he sent is Li et al. (2025). This note records what that paper and the published SciHal 2025 team reports say, and what the group can use on the current dataset.

## What a hallucination is

Li et al. (2025) describe a hallucination as a claim in a generated scientific answer that is not supported by the references. They follow earlier surveys in separating two cases (Huang et al., 2023; Zhang et al., 2023):

- An intrinsic hallucination contradicts the given reference.
- An extrinsic hallucination adds material the reference does not support.

The current project is the first case. The detector compares an answer with a context passage. An answer that copies the passage and changes one fact, such as a city, a date, or a number, is the kind of intrinsic error already seen in `baseline.ipynb`.

## The paper the supervisor assigned

Li et al. (2025) overview the SciHal shared task at the 2025 Scholarly Document Processing workshop. The data come from a real research assistant: 488 research questions, about 10,000 reference abstracts, and 4,592 claims taken from the assistant's answers. Each instance has a question, an answer, one extracted claim, and the reference abstracts. The split is 3,592 training claims, 500 validation claims, and 500 test claims. The score is weighted F1.

The same claims are labeled twice. These are the two taxonomies from the 14 September meeting.

Taxonomy 1, Subtask 1, has three labels:

- Entailment: the claim is supported by at least one passage and is not contradicted by another.
- Contradiction: a passage states a different entity, number, or relation.
- Unverifiable: the references do not cover the claim.

On the full Subtask 1 set, 38.71% of claims are entailment, 36.89% contradiction, and 24.40% unverifiable.

Taxonomy 2, Subtask 2, splits the errors into eight labels: entailment, unrelated and unverifiable, related but unverifiable, entity error, numeric error, opposite meaning, missing information, and misinterpretation. Dr. Zeyar said to start with Taxonomy 1 because it has fewer categories. The published scores support that choice. The best Subtask 1 test score is 0.62 weighted F1. The best Subtask 2 test scores are about 0.47.

Expert annotators also disagree. On the harder half of the data, two experts agreed on 0.55 of Subtask 1 labels (Cohen's kappa 0.297) and on 0.44 of Subtask 2 labels. Each claim took about 7 minutes to label (Li et al., 2025). Hallucination labels are uncertain even for people, so a detector score on a small test set has to be read with that limit in mind.

The competition file was not used in this repository. On 21 September Dr. Zeyar wrote that the download was too difficult and named the Kaggle dataset that is now in `data/`. Subtask 1 of SciHal is therefore the guide for the method, not the file the group is scoring.

## What the competing teams did

Five team reports were published. Li et al. (2025) summarize them as follows.

Schopf et al. (2025) treated the task as natural language inference. The reference is the premise and the claim is the hypothesis. They fine-tuned ModernBERT and DeBERTa-v3-large and combined them in a weighted ensemble. That system placed first on Subtask 1, with 0.60 weighted F1 on the public validation set and 0.62 on the private test set. Fine-tuning beat prompting, and training data that resembled the task mattered.

Galimzianova et al. (2025) also fine-tuned DeBERTa-v3 that had already been trained on natural language inference. Direct fine-tuning beat a pipeline that added extra features, and it beat prompting a large language model. On the public Subtask 1 board their score was 0.58. On the private board they placed third at 0.59. Their conclusion is the one that fits this SDP: on a small scientific set, fine-tuning an encoder is more accurate and cheaper than prompting a large model.

Cao et al. (2025) prompted larger instruction-tuned models with zero-shot, few-shot, and chain-of-thought prompts, then trained a logistic regression classifier on the model's hidden states without fine-tuning the model itself. They scored 0.59 on public Subtask 1 and 0.60 on the private test set.

Le and Thin (2025) used prompt engineering only. A two-step prompt, fine label first and then the coarse label, beat predicting the coarse label directly. Their best model, Gemini 2.5 Flash, scored 0.56 and 0.44 weighted F1 on the two subtasks in their own report. On the public board they scored 0.59 on Subtask 1.

Carla and Uban (2025) combined SciBERT with contrastive learning so that a claim and a reference can match even when the wording differs. This is the SciBERT method Dr. Zeyar named on 14 September. Their public Subtask 1 score was 0.51, below the fine-tuned DeBERTa systems.

The discussion in Li et al. (2025) draws one conclusion from these results: the strongest systems fine-tuned transformer encoders. Prompting a large model was tried, and it did not win.

## What this means for the current detector

The repository now uses a word-overlap rule and two logistic regression models on the synthetic Kaggle file. On 39 unique test rows the word check scores 97% accuracy. That number is not comparable to the SciHal weighted F1 scores near 0.60. The label sets differ, the text is synthetic, and the test has 39 rows.

The word check also matches a weakness the SciHal taxonomy names. Entity errors and numeric errors can keep most of the original words. Overlap then calls the answer faithful. The missed rows in `baseline.ipynb` are that pattern.

The next method Dr. Zeyar asked for on 26 October is BERT and RoBERTa, with the context column still provided. That request matches the winning SciHal approach:

- Treat the context as the premise and the answer as the hypothesis.
- Fine-tune an encoder. BERT and RoBERTa are the models he named. DeBERTa-v3 and SciBERT are the encoders the published teams actually trained.
- Keep the label binary for now, faithful or hallucinated, because that is the column in the current file. Taxonomy 1's third label, unverifiable, can be added only if the data mark claims that the context does not cover.
- Leave prompting of ChatGPT or Gemini until the end. He said not to ask one model to judge an answer that the same model may have written, and the SciHal results say fine-tuned encoders were stronger than prompting.

Li et al. (2025) also warn that their own split reused questions across train and test. This project already drops duplicate prompt, context, and answer rows before scoring. The train and test split should stay disjoint in the same way when BERT or RoBERTa is trained.

## References

Carla, C. and Uban, A. S. (2025). SciBERT meets contrastive learning: A solution for scientific hallucination detection. In *Proceedings of the Fifth Workshop on Scholarly Document Processing (SDP 2025)*.

Cao, Y., Yu, C.-N. and Subbalakshmi, K. P. (2025). Detecting hallucinations in scientific claims by combining prompting strategies and internal state classification. In *Proceedings of the Fifth Workshop on Scholarly Document Processing (SDP 2025)*.

Galimzianova, D., Boriskin, A. and Arshinov, G. (2025). From RAG to reality: Coarse-grained hallucination detection via NLI fine-tuning. In *Proceedings of the Fifth Workshop on Scholarly Document Processing (SDP 2025)*. https://aclanthology.org/2025.sdp-1.34/

Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng, W., Feng, X., Qin, B. and others (2023). A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. arXiv:2311.05232.

Le, K. N.-A. and Thin, D. V. (2025). A.M.P at SciHal2025: Automated hallucination detection in scientific content via LLMs and prompt engineering. In *Proceedings of the Fifth Workshop on Scholarly Document Processing (SDP 2025)*.

Li, D., Palfi, B., Zhang, C., Subramanian, J., Raudaschl, A., Kakita, Y., De Waard, A., Afzal, Z. and Tsatsaronis, G. (2025). Overview of the SciHal25 shared task on hallucination detection for scientific content. In *Proceedings of the Fifth Workshop on Scholarly Document Processing (SDP 2025)*, pages 307–315, Vienna, Austria. Association for Computational Linguistics. https://aclanthology.org/2025.sdp-1.29/

Schopf, T., Vladika, J., Färber, M. and Matthes, F. (2025). Natural language inference fine-tuning for scientific hallucination detection. In *Proceedings of the Fifth Workshop on Scholarly Document Processing (SDP 2025)*. https://aclanthology.org/2025.sdp-1.33/

Zhang, Y., Li, Y., Cui, L., Cai, D., Liu, L., Fu, T., Huang, X., Zhao, E., Zhang, Y., Chen, Y. and others (2023). Siren's song in the AI ocean: A survey on hallucination in large language models. arXiv:2309.01219.

## Still separate from this note

The proposal also lists a patent review. This document does not cover patents.
