# Additional Literature Review

## 1. SelfCheckGPT: Hallucination Detection Without a Reference

### Main idea

Manakul et al. (2023) propose SelfCheckGPT, a method that generates
multiple responses to the same prompt and checks whether the facts in
those responses are consistent. If the responses disagree or contradict
one another, the original answer may contain a hallucination.

The method does not require an external reference database, but it
does require multiple generated responses.

### Connection to our project

SelfCheckGPT is relevant to our project because we are investigating
how to detect hallucinations when a context passage is not supplied.
It offers a different approach from our current BERT and RoBERTa
classifiers.

### Limitation

Consistent answers are not necessarily correct: a model could repeat
the same incorrect fact in several responses. The method also relies
on generated responses, so it should be discussed with the supervisor,
who advised against relying on the same model to check its own answer.

## 2. SummaC: Checking Generated Text Against a Source

### Main idea

Laban et al. (2022) propose SummaC to check whether a generated
summary is consistent with its source document. The method compares
sentences from the summary and the source using natural language
inference (NLI), then combines the comparison scores.

NLI is a way of examining the relationship between two pieces of text,
such as whether one supports or contradicts the other.

### Connection to our project

SummaC is relevant because our project also compares generated text
with supplied context. It provides an example of how NLI can be used
to check whether information in generated text is supported by a source.

### Limitation

The paper evaluates generated summaries against source documents.
Our current project tests question–answer examples, so the paper's
results do not establish how well SummaC would work on our dataset.

## Comparison

| Paper | Evidence used | Relevance to our project |
| --- | --- | --- |
| SelfCheckGPT | Multiple generated responses | A possible approach when no reference context is supplied |
| SummaC | A source document | An approach for checking generated text against supplied evidence |

These papers describe research methods. Neither method has been
implemented or tested in our project as part of this review.

## References

- Manakul, P., Liusie, A., & Gales, M. (2023). SelfCheckGPT:
  Zero-resource black-box hallucination detection for generative large
  language models. In *Proceedings of the 2023 Conference on Empirical
  Methods in Natural Language Processing* (pp. 9004–9017).
  https://doi.org/10.18653/v1/2023.emnlp-main.557

- Laban, P., Schnabel, T., Bennett, P. N., & Hearst, M. A. (2022).
  SummaC: Re-visiting NLI-based models for inconsistency detection in
  summarization. *Transactions of the Association for Computational
  Linguistics, 10*, 163–177. https://doi.org/10.1162/tacl_a_00453
