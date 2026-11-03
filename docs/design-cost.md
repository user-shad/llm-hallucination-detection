# Design and cost

COSC 497 Senior Design Project I
Project: LLM Hallucination Detection
Supervisor: Dr. U Zeyar Aung
Date: 3 November 2026

The Gantt plan places design and cost in the weeks of 3 November and 10 November. This note records the detector design for SDP I and what it costs to build. The same project continues in SDP II. Removing the context column, and comparing ChatGPT with Gemini, stay out of this design.

## What the system does

The detector reads one row and returns one label.

Inputs, from the Kaggle file already in `data/`:

- `prompt`, the question
- `context`, the passage the answer should follow
- `response`, the candidate answer

Output:

- `faithful` or `hallucinated`

These columns are not inputs. They describe the label or would leak it: `hallucination_type`, `factuality_score`, and `confidence_score`.

The file has 5,000 rows and 192 unique prompt, context, and response examples. Copies are removed before any score. The test set is 20% of the unique rows, 39 rows, with `random_state` 42. The other 153 unique rows are the training set. Train and test stay disjoint.

## Design now in the repository

`baseline.ipynb` is the reference design. It has three checks, all scored on the same 39 rows.

1. Word check. Lower-case the answer and the context, keep letters and digits, and measure what fraction of the answer's words also appear in the context. If that fraction is under 0.60, the label is `hallucinated`. Accuracy 97%. For hallucinated answers, precision 100% and recall 95%.
2. TF-IDF and logistic regression on the prompt, context, and answer joined into one string. Accuracy 54%.
3. The same model trained only on words that are in the answer and missing from the context. Accuracy 97%.

The word check stays the reference result. It fails when an answer copies the context and changes one fact, because most of the words still match. That limit is part of the design record, not a number to hide.

## Next design

On 26 October Dr. Zeyar accepted the reference result and asked for BERT and RoBERTa next, with the context column still provided. The literature review points the same way. SciHal teams that fine-tuned an encoder beat teams that only prompted a large model (Li et al., 2025; Schopf et al., 2025; Galimzianova et al., 2025).

The next notebook treats detection as natural language inference.

- Premise: `context`
- Hypothesis: `response`
- The `prompt` can be placed in front of the hypothesis so the model sees the question
- Label: `faithful` or `hallucinated`

Two models are trained separately on the 153 training rows:

- `bert-base-uncased` (Devlin et al., 2019)
- `roberta-base` (Liu et al., 2019)

Each model is fine-tuned for a few epochs with a small learning rate. The saved score is accuracy, precision, and recall on the same 39 test rows used by the word check, plus weighted F1 so the report can sit next to the SciHal metric. The word-check score on those rows stays in the table. A new model is kept only when it catches the one-fact changes the word check misses, or when it beats 97% accuracy without using the leaked columns.

DeBERTa-v3 and SciBERT appear in the SciHal reports. They are optional comparisons after BERT and RoBERTa, because those two are the models the supervisor named.

```text
prompt + context + response
            |
            v
   drop duplicate rows
            |
            v
   153 train rows | 39 test rows
            |
            +-- word check (reference, already run)
            |
            +-- BERT fine-tune
            |
            +-- RoBERTa fine-tune
            |
            v
   faithful / hallucinated
   same metrics on the same 39 rows
```

## Left for a later design

- A detector that sees only the question and the answer, with no context column. Dr. Zeyar said this is a later step. Building a context with another model is part of that step.
- Asking ChatGPT or Gemini to judge an answer. He said that comparison is last, and that one model should not judge an answer it may have written.
- Taxonomy 1's third label, unverifiable. The current file has only `faithful` and `hallucinated`.
- Taxonomy 2, the eight-way label set. It is the harder SciHal task and is outside SDP I.

## Cost

Money:

- The dataset is already in the repository under the MIT license. No purchase.
- Python, pandas, scikit-learn, PyTorch, and Hugging Face Transformers are free.
- BERT-base and RoBERTa-base weights are free to download.
- No paid API is used in this design. ChatGPT and Gemini are not called.
- The Computer Science GPU form is optional. One student submits it for the group only if a laptop cannot finish the fine-tune. The form is the one sent by Khloud AlZaabi on 7 October.

Compute:

- The word check and the two logistic regression models already ran in `baseline.ipynb` on a normal machine.
- The training set is 153 short rows. Fine-tuning BERT-base or RoBERTa-base on that set is small. A laptop CPU can run it. A free Colab GPU or a department GPU makes it faster. It does not need a paid server.

Time, from the Gantt plan for COSC 497:

- This design note: weeks of 3 November and 10 November.
- BERT and RoBERTa on the current unique rows: the detector work carried from the weeks of 20 October and 27 October into the week of 10 November.
- The no-context detector: weeks of 10 November and 17 November.
- Final report: weeks of 17 November and 24 November.
- Presentation: weeks of 24 November and 1 December.

People:

- Four students share the repository. Each member still has to commit their own part. This design note does not replace that.

## References

Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019*, pages 4171–4186.

Galimzianova, D., Boriskin, A. and Arshinov, G. (2025). From RAG to reality: Coarse-grained hallucination detection via NLI fine-tuning. In *Proceedings of the Fifth Workshop on Scholarly Document Processing (SDP 2025)*. https://aclanthology.org/2025.sdp-1.34/

Li, D., Palfi, B., Zhang, C., Subramanian, J., Raudaschl, A., Kakita, Y., De Waard, A., Afzal, Z. and Tsatsaronis, G. (2025). Overview of the SciHal25 shared task on hallucination detection for scientific content. In *Proceedings of the Fifth Workshop on Scholarly Document Processing (SDP 2025)*, pages 307–315. https://aclanthology.org/2025.sdp-1.29/

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L. and Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv:1907.11692.

Schopf, T., Vladika, J., Färber, M. and Matthes, F. (2025). Natural language inference fine-tuning for scientific hallucination detection. In *Proceedings of the Fifth Workshop on Scholarly Document Processing (SDP 2025)*. https://aclanthology.org/2025.sdp-1.33/
