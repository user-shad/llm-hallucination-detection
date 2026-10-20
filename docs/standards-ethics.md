# Standards and ethics

COSC 497 Senior Design Project I
Project: LLM Hallucination Detection
Date: 20 October 2026

The final report has to cover the ethical responsibility of a project student and of a professional engineer, and the professional standards that relate to this work.

## Project student

- The test score is reported on the 39 unique test rows. The 5,000-row file repeats the same examples, so those copies are not treated as new evidence.
- The columns hallucination_type, factuality_score, and confidence_score are not used as inputs. They describe the label, so using them would leak the answer.
- The dataset and the SciHal paper are cited. The dataset license in this repository is MIT.
- The notebook, proposal, progress report, and this note are committed to the group repository.

## Professional engineer

A detector that calls a false answer faithful can pass on a wrong fact. The current word check fails when an answer copies the context and changes one fact, such as a city, a date, or a number. That limit is part of the result. The 97% score is for this dataset only. It is not a claim about live models or other data.

## Standards that relate to this work

- ACM Code of Ethics and Professional Conduct: state the result and its limit honestly.
- IEEE Code of Ethics: base the claim on the test that was actually run.
- NIST AI Risk Management Framework: record how the detector was tested and where it fails.
- Dataset license: the Kaggle dataset is included with its MIT license and source.
