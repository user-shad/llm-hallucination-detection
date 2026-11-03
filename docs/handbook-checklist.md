# Handbook checklist

COSC 497 Senior Design Project I
Project: LLM Hallucination Detection
Date: 3 November 2026

This note matches the repository to the assessment forms in `COSC 497 498 SDPI & SDPII Handbook_August 2026`. The detailed SDPI supervisor form scores GitHub at 15 points, with a separate column for each student. That matches Dr. Jamal's 19 October announcement. The summary table earlier in the handbook still prints GitHub as 7.5%. Use the detailed form.

The examiner form does not score GitHub. The examiner scores the report (20), the presentation (20), the live demo (25), and the questions (35).

Dr. Zeyar said GitHub commits replace the signed daily logbook. The handbook sentence that still asks for a paper logbook is not the rule for this group.

## Supervisor form: GitHub record, 15 points

| Criterion | Points | What the repository has | What is still missing |
| --- | --- | --- | --- |
| Repository organization and documentation | 4 | `README.md` names the team, the supervisor, and the task. `data/DATA_README.md` describes the file. | The README does not say how to install the libraries or how to run `baseline.ipynb`. |
| Commit history and progress | 5 | The notebook was built on 19–20 October. Minutes, the literature review, and the design note were added later. | The history is two short bursts. The form asks for regular commits across the semester. |
| Individual contributions | 4 | `docs/minutes.md` names who spoke in each meeting. | Every commit is from `user-shad`. Wadha, Mouza, and Maram have no commits. This is task 2. |
| Version-control practices | 2 | Work is committed on `main`. | There are no branches and no GitHub issues. |

## Files already on GitHub, and the handbook item they cover

| File | Handbook item |
| --- | --- |
| `docs/minutes.md` | Minutes of meetings. The handbook says these go on GitHub. |
| `docs/literature-review.md` | Background reading for report chapter 1. Eight papers are cited. The report itself needs at least ten, from papers and books, with an access date on any web source. No Wikipedia. |
| `docs/design-cost.md` | Design options, the chosen next design, and cost. Report chapter 4 asks for more than one option and a reason for the choice. |
| `docs/gantt.md` | Project plan. The handbook figure is a bar chart in Excel, with dependencies. This file is a table, not that chart. |
| `docs/decisions.md` | Decisions taken. |
| `docs/progress-report.md` | The progress document Dr. Jamal asked to keep on GitHub. He said a separate progress-report submission is not required. |
| `docs/proposal.md` | The project proposal. |
| `docs/standards-ethics.md` | Professional ethics and the IEEE and ACM codes. Report chapter 7 also asks for environmental impact. That section is not written yet. |
| `baseline.ipynb` | Implementation and the tests behind the 97% word-check result. |
| `data/llm_hallucination_detection_dataset.jsonl` | The dataset. `data/LICENSE` is the MIT license. |
| `README.md` | The GitHub link the report must include, with each student's contribution. |

## Handbook items with no file yet

These are the gaps to close before the report and the demo.

- Requirements and system specification, as their own notes. The design note lists inputs and outputs. The handbook still wants a requirements note and a specification the supervisor can approve.
- Patent search. Chapter 1 asks for a literature and patent search. The literature review does not cover patents.
- A record that no equipment or software was purchased. The handbook asks for purchased items to be written on GitHub. The cost note says the tools are free. A one-line decision would make that explicit.
- AI prompts used to generate code, if any were used. The report chapter on the GitHub account asks for them.
- Setup instructions: Python libraries and how to open the notebook.
- Final report. Due at the end of week 14. One submission for the group. The handbook title is the SDPI final report. Dr. Jamal's filename is `COSC# SDP1 Final report`, using the group number from the Blackboard table.
- Presentation, live demo, and questions. These are not files. The demo has to show the detector on examples, including an answer that copies the context and changes one fact.

## Forms submitted with the report, not as GitHub files

- Risk assessment.
- Health and safety. This project is software only, so that box is the one to tick.
- Project clearance, after the signatures.

## What to do next on the repository

1. Add run instructions to `README.md`.
2. Add a short requirements and specification note, and ask Dr. Zeyar to confirm it.
3. Add the patent search, or a note that none was found, with the date of the search.
4. Have each member commit their own part so the 4-point individual-contribution line can be scored.
