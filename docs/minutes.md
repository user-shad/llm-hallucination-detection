# Meeting minutes

COSC 497 Senior Design Project I
Project: LLM Hallucination Detection

These notes are taken from the Read report for 14 September and from the Teams transcripts for 21 September and 26 October. The transcripts are automatic, so a few words are unclear. Lines with no speaker name are not assigned to anyone. Mouza is not named as a speaker in any of the three transcripts.

## 14 September 2026, 2:00 PM

Supervisor meeting. The Read report times it as 2:00–2:07 PM. Named speakers are Wadha Msaiqer, Maram Alkatheeri, and Dr. U Zeyar Aung.

The new topic is detecting inaccurate answers from large language models such as ChatGPT and Gemini. The dataset is the link Dr. Zeyar had sent, and the group had opened it. He described two taxonomies. Taxonomy 1 has fewer categories, so the senior project should start there. Taxonomy 2 is the harder one.

He asked the group to download and read the competition paper he sent. The competition was already finished. The group should follow what other people did first, and change the method later. He named BERT or SciBERT as a way to recognize a hallucinated answer.

He also passed on Dr. Jamal's instruction for this year. There is no manual logbook. GitHub commits are the logbook, and every member has to commit, not only one person. He will check the commit records.

The group should meet weekly for the first two or three weeks, and later the meetings may be every two weeks. Monday at 2:00 does not work because Mouza cannot attend then. Tuesday at 1:00 conflicts with Wadha's class. A Tuesday 3:00 time was raised and did not fit. Dr. Zeyar said he would email his class and meeting times, and the group would propose one time that works for everyone.

## 21 September 2026, 4:00 PM

Supervisor meeting. The Teams recording is about 6 minutes and starts at 4:06 PM. Named speakers are Shahad Aljasmi, Wadha Msaiqer, Maram Alkatheeri, and Dr. U Zeyar Aung.

Dr. Zeyar asked the group to download and read the paper that was open in the meeting. The dataset he was showing was on Kaggle. Each person needs a Kaggle account. In the Data Explorer there are two subtasks. The group should use only Subtask 1 and ignore Subtask 2.

He asked them to download the sample submission, the test file, and three training batches. Train on those three batches, then score the model on the test file. The competition was already closed, so the results were to be shown to him rather than uploaded. The paper is the guide for how to build that first model. If the download was confusing, the group could message him on Teams instead of waiting until the next week.

The same afternoon, at 4:29 PM, he emailed that the competition download was too difficult. He named the Kaggle dataset that is now in this repository. Subtask 1 is not the dataset the project is using.

## 26 October 2026, 4:00 PM

Supervisor meeting. The Teams recording is about 7 and a half minutes and starts at 4:02 PM. Named speakers are Shahad Aljasmi, Wadha Msaiqer, Maram Alkatheeri, and Dr. U Zeyar Aung.

The team of four is registered with Dr. Jamal. Dr. Zeyar had thought the group was three, and he confirmed that four is correct. He had looked at the GitHub repository and the first results, and he said they are fine.

He gave two comments on the method. Using the context column is fine for now. Later the detector has to work without that given context, because a real case is only the question and the model's answer. Building a context with other models is a later step. The next models to try are BERT and RoBERTa.

He said not to ask one model to check an answer that the same model may have produced. Comparing several models, including ChatGPT and Gemini, is a last step. He also asked for a literature review: download and read papers on how other people detect hallucinations in large language models.

He told the group to ignore an email he had sent about the dataset. He had written to the wrong person, someone with the same name as the uploader but different research.

The next meeting should be next week or the week after, and there should be at least one or two meetings before the SDP I examination. Wadha said she will check the examination date on Blackboard. Dr. Zeyar repeated the examiner's name as Dr. Shamsi. The transcript of the sentence just before that is unclear.
