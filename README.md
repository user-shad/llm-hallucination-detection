# LLM Hallucination Detection

COSC 497 Senior Design Project I

## Team
- Shahad Aljasmi
- Wadha Msaiqer
- Mouza Alshehhi
- Maram Alkatheeri

## Supervisor
Dr. U Zeyar Aung, Computer Science, Khalifa University

## Project
Build a detector that reads a prompt, a context passage, and an AI answer, then decides whether the answer is faithful or hallucinated.

## Setup

From this folder:

pip install pandas scikit-learn torch transformers notebook

PyTorch on this project was installed from the CPU build:

pip install torch --index-url https://download.pytorch.org/whl/cpu

The dataset is already in data/llm_hallucination_detection_dataset.jsonl.

## Run

baseline.ipynb is the word check and the two logistic regression models. It uses the context column.

bert_roberta.ipynb trains BERT and RoBERTa. One part uses the context. The later part uses only the question and the answer.

Open either notebook from this folder with:

python -m jupyter notebook

Run the cells from the top. The first data cell should print 153 39.

Do not commit the outputs folder or the downloaded model weights. Those are created during training and are not part of the project record.
