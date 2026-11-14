"""Evaluate a pretrained NLI model without project-specific fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import (  # noqa: E402
    ID_TO_LABEL,
    fold_metrics,
    grouped_folds,
    load_dataset,
    summarize_folds,
)


DEFAULT_CHECKPOINT = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def find_entailment_index(model) -> int:
    labels = {
        int(index): str(label).lower()
        for index, label in model.config.id2label.items()
    }
    matches = [index for index, label in labels.items() if "entail" in label]
    if len(matches) != 1:
        raise ValueError(f"Could not identify one entailment label from {labels}")
    return matches[0]


def predict_all(model, tokenizer, frame, batch_size: int, max_length: int):
    entailment_index = find_entailment_index(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    probability_batches = []

    start = time.perf_counter()
    with torch.no_grad():
        for start_index in range(0, len(frame), batch_size):
            batch = frame.iloc[start_index : start_index + batch_size]
            encoded = tokenizer(
                batch["context"].tolist(),
                batch["response"].tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())

    probabilities = np.concatenate(probability_batches, axis=0)
    predicted_nli_labels = probabilities.argmax(axis=1)
    predictions = np.where(predicted_nli_labels == entailment_index, 0, 1)
    elapsed = time.perf_counter() - start
    return predictions, probabilities[:, entailment_index], elapsed, entailment_index


def evaluate(args) -> None:
    frame = load_dataset(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    predictions, entailment_scores, elapsed, entailment_index = predict_all(
        model, tokenizer, frame, args.batch_size, args.max_length
    )

    fold_rows = []
    prediction_rows = []
    for fold, (_, test_indices) in enumerate(grouped_folds(frame), start=1):
        fold_predictions = predictions[test_indices]
        fold_truth = frame.iloc[test_indices]["label_id"].to_numpy()
        metrics = fold_metrics(fold_truth, fold_predictions)
        metrics.update(
            {
                "model": "pretrained_nli",
                "checkpoint": args.checkpoint,
                "fold": fold,
                "test_examples": len(test_indices),
            }
        )
        fold_rows.append(metrics)

        for row_position in test_indices:
            row = frame.iloc[row_position]
            prediction_rows.append(
                {
                    "model": "pretrained_nli",
                    "fold": fold,
                    "id": row.get("id", row_position),
                    "context": row["context"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "true_label": row["label"],
                    "predicted_label": ID_TO_LABEL[int(predictions[row_position])],
                    "entailment_probability": float(entailment_scores[row_position]),
                    "correct": int(predictions[row_position]) == int(row["label_id"]),
                    "hallucination_type": row.get("hallucination_type", "unknown"),
                    "domain": row.get("domain", "unknown"),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(fold_rows)
    summary = summarize_folds(metrics_frame).reset_index(names="metric")
    metrics_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        args.output_dir / "oof_predictions.csv", index=False
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    config = {
        "checkpoint": args.checkpoint,
        "data": str(args.data),
        "output_dir": str(args.output_dir),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "unique_examples": len(frame),
        "context_groups": int(frame["context"].nunique()),
        "entailment_label_index": entailment_index,
        "inference_seconds": elapsed,
        "decision_rule": "entailment -> faithful; contradiction or neutral -> hallucinated",
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(summary[summary["metric"].isin(["weighted_f1", "macro_f1", "accuracy"])])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--data", type=Path, default=Path("data/llm_hallucination_detection_dataset.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nli_baseline"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
