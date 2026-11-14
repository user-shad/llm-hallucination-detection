"""Evaluate classical hallucination baselines with context-grouped folds."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import (  # noqa: E402
    ID_TO_LABEL,
    build_input,
    fold_metrics,
    grouped_folds,
    load_dataset,
    summarize_folds,
)


TOKEN_PATTERN = re.compile(r"\b\w+\b")


def overlap_predictions(frame: pd.DataFrame, threshold: float = 0.60) -> np.ndarray:
    """Reproduce the repository's original word-overlap rule."""
    predictions = []
    for context, response in zip(frame["context"], frame["response"]):
        context_words = set(TOKEN_PATTERN.findall(context.lower()))
        response_words = TOKEN_PATTERN.findall(response.lower())
        supported = sum(word in context_words for word in response_words)
        ratio = supported / max(1, len(response_words))
        predictions.append(0 if ratio >= threshold else 1)
    return np.asarray(predictions)


def model_factories() -> dict:
    return {
        "majority": lambda: DummyClassifier(strategy="most_frequent"),
        "tfidf_logistic_regression": lambda: Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", max_iter=2000, random_state=42
                    ),
                ),
            ]
        ),
        "tfidf_linear_svm": lambda: Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
            ]
        ),
    }


def evaluate(data_path: Path, output_dir: Path, condition: str) -> None:
    frame = load_dataset(data_path)
    texts = build_input(frame, condition)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_predictions = []
    factories = model_factories()

    for fold, (train_indices, test_indices) in enumerate(grouped_folds(frame), start=1):
        train = frame.iloc[train_indices]
        test = frame.iloc[test_indices]
        y_train = train["label_id"].to_numpy()
        y_test = test["label_id"].to_numpy()

        model_names = [
            "majority",
            "word_overlap",
            "tfidf_logistic_regression",
            "tfidf_linear_svm",
        ]
        for model_name in model_names:
            start = time.perf_counter()
            if model_name == "word_overlap":
                predictions = overlap_predictions(test)
            else:
                model = factories[model_name]()
                model.fit(texts.iloc[train_indices], y_train)
                predictions = model.predict(texts.iloc[test_indices])
            elapsed = time.perf_counter() - start

            metrics = fold_metrics(y_test, predictions)
            metrics.update(
                {
                    "model": model_name,
                    "fold": fold,
                    "input_condition": condition,
                    "fit_and_predict_seconds": elapsed,
                    "train_examples": len(train_indices),
                    "test_examples": len(test_indices),
                    "test_contexts": test["context"].nunique(),
                }
            )
            all_metrics.append(metrics)

            for row_position, prediction in zip(test_indices, predictions):
                row = frame.iloc[row_position]
                all_predictions.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "id": row.get("id", row_position),
                        "context": row["context"],
                        "prompt": row["prompt"],
                        "response": row["response"],
                        "true_label": row["label"],
                        "predicted_label": ID_TO_LABEL[int(prediction)],
                        "correct": int(prediction) == int(row["label_id"]),
                        "hallucination_type": row.get("hallucination_type", "unknown"),
                        "domain": row.get("domain", "unknown"),
                    }
                )

    metrics_frame = pd.DataFrame(all_metrics)
    predictions_frame = pd.DataFrame(all_predictions)
    summaries = []
    for model_name, model_metrics in metrics_frame.groupby("model"):
        summary = summarize_folds(model_metrics).reset_index(names="metric")
        summary.insert(0, "model", model_name)
        summaries.append(summary)
    summary_frame = pd.concat(summaries, ignore_index=True)

    metrics_frame.to_csv(output_dir / "classical_fold_metrics.csv", index=False)
    predictions_frame.to_csv(output_dir / "classical_oof_predictions.csv", index=False)
    summary_frame.to_csv(output_dir / "classical_summary.csv", index=False)

    run_config = {
        "data": str(data_path),
        "raw_rows": 5000,
        "unique_examples": len(frame),
        "context_groups": int(frame["context"].nunique()),
        "folds": 6,
        "group_column": "context",
        "input_condition": condition,
        "random_state": 42,
        "models": [
            "majority",
            "word_overlap",
            "tfidf_logistic_regression",
            "tfidf_linear_svm",
        ],
    }
    (output_dir / "classical_run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    weighted_f1 = summary_frame[summary_frame["metric"] == "weighted_f1"]
    print(weighted_f1[["model", "mean", "std"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data/llm_hallucination_detection_dataset.jsonl", type=Path
    )
    parser.add_argument("--output-dir", default="outputs/classical", type=Path)
    parser.add_argument(
        "--condition",
        default="context_response",
        choices=[
            "context_response",
            "prompt_context_response",
            "prompt_response",
            "response_only",
        ],
    )
    args = parser.parse_args()
    evaluate(args.data, args.output_dir, args.condition)


if __name__ == "__main__":
    main()
