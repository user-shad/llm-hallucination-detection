"""Shared data and metric utilities for hallucination experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold


LABEL_TO_ID = {"faithful": 0, "hallucinated": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
DEDUPLICATION_COLUMNS = ["prompt", "context", "response"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load JSONL, validate labels, and remove exact text duplicates."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number}") from error

    frame = pd.DataFrame(records)
    required = set(DEDUPLICATION_COLUMNS + ["label"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    unknown = set(frame["label"].dropna().unique()) - set(LABEL_TO_ID)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")

    conflicting = (
        frame.groupby(DEDUPLICATION_COLUMNS, dropna=False)["label"].nunique() > 1
    )
    if conflicting.any():
        raise ValueError("An identical prompt-context-response triple has conflicting labels")

    frame = frame.drop_duplicates(DEDUPLICATION_COLUMNS).reset_index(drop=True)
    frame["label_id"] = frame["label"].map(LABEL_TO_ID).astype(int)
    return frame


def grouped_folds(frame: pd.DataFrame, n_splits: int = 6):
    """Yield deterministic folds with complete contexts held together."""
    group_count = frame["context"].nunique()
    if group_count < n_splits:
        raise ValueError(f"Need at least {n_splits} contexts; found {group_count}")

    splitter = GroupKFold(n_splits=n_splits)
    yield from splitter.split(frame, frame["label_id"], groups=frame["context"])


def build_input(frame: pd.DataFrame, condition: str) -> pd.Series:
    """Construct a documented model input condition."""
    if condition == "context_response":
        return "context: " + frame["context"] + " response: " + frame["response"]
    if condition == "prompt_context_response":
        return (
            "prompt: " + frame["prompt"]
            + " context: " + frame["context"]
            + " response: " + frame["response"]
        )
    if condition == "prompt_response":
        return "prompt: " + frame["prompt"] + " response: " + frame["response"]
    if condition == "response_only":
        return frame["response"]
    raise ValueError(f"Unknown input condition: {condition}")


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return the metrics required by docs/experimental-protocol.md."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "faithful_precision": precision[0],
        "faithful_recall": recall[0],
        "faithful_f1": f1[0],
        "faithful_support": int(support[0]),
        "hallucinated_precision": precision[1],
        "hallucinated_recall": recall[1],
        "hallucinated_f1": f1[1],
        "hallucinated_support": int(support[1]),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def summarize_folds(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate numeric fold metrics as mean and standard deviation."""
    numeric = metrics.select_dtypes(include="number").drop(columns=["fold"], errors="ignore")
    return pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(ddof=1)})
