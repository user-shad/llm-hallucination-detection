"""Summarize out-of-fold errors by model, hallucination type, and domain."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def rate_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    table = (
        frame.groupby(group_columns, dropna=False)["correct"]
        .agg(examples="size", correct="sum", accuracy="mean")
        .reset_index()
    )
    table["errors"] = table["examples"] - table["correct"]
    return table.sort_values(["accuracy", "examples"], ascending=[True, False])


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def analyze(prediction_path: Path, output_dir: Path) -> None:
    predictions = pd.read_csv(prediction_path)
    required = {
        "model",
        "true_label",
        "predicted_label",
        "correct",
        "hallucination_type",
        "domain",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions["correct"] = predictions["correct"].astype(bool)

    overall = rate_table(predictions, ["model"])
    by_type = rate_table(predictions, ["model", "hallucination_type"])
    by_domain = rate_table(predictions, ["model", "domain"])
    confusion = (
        predictions.groupby(["model", "true_label", "predicted_label"])
        .size()
        .reset_index(name="examples")
    )
    errors = predictions.loc[~predictions["correct"]].copy()

    overall.to_csv(output_dir / "accuracy_by_model.csv", index=False)
    by_type.to_csv(output_dir / "accuracy_by_hallucination_type.csv", index=False)
    by_domain.to_csv(output_dir / "accuracy_by_domain.csv", index=False)
    confusion.to_csv(output_dir / "confusion_counts.csv", index=False)
    errors.to_csv(output_dir / "incorrect_predictions.csv", index=False)

    report = [
        "# Error Analysis",
        "",
        "Generated from out-of-fold predictions. Metadata is used only after prediction, not as model input.",
        "",
        "## Overall accuracy",
        "",
        markdown_table(overall),
        "",
        "## Hardest hallucination-type groups",
        "",
        markdown_table(by_type.head(20)),
        "",
        "## Hardest domain groups",
        "",
        markdown_table(by_domain.head(20)),
        "",
        "## Manual review checklist",
        "",
        "For each important model, inspect incorrect_predictions.csv and record:",
        "",
        "- numeric, entity, date, or location substitutions;",
        "- negation and direct contradiction;",
        "- unsupported additions and overgeneralization;",
        "- faithful answers incorrectly rejected;",
        "- ambiguous or questionable dataset labels;",
        "- errors shared by all models and errors unique to one model.",
        "",
    ]
    (output_dir / "error_analysis.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/error_analysis"))
    args = parser.parse_args()
    analyze(args.predictions, args.output_dir)


if __name__ == "__main__":
    main()
