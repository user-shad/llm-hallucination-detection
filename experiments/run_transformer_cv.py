"""Fine-tune one transformer detector with context-grouped cross-validation."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import (  # noqa: E402
    ID_TO_LABEL,
    LABEL_TO_ID,
    fold_metrics,
    grouped_folds,
    load_dataset,
    summarize_folds,
)


CHECKPOINTS = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "deberta": "microsoft/deberta-v3-base",
    "scibert": "allenai/scibert_scivocab_uncased",
}


class EncodedPairs(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = int(self.labels[index])
        return item


def pair_columns(frame: pd.DataFrame, condition: str) -> tuple[list[str], list[str]]:
    if condition == "context_response":
        return frame["context"].tolist(), frame["response"].tolist()
    if condition == "prompt_context_response":
        first = ("question: " + frame["prompt"] + " context: " + frame["context"]).tolist()
        return first, frame["response"].tolist()
    if condition == "prompt_response":
        return frame["prompt"].tolist(), frame["response"].tolist()
    if condition == "response_only":
        return frame["response"].tolist(), [""] * len(frame)
    raise ValueError(f"Unknown condition: {condition}")


def encode(tokenizer, frame: pd.DataFrame, condition: str, max_length: int):
    first, second = pair_columns(frame, condition)
    return tokenizer(
        first,
        second,
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def trainer_metrics(output) -> dict:
    predictions = np.argmax(output.predictions, axis=1)
    metrics = fold_metrics(output.label_ids, predictions)
    return {
        "weighted_f1": metrics["weighted_f1"],
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
    }


def validation_split(frame: pd.DataFrame, train_indices: np.ndarray, seed: int):
    groups = frame.iloc[train_indices]["context"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    inner_train, validation = next(splitter.split(train_indices, groups=groups))
    return train_indices[inner_train], train_indices[validation]


def evaluate(args) -> None:
    set_seed(args.seed)
    frame = load_dataset(args.data)
    checkpoint = args.checkpoint or CHECKPOINTS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.checkpoint_dir or (
        Path(tempfile.gettempdir()) / "hallucination_checkpoints"
    )

    fold_rows = []
    prediction_rows = []

    for fold, (train_validation, test_indices) in enumerate(grouped_folds(frame), start=1):
        train_indices, validation_indices = validation_split(
            frame, np.asarray(train_validation), args.seed
        )
        train = frame.iloc[train_indices]
        validation = frame.iloc[validation_indices]
        test = frame.iloc[test_indices]

        train_data = EncodedPairs(
            encode(tokenizer, train, args.condition, args.max_length),
            train["label_id"].to_numpy(),
        )
        validation_data = EncodedPairs(
            encode(tokenizer, validation, args.condition, args.max_length),
            validation["label_id"].to_numpy(),
        )
        test_data = EncodedPairs(
            encode(tokenizer, test, args.condition, args.max_length),
            test["label_id"].to_numpy(),
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=2,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        )
        # Checkpoints are large and slow to write to mounted Google Drive.
        # Keep them on fast local storage; only final CSV/JSON results go to
        # args.output_dir.
        fold_dir = checkpoint_root / args.model / f"fold_{fold}"
        training_kwargs = dict(
            output_dir=str(fold_dir),
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="weighted_f1",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
        )
        strategy_parameter = (
            "eval_strategy"
            if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
            else "evaluation_strategy"
        )
        training_kwargs[strategy_parameter] = "epoch"
        training_args = TrainingArguments(**training_kwargs)

        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=validation_data,
            data_collator=collator,
            compute_metrics=trainer_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        if "processing_class" in inspect.signature(Trainer.__init__).parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer
        trainer = Trainer(**trainer_kwargs)

        start = time.perf_counter()
        trainer.train()
        training_seconds = time.perf_counter() - start

        start = time.perf_counter()
        output = trainer.predict(test_data)
        inference_seconds = time.perf_counter() - start
        predictions = np.argmax(output.predictions, axis=1)
        metrics = fold_metrics(test["label_id"].to_numpy(), predictions)
        metrics.update(
            {
                "model": args.model,
                "checkpoint": checkpoint,
                "fold": fold,
                "condition": args.condition,
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                "train_examples": len(train),
                "validation_examples": len(validation),
                "test_examples": len(test),
            }
        )
        fold_rows.append(metrics)

        for row_position, prediction, logits in zip(test_indices, predictions, output.predictions):
            row = frame.iloc[row_position]
            probabilities = torch.softmax(torch.tensor(logits), dim=0).numpy()
            prediction_rows.append(
                {
                    "model": args.model,
                    "fold": fold,
                    "id": row.get("id", row_position),
                    "context": row["context"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "true_label": row["label"],
                    "predicted_label": ID_TO_LABEL[int(prediction)],
                    "faithful_probability": float(probabilities[0]),
                    "hallucinated_probability": float(probabilities[1]),
                    "correct": int(prediction) == int(row["label_id"]),
                    "hallucination_type": row.get("hallucination_type", "unknown"),
                    "domain": row.get("domain", "unknown"),
                }
            )

        if not args.keep_checkpoints:
            shutil.rmtree(fold_dir, ignore_errors=True)

    metrics_frame = pd.DataFrame(fold_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    summary = summarize_folds(metrics_frame).reset_index(names="metric")
    metrics_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    predictions_frame.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    config = vars(args).copy()
    config.update(
        {
            "data": str(args.data),
            "output_dir": str(args.output_dir),
            "checkpoint_dir": str(checkpoint_root),
            "resolved_checkpoint": checkpoint,
            "unique_examples": len(frame),
            "context_groups": int(frame["context"].nunique()),
            "folds": 6,
        }
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(summary[summary["metric"].isin(["weighted_f1", "macro_f1", "accuracy"])])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--checkpoint", help="Override the default checkpoint")
    parser.add_argument(
        "--data", type=Path, default=Path("data/llm_hallucination_detection_dataset.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Temporary local checkpoint directory (defaults to the system temp folder)",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep per-fold checkpoints after predictions are saved",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
