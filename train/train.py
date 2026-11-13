#!/usr/bin/env python3
"""Train a tiny ticket classifier and write a model pack."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with text,label columns")
    parser.add_argument("--out", required=True, help="Output model directory")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts: list[str] = []
    labels: list[str] = []
    with data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"text", "label"}.issubset(reader.fieldnames):
            raise SystemExit("CSV must have text and label columns")
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    score = float(pipe.score(x_test, y_test))

    model_path = out_dir / "model.joblib"
    joblib.dump(pipe, model_path)

    meta = {
        "model_id": "ticket-clf-v1",
        "framework": "scikit-learn",
        "task": "text-classification",
        "labels": sorted(set(labels)),
        "holdout_accuracy": score,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "dataset_file": data_path.name,
    }
    (out_dir / "model_card.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {model_path}")
    print(f"holdout_accuracy={score:.4f}")


if __name__ == "__main__":
    main()
