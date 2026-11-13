#!/usr/bin/env python3
"""Fail closed when canary prompts miss expected labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


CANARIES = [
    {"text": "Cannot login after password reset", "expect": "auth"},
    {"text": "I was charged twice on my invoice", "expect": "billing"},
    {"text": "The export button crashes the app", "expect": "bug"},
    {"text": "How do I invite a teammate", "expect": "howto"},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model pack directory")
    parser.add_argument("--report", required=True, help="Eval JSON output path")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=1.0,
        help="Minimum canary accuracy required to pass",
    )
    args = parser.parse_args()

    model_dir = Path(args.model)
    pipe = joblib.load(model_dir / "model.joblib")
    card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))

    results = []
    hits = 0
    for row in CANARIES:
        pred = str(pipe.predict([row["text"]])[0])
        ok = pred == row["expect"]
        hits += int(ok)
        results.append(
            {
                "text": row["text"],
                "expect": row["expect"],
                "pred": pred,
                "ok": ok,
            }
        )

    accuracy = hits / len(CANARIES)
    canary_pass = accuracy >= args.min_accuracy
    report = {
        "model_id": card.get("model_id"),
        "canary_pass": canary_pass,
        "canary_accuracy": accuracy,
        "min_accuracy": args.min_accuracy,
        "results": results,
        "holdout_accuracy": card.get("holdout_accuracy"),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not canary_pass:
        raise SystemExit("canary gate failed")


if __name__ == "__main__":
    main()
