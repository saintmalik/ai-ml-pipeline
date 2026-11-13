#!/usr/bin/env python3
"""Promote a model pack to Hugging Face Hub (startup-shaped Model Registry)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "saintmalik/ticket-clf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local model pack directory")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_MODEL_REPO") or DEFAULT_REPO_ID,
        help=f"HF model repo (default: $HF_MODEL_REPO or {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--dataset-sha256",
        default="",
        help="Training dataset digest to attach as card metadata",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is required to register")

    model_dir = Path(args.model)
    joblib_path = model_dir / "model.joblib"
    card_path = model_dir / "model_card.json"
    if not joblib_path.is_file() or not card_path.is_file():
        raise SystemExit(f"missing pack files under {model_dir}")

    card = json.loads(card_path.read_text(encoding="utf-8"))
    if args.dataset_sha256:
        card["dataset_sha256"] = args.dataset_sha256
        card_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

    # Use env HF_TOKEN only — no login() (that stores a second token and prints a confusing notice).
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=False)

    commit = api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"promote {card.get('model_id', 'model')}",
    )
    revision = getattr(commit, "oid", None) or str(commit)
    print(f"registered repo_id={args.repo_id} revision={revision}")
    print(f"::notice::HF model registry promote: {args.repo_id}@{revision}")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as handle:
            handle.write(f"repo_id={args.repo_id}\n")
            handle.write(f"revision={revision}\n")


if __name__ == "__main__":
    main()
