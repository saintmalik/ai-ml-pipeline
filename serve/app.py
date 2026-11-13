#!/usr/bin/env python3
"""Serve the ticket classifier like a normal product API."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

REQUIRE_TOKEN = os.environ.get("REQUIRE_TOKEN", "1") == "1"
API_TOKEN = os.environ.get("API_TOKEN", "lab-token")

# Two planes:
#   GHCR image = API container (this process).
#   Model Registry (HF Hub / MLflow / SageMaker) = model pack revision.
#
# Prod-shaped: MODEL_SOURCE=hub + HF_MODEL_ID + HF_REVISION (pin a registry commit).
# Smoke / CI without Hub: MODEL_SOURCE=local loads the pack baked into the image.
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "local")  # local | hub
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/model"))
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "")
HF_REVISION = os.environ.get("HF_REVISION", "")


def _load_pack() -> tuple[object, dict]:
    if MODEL_SOURCE == "hub":
        if not HF_MODEL_ID or not HF_REVISION:
            raise RuntimeError(
                "Prod-shaped serve: MODEL_SOURCE=hub requires HF_MODEL_ID and HF_REVISION"
            )
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id=HF_MODEL_ID,
            filename="model.joblib",
            revision=HF_REVISION,
        )
        card_path = hf_hub_download(
            repo_id=HF_MODEL_ID,
            filename="model_card.json",
            revision=HF_REVISION,
        )
        pipe = joblib.load(model_path)
        card = json.loads(Path(card_path).read_text(encoding="utf-8"))
        card["registry"] = {
            "kind": "huggingface-hub",
            "model_id": HF_MODEL_ID,
            "revision": HF_REVISION,
            "path": "registry-revision",
        }
        return pipe, card

    # Smoke path only: baked pack from the image filesystem.
    pipe = joblib.load(MODEL_DIR / "model.joblib")
    card = json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))
    card["registry"] = {
        "kind": "local-pack-smoke",
        "path": str(MODEL_DIR),
        "note": "Baked pack for smoke/CI. Prefer MODEL_SOURCE=hub + HF_REVISION in prod.",
    }
    return pipe, card


pipe, card = _load_pack()

app = FastAPI(title="ticket-clf", version="1.0.0")


class PredictIn(BaseModel):
    text: str = Field(min_length=1)


class PredictOut(BaseModel):
    model_id: str
    label: str


def _auth(authorization: str | None) -> None:
    if not REQUIRE_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model_id": card["model_id"],
        "labels": card["labels"],
        "registry": card.get("registry"),
    }


@app.post("/predict", response_model=PredictOut)
def predict(
    body: PredictIn,
    authorization: str | None = Header(default=None),
) -> PredictOut:
    _auth(authorization)
    label = str(pipe.predict([body.text])[0])
    return PredictOut(model_id=card["model_id"], label=label)
