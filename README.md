# ai-ml-pipeline

CPU-friendly lab: **pipeline security gates** for an AI/ML product path (dataset pin → train/eval → SBOM/ML-BOM/AIBOM → Model Registry promote → attested API image → production admit). AI Sec owns the gates; platform owns where inference runs.

## Two artifacts

| Artifact | What | Where |
| --- | --- | --- |
| **Model pack** | Weights (`model.joblib`) + card | Actions artifact; **HF Hub** when `HF_TOKEN` is set |
| **API image** | FastAPI app that loads the model | **GHCR** (`ghcr.io/saintmalik/ai-ml-pipeline`) |

Hub holds the model revision. GHCR holds the product container. Smoke serve bakes a local pack (`MODEL_SOURCE=local`); prod-shaped serve pins Hub (`MODEL_SOURCE=hub` + `HF_MODEL_ID` + `HF_REVISION`).

## Pipeline

Push to `main` → **gates** → **train** → Hub promote (if secret) → GHCR image + attest → **production** admit (required reviewers).

Workflow: `.github/workflows/train-serve.yml`

## Secrets / vars

| Name | Kind | Required? | Purpose |
| --- | --- | --- | --- |
| `HF_TOKEN` | **repo secret** | Hub promote | HF write token (`env: HF_TOKEN: ${{ secrets.HF_TOKEN }}`) |
| `GITLEAKS_LICENSE` | secret | Optional | Gitleaks on public repos is free without it |
| `API_TOKEN` | runtime env | Runtime | Bearer for `/predict` |
| `HF_MODEL_REPO` | repo variable | Optional | Hub id (default `saintmalik/ticket-clf`) |

Without `HF_TOKEN`, train + GHCR still succeed; only Hub promote is skipped.

```bash
printf '%s' "$HF_TOKEN" | gh secret set HF_TOKEN --repo saintmalik/ai-ml-pipeline
```

## Optional local smoke

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python train/train.py --data data/tickets.csv --out model
```

```bash
.venv/bin/python train/eval_canary.py --model model --report out/eval.json
```

```bash
docker build -f serve/Dockerfile -t ticket-clf:local .
```

```bash
docker run --rm -p 18080:8080 -e API_TOKEN=lab-token -e MODEL_SOURCE=local ticket-clf:local
```

```bash
curl -sS http://127.0.0.1:18080/health
```

## Companion post

[From securing conventional product CI/CD to securing AI/ML product pipelines](https://blog.saintmalik.me/from-securing-cicd-to-securing-ai-ml-pipelines/)
