#!/usr/bin/env python3
"""Generate or verify a CycloneDX-ish ML-BOM for the model pack.

No Syft-class one-button ML-BOM scanner is mainstream for arbitrary packs yet.
This script hashes model/ files and writes inventory/mlbom.json (or --check).
AIBOM stays hand YAML.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PACK_FILES = ("model.joblib", "model_card.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_template(template: Path, model_id: str) -> dict:
    if template.is_file():
        bom = json.loads(template.read_text(encoding="utf-8"))
    else:
        bom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "metadata": {
                "component": {
                    "type": "machine-learning-model",
                    "name": model_id,
                    "version": "1",
                }
            },
            "components": [],
        }
    return bom


def build_bom(model_dir: Path, template: Path) -> dict:
    card_path = model_dir / "model_card.json"
    model_id = "ticket-clf-v1"
    if card_path.is_file():
        try:
            model_id = json.loads(card_path.read_text(encoding="utf-8")).get(
                "model_id", model_id
            )
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid model card: {card_path}: {exc}") from exc

    missing = [name for name in PACK_FILES if not (model_dir / name).is_file()]
    if missing:
        raise SystemExit(f"missing pack files under {model_dir}: {', '.join(missing)}")

    bom = load_template(template, model_id)
    bom.setdefault("metadata", {}).setdefault("component", {})
    bom["metadata"]["component"]["type"] = "machine-learning-model"
    bom["metadata"]["component"]["name"] = model_id
    bom["metadata"]["component"].setdefault("version", "1")

    components = []
    for name in PACK_FILES:
        path = model_dir / name
        components.append(
            {
                "type": "file",
                "name": name,
                "hashes": [{"alg": "SHA-256", "content": sha256_file(path)}],
            }
        )
    bom["components"] = components
    bom["bomFormat"] = "CycloneDX"
    bom.setdefault("specVersion", "1.5")
    return bom


def hash_map_from_bom(bom: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for component in bom.get("components") or []:
        name = component.get("name")
        hashes = component.get("hashes") or []
        if not name or not hashes:
            continue
        content = hashes[0].get("content")
        if content:
            out[str(name)] = str(content)
    return out


def check_bom(model_dir: Path, bom_path: Path) -> int:
    if not bom_path.is_file():
        print(f"FAIL: missing ML-BOM {bom_path}", file=sys.stderr)
        return 1
    try:
        bom = json.loads(bom_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"FAIL: invalid ML-BOM JSON: {exc}", file=sys.stderr)
        return 1

    recorded = hash_map_from_bom(bom)
    failed = False
    for name in PACK_FILES:
        path = model_dir / name
        if not path.is_file():
            print(f"FAIL: missing {path}", file=sys.stderr)
            failed = True
            continue
        actual = sha256_file(path)
        expected = recorded.get(name)
        if expected is None:
            print(f"FAIL: {name} not listed in {bom_path}", file=sys.stderr)
            failed = True
        elif actual != expected:
            print(
                f"FAIL: {name} hash mismatch expected={expected} actual={actual}",
                file=sys.stderr,
            )
            failed = True
        else:
            print(f"ok {name} {actual}")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or verify ML-BOM hashes for the model pack"
    )
    parser.add_argument("--model", default="model", help="Model pack directory")
    parser.add_argument(
        "--out",
        default="inventory/mlbom.json",
        help="Output ML-BOM path (generate mode)",
    )
    parser.add_argument(
        "--template",
        default="inventory/mlbom.template.json",
        help="Optional CycloneDX-ish template",
    )
    parser.add_argument(
        "--check",
        nargs="?",
        const="inventory/mlbom.json",
        default=None,
        metavar="MLBOM",
        help="Fail closed if pack hashes do not match ML-BOM (default inventory/mlbom.json)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model)
    if args.check is not None:
        raise SystemExit(check_bom(model_dir, Path(args.check)))

    bom = build_bom(model_dir, Path(args.template))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    for component in bom["components"]:
        print(f"  {component['name']} {component['hashes'][0]['content']}")


if __name__ == "__main__":
    main()
