#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from modulation_key_estimator.inference import run_inference
from modulation_key_estimator.keys import key_to_index, normalize_key
from modulation_key_estimator.model_loader import load_model


def dominant_region_key(result: dict) -> str:
    regions = result.get("region_infos", [])
    if not regions:
        raise ValueError("No regions returned by inference.")
    region = max(regions, key=lambda item: item["end_time_sec"] - item["start_time_sec"])
    return region["key_name"].lower()


def evaluate(manifest_path: Path, model_path: Path | None = None) -> dict:
    model = load_model(model_path)
    rows = []
    correct = 0

    with manifest_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            audio_path = Path(row["path"])
            expected = normalize_key(row["expected_key"])
            result = run_inference(
                audio_path,
                model,
                target_key_index=key_to_index(expected),
                write_shifted=False,
            )
            predicted = normalize_key(dominant_region_key(result))
            is_correct = predicted == expected
            correct += int(is_correct)
            rows.append({
                "path": str(audio_path),
                "expected_key": expected,
                "predicted_key": predicted,
                "correct": is_correct,
                "regions": result["region_infos"],
                "modulation_points": result["modulation_points"],
            })

    total = len(rows)
    return {
        "manifest": str(manifest_path),
        "total": total,
        "correct": correct,
        "exact_key_accuracy": correct / total if total else 0.0,
        "files": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate key estimation on a labeled CSV manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = evaluate(args.manifest, args.model)
    print(json.dumps(report, ensure_ascii=False, indent=None if args.json else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

