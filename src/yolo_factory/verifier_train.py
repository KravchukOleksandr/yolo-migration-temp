from __future__ import annotations

import argparse
from pathlib import Path

from .common import dump_json, load_yaml
from .verifier_data import group_holdout_split, read_manifest
from .verifier_training import evaluate_checkpoint, train_verifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the final helmet verifier")
    parser.add_argument("--config", default="configs/verifier.yaml")
    parser.add_argument("--tune-result")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    prepared = Path(config["prepare"]["output"])
    if not prepared.is_absolute():
        prepared = (config_path.parent / prepared).resolve()
    root = Path(config["output"])
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    tune_result = Path(args.tune_result or root / "tune" / "best.json").resolve()
    output = Path(args.output or root / "train").resolve()
    parameters = load_yaml(tune_result)["parameters"]
    train_samples = read_manifest(prepared / "manifest.csv", "train")
    official_validation = read_manifest(prepared / "manifest.csv", "val")
    tuning = config["tune"]
    fit_samples, calibration_samples = group_holdout_split(
        train_samples,
        float(tuning.get("calibration_fraction", 0.10)),
        int(tuning.get("calibration_seed", 2718)),
    )
    result = train_verifier(
        fit_samples,
        calibration_samples,
        parameters,
        output,
        int(config["train"].get("seed", 42)),
        save_best=True,
    )
    threshold = float(result["final_metrics"]["threshold_at_recall_95"])
    validation = evaluate_checkpoint(
        output / "last.pt",
        official_validation,
        parameters,
        threshold,
    )
    dump_json(
        {
            "checkpoint": output / "last.pt",
            "threshold_source": "held-out train calibration split",
            "calibration": result["final_metrics"],
            "official_validation": validation,
        },
        output / "validation.json",
    )
    print(f"Final verifier written to {output}")


if __name__ == "__main__":
    main()
