from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .common import load_yaml, materialize_dataset_config


def run_module(module: str, *arguments: object) -> None:
    command = [sys.executable, "-m", module, *(str(value) for value in arguments)]
    subprocess.run(command, check=True)


def absolute_from(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune, train, and validate YOLOv8")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--tune-config", default="configs/tune.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--skip-audit", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_path = Path(args.data).resolve()
    train_path = Path(args.train_config).resolve()
    tune_path = Path(args.tune_config).resolve()
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    run_root = Path("runs/experiments").resolve()
    experiment = run_root / run_id
    experiment.mkdir(parents=True, exist_ok=True)
    Path("runs/LATEST.txt").write_text(f"{experiment}\n", encoding="utf-8")
    materialize_dataset_config(data_path)

    if not args.skip_audit:
        run_module(
            "yolo_factory.audit",
            "--data",
            data_path,
            "--output",
            Path("runs/audit").resolve(),
        )

    tune_config = load_yaml(tune_path)
    tune_data = absolute_from(tune_path, tune_config["data"])
    run_module(
        "yolo_factory.subset",
        "--data",
        data_path,
        "--output",
        tune_data,
    )
    tune_config["data"] = str(tune_data)
    tune_config["project"] = str(experiment / "tune")
    generated_tune_config = experiment / "resolved-tune.yaml"
    generated_tune_config.write_text(
        yaml.safe_dump(tune_config, sort_keys=False), encoding="utf-8"
    )
    run_module("yolo_factory.tune", "--config", generated_tune_config)

    tune_project = experiment / "tune"
    best = load_yaml(tune_project / "best.json")
    train_config = load_yaml(train_path)
    train_config["data"] = str(data_path)
    train_config["project"] = str(experiment)
    train_config["name"] = "train"
    train_config["train"].update(best["parameters"])

    generated_config = experiment / "resolved-train.yaml"
    generated_config.write_text(
        yaml.safe_dump(train_config, sort_keys=False), encoding="utf-8"
    )
    run_module("yolo_factory.train", "--config", generated_config)

    weights = experiment / "train" / "weights" / "best.pt"
    validation_arguments: list[object] = [
        "--weights",
        weights,
        "--data",
        data_path,
        "--output",
        experiment / "validation",
    ]
    selected_classes = train_config["train"].get("classes")
    if selected_classes:
        validation_arguments.extend(["--classes", *selected_classes])
    run_module("yolo_factory.validate", *validation_arguments)
    print(f"Pipeline completed: {experiment}")


if __name__ == "__main__":
    main()
