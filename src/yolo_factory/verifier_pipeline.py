from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .common import load_yaml


def run_module(module: str, *arguments: object) -> None:
    command = [sys.executable, "-m", module, *(str(value) for value in arguments)]
    subprocess.run(command, check=True)


def relative_to_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the helmet verifier pipeline")
    parser.add_argument("--config", default="configs/verifier.yaml")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--skip-prepare", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    detector = Path(args.detector).resolve()
    if not detector.is_file():
        raise FileNotFoundError(f"Detector checkpoint does not exist: {detector}")
    config = load_yaml(config_path)
    config["data"] = str(relative_to_config(config_path, config["data"]))
    config["prepare"]["output"] = str(
        relative_to_config(config_path, config["prepare"]["output"])
    )
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "verifier-%Y%m%d-%H%M%S-%f"
    )
    experiment = (Path("runs/verifier") / run_id).resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    config["output"] = str(experiment)
    resolved = experiment / "resolved-config.yaml"
    resolved.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    Path("runs/LATEST_VERIFIER.txt").write_text(
        f"{experiment}\n", encoding="utf-8"
    )
    if not args.skip_prepare:
        run_module("yolo_factory.verifier_prepare", "--config", resolved)
    run_module(
        "yolo_factory.verifier_tune",
        "--config",
        resolved,
        "--output",
        experiment / "tune",
    )
    run_module(
        "yolo_factory.verifier_train",
        "--config",
        resolved,
        "--tune-result",
        experiment / "tune" / "best.json",
        "--output",
        experiment / "train",
    )
    run_module(
        "yolo_factory.verifier_validate",
        "--config",
        resolved,
        "--detector",
        detector,
        "--verifier",
        experiment / "train" / "last.pt",
        "--output",
        experiment / "yolo-validation",
    )
    print(f"Verifier pipeline completed: {experiment}")


if __name__ == "__main__":
    main()
