from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from .common import dump_json, load_yaml, materialize_dataset_config, run_metadata

MODEL_SIZES = {"n", "s", "m"}


def model_name(size: str) -> str:
    if size not in MODEL_SIZES:
        raise ValueError(f"Model size must be one of {sorted(MODEL_SIZES)}")
    return f"yolov8{size}.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 detector")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", choices=sorted(MODEL_SIZES))
    parser.add_argument("--name")
    parser.add_argument("--resume", help="Path to a last.pt checkpoint")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    size = args.model or config["model"]
    name = args.name or config.get("name") or f"yolov8{size}"
    project = Path(config.get("project", "runs/train"))
    parameters = dict(config.get("train", {}))
    data_path = Path(config["data"])
    if not data_path.is_absolute():
        data_path = config_path.parent / data_path
    data = materialize_dataset_config(data_path)
    parameters.update(data=str(data), project=str(project), name=name)
    weights = model_name(size)
    if args.resume:
        weights = args.resume
        parameters["resume"] = True
    model = YOLO(weights)
    model.train(**parameters)
    run_dir = Path(model.trainer.save_dir)
    dump_json(
        {"model": weights, "parameters": parameters, **run_metadata()},
        run_dir / "run.json",
    )


if __name__ == "__main__":
    main()
