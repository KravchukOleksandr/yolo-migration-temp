from __future__ import annotations

import argparse
from pathlib import Path

from .common import dump_json, load_yaml, materialize_dataset_config, run_metadata
from .offline import offline_runtime

MODEL_FAMILIES = {"yolov8", "yolo26"}
MODEL_SIZES = {"n", "s", "m", "l", "x"}


def local_checkpoint(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Local checkpoint does not exist: {path}")
    return path


def model_path(
    size: str,
    weights_dir: str | Path = "weights",
    family: str = "yolov8",
) -> Path:
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Model family must be one of {sorted(MODEL_FAMILIES)}")
    if size not in MODEL_SIZES:
        raise ValueError(f"Model size must be one of {sorted(MODEL_SIZES)}")
    filename = f"{family}{size}.pt"
    path = Path(weights_dir) / filename
    if not path.is_absolute():
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing local weights: {path}. Place {filename} there first."
        )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an Ultralytics detector")
    parser.add_argument("--config", default="configs/train_yolo26.yaml")
    parser.add_argument("--model", choices=sorted(MODEL_SIZES))
    parser.add_argument("--name")
    parser.add_argument("--resume", help="Path to a last.pt checkpoint")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    size = args.model or config["model"]
    family = config.get("family", "yolov8")
    name = args.name or config.get("name") or f"{family}{size}"
    project = Path(config.get("project", "runs/train")).resolve()
    parameters = dict(config.get("train", {}))
    data_path = Path(config["data"])
    if not data_path.is_absolute():
        data_path = config_path.parent / data_path
    data = materialize_dataset_config(data_path)
    parameters.update(data=str(data), project=str(project), name=name)
    if args.resume:
        weights = local_checkpoint(args.resume)
        parameters["resume"] = True
    else:
        weights = model_path(size, config.get("weights_dir", "weights"), family)
    with offline_runtime() as model_type:
        model = model_type(str(weights))
        model.train(**parameters)
        run_dir = Path(model.trainer.save_dir)
    dump_json(
        {"model": weights, "parameters": parameters, **run_metadata()},
        run_dir / "run.json",
    )


if __name__ == "__main__":
    main()
