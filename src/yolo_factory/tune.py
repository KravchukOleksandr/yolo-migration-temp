from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from ultralytics import YOLO

from .common import dump_json, load_yaml, materialize_dataset_config
from .train import model_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune YOLOv8 with Optuna")
    parser.add_argument("--config", default="configs/tune.yaml")
    return parser


def main() -> None:
    try:
        import optuna
    except ImportError as error:
        message = "Install tuning dependencies with pip install -e '.[tune]'"
        raise SystemExit(message) from error

    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    data_path = Path(config["data"])
    if not data_path.is_absolute():
        data_path = config_path.parent / data_path
    data = materialize_dataset_config(data_path)

    def objective(trial: optuna.Trial) -> float:
        parameters = {
            "data": str(data),
            "epochs": config["epochs"],
            "imgsz": config.get("imgsz", 960),
            "batch": config.get("batch", -1),
            "workers": config.get("workers", 8),
            "seed": config.get("seed", 42),
            "optimizer": "AdamW",
            "lr0": trial.suggest_float("lr0", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "mosaic": trial.suggest_float("mosaic", 0.5, 1.0),
            "scale": trial.suggest_float("scale", 0.2, 0.7),
            "translate": trial.suggest_float("translate", 0.0, 0.15),
            "mixup": trial.suggest_float("mixup", 0.0, 0.15),
            "amp": True,
            "cos_lr": True,
            "single_cls": True,
            "project": config.get("project", "runs/tune"),
            "name": f"trial-{trial.number:03d}",
            "plots": False,
        }
        try:
            metrics = YOLO(model_name(config.get("model", "s"))).train(**parameters)
            return float(metrics.results_dict["metrics/mAP50-95(B)"])
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.get("trials", 20))
    dump_json(
        {"value": study.best_value, "parameters": study.best_params},
        f"{config.get('project', 'runs/tune')}/best.json",
    )


if __name__ == "__main__":
    main()
