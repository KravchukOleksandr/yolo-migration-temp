from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from .common import (
    dump_json,
    load_yaml,
    materialize_dataset_config,
    promotable_train_parameters,
)
from .offline import offline_runtime
from .train import model_path

DEFAULT_FIXED = {
    "optimizer": "AdamW",
    "amp": True,
    "cos_lr": True,
}

DEFAULT_SEARCH_SPACE = {
    "lr0": {"type": "float", "low": 1e-4, "high": 5e-3, "log": True},
    "weight_decay": {
        "type": "float",
        "low": 1e-5,
        "high": 1e-3,
        "log": True,
    },
    "mosaic": {"type": "float", "low": 0.5, "high": 1.0},
    "scale": {"type": "float", "low": 0.2, "high": 0.7},
    "translate": {"type": "float", "low": 0.0, "high": 0.15},
    "mixup": {"type": "float", "low": 0.0, "high": 0.15},
}


def suggest_parameters(trial: object, search_space: dict) -> dict[str, object]:
    parameters: dict[str, object] = {}
    for name, specification in search_space.items():
        kind = specification.get("type", "float")
        if kind == "float":
            parameters[name] = trial.suggest_float(
                name,
                float(specification["low"]),
                float(specification["high"]),
                log=bool(specification.get("log", False)),
                step=specification.get("step"),
            )
        elif kind == "int":
            parameters[name] = trial.suggest_int(
                name,
                int(specification["low"]),
                int(specification["high"]),
                step=int(specification.get("step", 1)),
                log=bool(specification.get("log", False)),
            )
        elif kind == "categorical":
            parameters[name] = trial.suggest_categorical(name, specification["choices"])
        else:
            raise ValueError(f"Unsupported search-space type for {name}: {kind}")
    return parameters


def objective_value(metrics: object, class_id: int | None) -> tuple[float, float]:
    aggregate = float(metrics.results_dict["metrics/mAP50-95(B)"])
    if class_id is None:
        return aggregate, aggregate
    maps = metrics.box.maps
    if (
        class_id < 0
        or class_id >= len(maps)
        or class_id not in metrics.box.ap_class_index
    ):
        message = f"Objective class {class_id} is absent from validation metrics"
        raise ValueError(message)
    return float(maps[class_id]), aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune an Ultralytics model")
    parser.add_argument("--config", default="configs/tune_yolo26.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import optuna
    except ImportError as error:
        message = "Install tuning dependencies with pip install -e '.[tune]'"
        raise SystemExit(message) from error

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    data_path = Path(config["data"])
    if not data_path.is_absolute():
        data_path = config_path.parent / data_path
    data = materialize_dataset_config(data_path)
    project = Path(config.get("project", "runs/tune")).resolve()
    fixed = {**DEFAULT_FIXED, **config.get("fixed", {})}
    search_space = config.get("search_space", DEFAULT_SEARCH_SPACE)
    objective_class = config.get("objective_class")

    def objective(trial: optuna.Trial) -> float:
        parameters = {
            "data": str(data),
            "epochs": config["epochs"],
            "imgsz": config.get("imgsz", 960),
            "batch": config.get("batch", 16),
            "workers": config.get("workers", 8),
            "seed": config.get("seed", 42),
            "single_cls": config.get("single_cls", False),
            "classes": config.get("classes"),
            "project": str(project),
            "name": f"trial-{trial.number:03d}",
            "plots": False,
            **fixed,
            **suggest_parameters(trial, search_space),
        }
        try:
            weights = model_path(
                config.get("model", "s"),
                config.get("weights_dir", "weights"),
                config.get("family", "yolov8"),
            )
            with offline_runtime() as model_type:
                metrics = model_type(str(weights)).train(**parameters)
            score, aggregate = objective_value(metrics, objective_class)
            trial.set_user_attr("aggregate_map50_95", aggregate)
            trial.set_user_attr("objective_map50_95", score)
            print(
                f"Trial {trial.number}: objective mAP50-95={score:.5f}, "
                f"all mAP50-95={aggregate:.5f}",
                flush=True,
            )
            return score
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    sampler = optuna.samplers.TPESampler(
        seed=config.get("seed", 42),
        n_startup_trials=config.get("n_startup_trials", 3),
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=config.get("trials", 20))
    dump_json(
        {
            "value": study.best_value,
            "objective_class": objective_class,
            "parameters": promotable_train_parameters(fixed, study.best_params),
            "metrics": study.best_trial.user_attrs,
        },
        project / "best.json",
    )


if __name__ == "__main__":
    main()
