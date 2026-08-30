from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .common import load_yaml, materialize_dataset_config

LEGACY_SEARCH_PARAMETERS = {
    "lr0",
    "weight_decay",
    "mosaic",
    "scale",
    "translate",
    "mixup",
}


def run_module(module: str, *arguments: object) -> None:
    command = [sys.executable, "-m", module, *(str(value) for value in arguments)]
    subprocess.run(command, check=True)


def absolute_from(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def selected_trial_parameters(
    selection: str | Path,
    tune_config: dict,
    train_config: dict,
) -> tuple[Path, dict[str, object]]:
    source = Path(selection).expanduser().resolve()
    args_path = source / "args.yaml" if source.is_dir() else source
    if not args_path.is_file():
        raise FileNotFoundError(f"Selected trial args.yaml not found: {args_path}")
    trial_args = load_yaml(args_path)

    train_settings = train_config.get("train", {})
    expected = {
        "classes": train_settings.get("classes"),
        "single_cls": train_settings.get("single_cls", False),
    }
    mismatches = [
        name
        for name, value in expected.items()
        if name in trial_args and trial_args[name] != value
    ]
    if mismatches:
        names = ", ".join(mismatches)
        raise ValueError(f"Selected trial disagrees with training on: {names}")

    family = train_config.get("family", "yolov8")
    size = train_config.get("model")
    expected_weights = f"{family}{size}.pt"
    trial_model = trial_args.get("model")
    if trial_model and Path(str(trial_model)).name != expected_weights:
        raise ValueError(
            f"Selected trial used {Path(str(trial_model)).name}, "
            f"expected {expected_weights}"
        )

    search_space = tune_config.get("search_space")
    search_names = set(search_space or LEGACY_SEARCH_PARAMETERS)
    fixed_names = set(tune_config.get("fixed", {}))
    promoted_names = search_names | fixed_names
    missing = sorted(promoted_names - trial_args.keys())
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Selected trial is missing parameters: {names}")
    parameters = {name: trial_args[name] for name in sorted(promoted_names)}
    return args_path, parameters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune, train, and validate YOLO")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--tune-config", default="configs/tune.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument(
        "--use-trial",
        help="Completed tuning trial directory or its args.yaml",
    )
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
    train_config = load_yaml(train_path)
    train_classes = train_config.get("train", {}).get("classes")
    train_single_class = train_config.get("train", {}).get(
        "single_cls", False
    )
    train_family = train_config.get("family", "yolov8")
    tune_family = tune_config.get("family", "yolov8")
    compatibility = {
        "family": (train_family, tune_family),
        "model": (train_config.get("model"), tune_config.get("model")),
        "classes": (train_classes, tune_config.get("classes")),
        "single_cls": (train_single_class, tune_config.get("single_cls", False)),
    }
    mismatches = [
        name for name, values in compatibility.items() if values[0] != values[1]
    ]
    if mismatches:
        names = ", ".join(mismatches)
        raise ValueError(f"Train and tune configurations disagree on: {names}")
    objective_class = tune_config.get("objective_class")
    if objective_class is not None and (
        not train_classes or objective_class not in train_classes
    ):
        raise ValueError("The tuning objective class must be selected for training")
    if args.use_trial:
        trial_args, selected_parameters = selected_trial_parameters(
            args.use_trial,
            tune_config,
            train_config,
        )
        selection = {
            "source": str(trial_args),
            "parameters": selected_parameters,
        }
        (experiment / "selected-trial.yaml").write_text(
            yaml.safe_dump(selection, sort_keys=False), encoding="utf-8"
        )
        print(f"Using selected tuning trial: {trial_args}", flush=True)
    else:
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
        best = load_yaml(experiment / "tune" / "best.json")
        selected_parameters = best["parameters"]

    train_config["data"] = str(data_path)
    train_config["project"] = str(experiment)
    train_config["name"] = "train"
    train_config["train"].update(selected_parameters)

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
