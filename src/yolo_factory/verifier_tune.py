from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import optuna

from .common import dump_json, load_yaml
from .verifier_data import group_holdout_split, read_manifest, tuning_split
from .verifier_training import metric_rank, train_verifier


def suggest(trial: optuna.Trial, search_space: dict[str, dict]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, specification in search_space.items():
        kind = specification.get("type", "float")
        if kind == "float":
            values[name] = trial.suggest_float(
                name,
                float(specification["low"]),
                float(specification["high"]),
                log=bool(specification.get("log", False)),
                step=specification.get("step"),
            )
        elif kind == "int":
            values[name] = trial.suggest_int(
                name,
                int(specification["low"]),
                int(specification["high"]),
                step=int(specification.get("step", 1)),
            )
        elif kind == "categorical":
            values[name] = trial.suggest_categorical(
                name, specification["choices"]
            )
        else:
            raise ValueError(f"Unsupported search type for {name}: {kind}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune the helmet verifier")
    parser.add_argument("--config", default="configs/verifier.yaml")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    prepared = Path(config["prepare"]["output"])
    if not prepared.is_absolute():
        prepared = (config_path.parent / prepared).resolve()
    output = Path(args.output or config["output"])
    if not output.is_absolute():
        output = (config_path.parent / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tuning = config["tune"]
    samples = read_manifest(prepared / "manifest.csv", "train")
    tuning_pool, calibration = group_holdout_split(
        samples,
        float(tuning.get("calibration_fraction", 0.10)),
        int(tuning.get("calibration_seed", 2718)),
    )
    fit, validation = tuning_split(
        tuning_pool,
        float(tuning.get("validation_fraction", 0.20)),
        int(tuning.get("split_seed", 1729)),
    )
    fixed = dict(config["train"])
    search_space = tuning["search_space"]

    def objective(trial: optuna.Trial) -> float:
        parameters = {**fixed, **suggest(trial, search_space)}
        run = train_verifier(
            fit,
            validation,
            parameters,
            output / "trials" / f"trial-{trial.number:04d}",
            int(tuning.get("seed", 42)),
            save_best=False,
        )
        metrics = run["final_metrics"]
        for name in (
            "achieved_recall_at_95",
            "roc_auc",
            "pr_auc",
            "threshold_at_recall_95",
        ):
            trial.set_user_attr(name, float(metrics[name]))
        return float(metrics["fpr_at_recall_95"])

    storage = f"sqlite:///{(output / 'study.db').as_posix()}"
    study = optuna.create_study(
        study_name="helmet-verifier",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=int(tuning.get("seed", 42)),
            n_startup_trials=int(tuning.get("startup_trials", 40)),
        ),
    )
    target_trials = int(tuning.get("trials", 200))
    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    if completed < target_trials:
        study.optimize(objective, n_trials=target_trials - completed)
    candidates = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not candidates:
        raise RuntimeError("No verifier tuning trial completed")

    def rank(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
        metrics = {
            "fpr_at_recall_95": float(trial.value),
            **trial.user_attrs,
        }
        return metric_rank(metrics)

    best = min(candidates, key=rank)
    parameter_names = sorted(search_space)
    metric_names = [
        "achieved_recall_at_95",
        "roc_auc",
        "pr_auc",
        "threshold_at_recall_95",
    ]
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "trial",
            "state",
            "fpr_at_recall_95",
            *(f"parameter_{name}" for name in parameter_names),
            *metric_names,
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trial in study.trials:
            writer.writerow(
                {
                    "trial": trial.number,
                    "state": trial.state.name,
                    "fpr_at_recall_95": trial.value,
                    **{
                        f"parameter_{name}": trial.params.get(name)
                        for name in parameter_names
                    },
                    **{
                        name: trial.user_attrs.get(name) for name in metric_names
                    },
                }
            )
    payload = {
        "trial": best.number,
        "objective": best.value,
        "parameters": {**fixed, **best.params},
        "metrics": best.user_attrs,
        "completed_trials": len(candidates),
        "fit_samples": len(fit),
        "tune_samples": len(validation),
        "calibration_samples": len(calibration),
    }
    dump_json(payload, output / "best.json")
    print(f"Best verifier trial: {best.number}; FPR@R95={100 * best.value:.3f}%")


if __name__ == "__main__":
    main()
