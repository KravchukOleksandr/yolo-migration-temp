from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from .common import (
    dump_json,
    load_yaml,
    materialize_dataset_config,
    resolve_dataset_paths,
)
from .dataset import image_files
from .verifier_data import is_holdout
from .verifier_metrics import proposal_gate_metrics, threshold_for_tp_retention
from .verifier_validate import (
    VerifierRuntime,
    serializable_metrics,
    validator_factory,
)


class ProposalCollectorValidator(DetectionValidator):
    def __init__(
        self,
        *args: Any,
        verifier: VerifierRuntime,
        verified_class: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.verifier = verifier
        self.verified_class = verified_class
        self.proposal_scores: list[np.ndarray] = []
        self.proposal_confidences: list[np.ndarray] = []
        self.proposal_true_positive: list[np.ndarray] = []
        self.proposal_valid: list[np.ndarray] = []

    def update_metrics(
        self,
        preds: list[dict[str, torch.Tensor]],
        batch: dict[str, Any],
    ) -> None:
        for index, prediction in enumerate(preds):
            prepared = self._prepare_batch(index, batch)
            native = self.scale_preds(prediction, prepared)
            selected = prediction["cls"] == self.verified_class
            scores, valid = self.verifier.score_with_validity(
                prepared["im_file"], native["bboxes"][selected]
            )
            true_positive = self._process_batch(prediction, prepared)["tp"]
            selected_cpu = selected.cpu().numpy()
            self.proposal_scores.append(scores)
            self.proposal_valid.append(valid)
            self.proposal_confidences.append(
                prediction["conf"][selected].detach().cpu().numpy()
            )
            self.proposal_true_positive.append(true_positive[selected_cpu])
        super().update_metrics(preds, batch)

    def arrays(self) -> dict[str, np.ndarray]:
        columns = len(self.iouv)
        return {
            "scores": np.concatenate(self.proposal_scores)
            if self.proposal_scores
            else np.zeros(0, dtype=np.float32),
            "confidences": np.concatenate(self.proposal_confidences)
            if self.proposal_confidences
            else np.zeros(0, dtype=np.float32),
            "true_positive": np.concatenate(self.proposal_true_positive)
            if self.proposal_true_positive
            else np.zeros((0, columns), dtype=bool),
            "valid": np.concatenate(self.proposal_valid)
            if self.proposal_valid
            else np.zeros(0, dtype=bool),
        }


def collector_factory(
    holder: dict[str, ProposalCollectorValidator],
    verifier: VerifierRuntime,
    verified_class: int,
) -> Callable[..., ProposalCollectorValidator]:
    def create(**kwargs: Any) -> ProposalCollectorValidator:
        validator = ProposalCollectorValidator(
            verifier=verifier,
            verified_class=verified_class,
            **kwargs,
        )
        holder["validator"] = validator
        return validator

    return create


def calibration_data(
    source: Path,
    output: Path,
    fraction: float,
    seed: int,
) -> tuple[Path, int]:
    root, config = resolve_dataset_paths(source)
    train_root = root / config["train"]
    selected = [
        path.resolve()
        for path in image_files(train_root)
        if is_holdout(path.relative_to(train_root).as_posix(), fraction, seed)
    ]
    if not selected:
        raise ValueError("Calibration image split is empty")
    image_list = output / "calibration-images.txt"
    image_list.write_text(
        "".join(f"{path}\n" for path in selected), encoding="utf-8"
    )
    calibration = dict(config)
    calibration["path"] = str(root)
    calibration["val"] = str(image_list.resolve())
    target = output / "calibration-data.yaml"
    target.write_text(yaml.safe_dump(calibration, sort_keys=False), encoding="utf-8")
    return target, len(selected)


def metric_deltas(first: Any, second: Any) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for name, value in first.results_dict.items():
        other = second.results_dict.get(name)
        if isinstance(value, (int, float)) and isinstance(other, (int, float)):
            deltas[name] = float(other - value)
    return deltas


def write_sweep(
    path: Path,
    scores: np.ndarray,
    true_positive: np.ndarray,
    retentions: list[float],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for retention in retentions:
        threshold = threshold_for_tp_retention(scores, true_positive, retention)
        rows.append(
            {
                "requested_tp_retention": retention,
                **proposal_gate_metrics(scores, true_positive, threshold),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep verifier on YOLO proposals")
    parser.add_argument("--config", default="configs/verifier.yaml")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--retention", type=float)
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    settings = config["cascade"]
    sweep = config.get("sweep", {})
    output = Path(
        args.output
        or Path("runs/verifier-sweeps")
        / datetime.now(timezone.utc).strftime("sweep-%Y%m%d-%H%M%S-%f")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    data_path = Path(config["data"])
    if not data_path.is_absolute():
        data_path = config_path.parent / data_path
    official_data = materialize_dataset_config(data_path)
    tuning = config["tune"]
    calibration, calibration_images = calibration_data(
        data_path,
        output,
        float(tuning.get("calibration_fraction", 0.10)),
        int(tuning.get("calibration_seed", 2718)),
    )
    device_name = str(settings.get("device", "cuda:0"))
    runtime = VerifierRuntime(
        Path(args.verifier).resolve(),
        torch.device(device_name),
        int(settings.get("classifier_batch", 256)),
        float(config["prepare"].get("scale", 1.25)),
        float(config["prepare"].get("minimum_scaled_side", 16.0)),
        float(config["prepare"].get("minimum_aspect_ratio", 0.72)),
        float(config["prepare"].get("maximum_aspect_ratio", 1.39)),
    )
    detector = YOLO(str(Path(args.detector).resolve()))
    verified_class = int(settings.get("verified_class", 2))
    selected_classes = settings.get("classes", [0, 2])
    target_retention = float(
        args.retention
        if args.retention is not None
        else sweep.get("target_tp_retention", 0.995)
    )
    if not 0.0 < target_retention <= 1.0:
        raise ValueError("TP retention must be in (0, 1]")
    names = detector.names
    if isinstance(names, dict):
        available_classes = set(names)
    else:
        available_classes = set(range(len(names)))
    if verified_class not in available_classes:
        raise ValueError(
            f"Detector has classes {sorted(available_classes)}, "
            f"but verifier requires source class {verified_class}"
        )
    if verified_class not in selected_classes:
        raise ValueError("cascade.classes must include cascade.verified_class")
    requested_retentions = sorted(
        {
            float(value)
            for value in sweep.get(
                "retentions", [0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
            )
        }
        | {target_retention}
    )
    summary: dict[str, Any] = {
        "detector": str(Path(args.detector).resolve()),
        "verifier": str(Path(args.verifier).resolve()),
        "calibration_images": calibration_images,
        "target_tp_retention": target_retention,
        "resolutions": {},
    }
    for image_size in settings.get("imgsz", [640, 960, 1280]):
        size_output = output / str(image_size)
        common = {
            "imgsz": int(image_size),
            "batch": int(settings.get("detector_batch", 8)),
            "workers": int(settings.get("workers", 8)),
            "device": device_name,
            "conf": float(settings.get("proposal_confidence", 0.001)),
            "classes": selected_classes,
            "exist_ok": True,
            "verbose": True,
        }
        holder: dict[str, ProposalCollectorValidator] = {}
        calibration_baseline = detector.val(
            validator=collector_factory(holder, runtime, verified_class),
            data=str(calibration),
            project=str(size_output),
            name="calibration-baseline",
            plots=False,
            **common,
        )
        arrays = holder["validator"].arrays()
        if not len(arrays["scores"]):
            raise ValueError(f"No class {verified_class} proposals at {image_size}")
        np.savez_compressed(size_output / "calibration-proposals.npz", **arrays)
        rows = write_sweep(
            size_output / "threshold-sweep.csv",
            arrays["scores"],
            arrays["true_positive"],
            requested_retentions,
        )
        selected_threshold = threshold_for_tp_retention(
            arrays["scores"], arrays["true_positive"], target_retention
        )
        zero = detector.val(
            validator=validator_factory(runtime, verified_class, 0.0),
            data=str(calibration),
            project=str(size_output),
            name="calibration-threshold-zero",
            plots=False,
            **common,
        )
        sanity_deltas = metric_deltas(calibration_baseline, zero)
        maximum_delta = max(
            (abs(value) for value in sanity_deltas.values()), default=0.0
        )
        tolerance = float(sweep.get("sanity_tolerance", 1e-9))
        if maximum_delta > tolerance:
            raise RuntimeError(
                f"Threshold-zero sanity check failed at {image_size}: "
                f"maximum metric delta is {maximum_delta}"
            )
        started = time.perf_counter()
        baseline = detector.val(
            data=str(official_data),
            project=str(size_output),
            name="baseline",
            plots=True,
            **common,
        )
        baseline_seconds = time.perf_counter() - started
        started = time.perf_counter()
        cascade = detector.val(
            validator=validator_factory(
                runtime, verified_class, selected_threshold
            ),
            data=str(official_data),
            project=str(size_output),
            name="cascade",
            plots=True,
            **common,
        )
        cascade_seconds = time.perf_counter() - started
        selected_gate = proposal_gate_metrics(
            arrays["scores"], arrays["true_positive"], selected_threshold
        )
        summary["resolutions"][str(image_size)] = {
            "selected_threshold": selected_threshold,
            "selected_calibration_gate": selected_gate,
            "calibration_sweep": rows,
            "calibration_baseline": serializable_metrics(calibration_baseline),
            "invalid_calibration_proposals": int((~arrays["valid"]).sum()),
            "threshold_zero_maximum_delta": maximum_delta,
            "baseline_elapsed_seconds": baseline_seconds,
            "cascade_elapsed_seconds": cascade_seconds,
            "metric_deltas": metric_deltas(baseline, cascade),
            "baseline": serializable_metrics(baseline),
            "cascade": serializable_metrics(cascade),
        }
        dump_json(summary, output / "summary.json")
        print(
            f"{image_size}: threshold={selected_threshold:.6f}, "
            f"minimum calibration TP retention="
            f"{100 * selected_gate['minimum_tp_retention']:.3f}%",
            flush=True,
        )
    print(f"Verifier proposal sweep written to {output}")


if __name__ == "__main__":
    main()
