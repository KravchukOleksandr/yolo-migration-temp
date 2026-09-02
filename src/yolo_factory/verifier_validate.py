from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from .common import dump_json, load_yaml, materialize_dataset_config
from .verifier_data import preprocess_image
from .verifier_model import HelmetMicroNeXt


class VerifierRuntime:
    def __init__(
        self,
        checkpoint: str | Path,
        device: torch.device,
        batch_size: int,
        scale: float,
        minimum_side: float,
        minimum_ratio: float,
        maximum_ratio: float,
    ) -> None:
        value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = value.get("model", value)
        self.metrics = value.get("metrics", {})
        self.model = HelmetMicroNeXt().to(device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.scale = scale
        self.minimum_side = minimum_side
        self.minimum_ratio = minimum_ratio
        self.maximum_ratio = maximum_ratio

    @torch.inference_mode()
    def score_with_validity(
        self,
        image_path: str | Path,
        boxes: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = np.ones(len(boxes), dtype=np.float32)
        valid = np.zeros(len(boxes), dtype=bool)
        if not len(boxes):
            return scores, valid
        tensors: list[torch.Tensor] = []
        indices: list[int] = []
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_width, image_height = image.size
            for index, values in enumerate(boxes.detach().cpu().tolist()):
                left, top, right, bottom = values
                center_x = (left + right) / 2
                center_y = (top + bottom) / 2
                width = (right - left) * self.scale
                height = (bottom - top) * self.scale
                expanded = (
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                )
                if min(width, height) < self.minimum_side:
                    continue
                if (
                    expanded[0] < 0
                    or expanded[1] < 0
                    or expanded[2] > image_width
                    or expanded[3] > image_height
                ):
                    continue
                rounded_width = math.ceil(expanded[2]) - math.floor(expanded[0])
                rounded_height = math.ceil(expanded[3]) - math.floor(expanded[1])
                ratio = rounded_width / rounded_height
                if ratio < self.minimum_ratio or ratio > self.maximum_ratio:
                    continue
                crop = image.crop(
                    (
                        math.floor(expanded[0]),
                        math.floor(expanded[1]),
                        math.ceil(expanded[2]),
                        math.ceil(expanded[3]),
                    )
                )
                tensors.append(preprocess_image(crop))
                indices.append(index)
                valid[index] = True
        for start in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[start : start + self.batch_size]).to(
                self.device
            )
            probabilities = self.model(batch).sigmoid().cpu().numpy()
            for index, probability in zip(
                indices[start : start + self.batch_size],
                probabilities,
                strict=True,
            ):
                scores[index] = probability
        return scores, valid

    def score(self, image_path: str | Path, boxes: torch.Tensor) -> np.ndarray:
        scores, _ = self.score_with_validity(image_path, boxes)
        return scores


class VerifierDetectionValidator(DetectionValidator):
    def __init__(
        self,
        *args: Any,
        verifier: VerifierRuntime,
        verified_class: int,
        threshold: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.verifier = verifier
        self.verified_class = verified_class
        self.threshold = threshold

    def update_metrics(
        self,
        preds: list[dict[str, torch.Tensor]],
        batch: dict[str, Any],
    ) -> None:
        filtered: list[dict[str, torch.Tensor]] = []
        for index, prediction in enumerate(preds):
            prepared = self._prepare_batch(index, batch)
            native = self.scale_preds(prediction, prepared)
            selected = prediction["cls"] == self.verified_class
            scores = self.verifier.score(
                prepared["im_file"], native["bboxes"][selected]
            )
            keep = torch.ones(len(prediction["cls"]), dtype=torch.bool)
            keep[selected.cpu()] = torch.from_numpy(scores >= self.threshold)
            keep = keep.to(prediction["cls"].device)
            filtered.append(
                {
                    name: value[keep]
                    if isinstance(value, torch.Tensor) and len(value) == len(keep)
                    else value
                    for name, value in prediction.items()
                }
            )
        super().update_metrics(filtered, batch)


def validator_factory(
    verifier: VerifierRuntime,
    verified_class: int,
    threshold: float,
) -> Callable[..., VerifierDetectionValidator]:
    def create(**kwargs: Any) -> VerifierDetectionValidator:
        validator = VerifierDetectionValidator(
            verifier=verifier,
            verified_class=verified_class,
            threshold=threshold,
            **kwargs,
        )
        return validator

    return create


def serializable_metrics(metrics: Any) -> dict[str, Any]:
    return {
        "results": metrics.results_dict,
        "classes": metrics.summary(),
        "speed_ms": metrics.speed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate YOLO with helmet veto")
    parser.add_argument("--config", default="configs/verifier.yaml")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--verifier")
    parser.add_argument("--output")
    parser.add_argument("--threshold", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    root = Path(config["output"])
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    verifier_path = Path(args.verifier or root / "train" / "last.pt").resolve()
    output = Path(args.output or root / "yolo-validation").resolve()
    data_path = Path(config["data"])
    if not data_path.is_absolute():
        data_path = config_path.parent / data_path
    data = materialize_dataset_config(data_path)
    settings = config["cascade"]
    device_name = str(settings.get("device", "cuda:0"))
    device = torch.device(device_name)
    runtime = VerifierRuntime(
        verifier_path,
        device,
        int(settings.get("classifier_batch", 256)),
        float(config["prepare"].get("scale", 1.25)),
        float(config["prepare"].get("minimum_scaled_side", 16.0)),
        float(config["prepare"].get("minimum_aspect_ratio", 0.72)),
        float(config["prepare"].get("maximum_aspect_ratio", 1.39)),
    )
    threshold = args.threshold
    if threshold is None:
        threshold = runtime.metrics.get("threshold_at_recall_95")
    if threshold is None or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(
            "Verifier checkpoint has no valid calibration threshold; "
            "provide --threshold"
        )
    threshold = float(threshold)
    detector = YOLO(str(Path(args.detector).resolve()))
    verified_class = int(settings.get("verified_class", 2))
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
    selected_classes = settings.get("classes", [0, 2])
    if verified_class not in selected_classes:
        raise ValueError("cascade.classes must include cascade.verified_class")
    summary: dict[str, Any] = {}
    for image_size in settings.get("imgsz", [640, 960, 1280]):
        common = {
            "data": str(data),
            "imgsz": int(image_size),
            "batch": int(settings.get("detector_batch", 8)),
            "workers": int(settings.get("workers", 8)),
            "device": device_name,
            "conf": float(settings.get("proposal_confidence", 0.001)),
            "classes": selected_classes,
            "plots": True,
            "exist_ok": True,
            "verbose": True,
        }
        started = time.perf_counter()
        baseline = detector.val(
            project=str(output / str(image_size)),
            name="baseline",
            **common,
        )
        baseline_seconds = time.perf_counter() - started
        started = time.perf_counter()
        cascade = detector.val(
            validator=validator_factory(
                runtime,
                verified_class,
                threshold,
            ),
            project=str(output / str(image_size)),
            name="cascade",
            **common,
        )
        cascade_seconds = time.perf_counter() - started
        summary[str(image_size)] = {
            "verifier_threshold": threshold,
            "threshold_source": "verifier checkpoint calibration split",
            "baseline_elapsed_seconds": baseline_seconds,
            "cascade_elapsed_seconds": cascade_seconds,
            "baseline": serializable_metrics(baseline),
            "cascade": serializable_metrics(cascade),
        }
        dump_json(summary, output / "summary.json")
    print(f"Cascade validation written to {output}")


if __name__ == "__main__":
    main()
