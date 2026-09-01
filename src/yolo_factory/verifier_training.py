from __future__ import annotations

import copy
import csv
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .common import dump_json
from .verifier_data import (
    VerifierDataset,
    VerifierSample,
    balanced_sampler,
)
from .verifier_metrics import binary_metrics
from .verifier_model import HelmetMicroNeXt, trainable_parameters


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            current = source[name].detach()
            if torch.is_floating_point(value):
                value.mul_(self.decay).add_(
                    current.to(dtype=value.dtype), alpha=1.0 - self.decay
                )
            else:
                value.copy_(current)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def learning_rate(
    step: int,
    total_steps: int,
    warmup_steps: int,
    maximum: float,
    minimum: float,
    warmup_factor: float,
) -> float:
    if warmup_steps and step < warmup_steps:
        progress = step / max(warmup_steps, 1)
        return maximum * (warmup_factor + (1.0 - warmup_factor) * progress)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[dict[str, float], float]:
    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        output = model(images)
        loss = nn.functional.binary_cross_entropy_with_logits(
            output, labels_device, reduction="none"
        )
        logits.append(output.cpu().numpy())
        targets.append(labels.numpy().astype(np.int64))
        losses.append(loss.cpu().numpy())
    combined_logits = np.concatenate(logits)
    combined_targets = np.concatenate(targets)
    metrics = binary_metrics(combined_logits, combined_targets, threshold)
    return metrics, float(np.concatenate(losses).mean())


def evaluate_checkpoint(
    checkpoint: str | Path,
    samples: Sequence[VerifierSample],
    parameters: dict[str, Any],
    threshold: float,
) -> dict[str, float]:
    device = resolve_device(str(parameters.get("device", "auto")))
    workers = int(parameters.get("workers", 4))
    loader = DataLoader(
        VerifierDataset(samples, train=False),
        batch_size=int(parameters.get("val_batch_size", 256)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = HelmetMicroNeXt().to(device)
    model.load_state_dict(value.get("model", value), strict=True)
    metrics, loss = evaluate(model, loader, device, threshold)
    del loader, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"loss": loss, "operating_threshold": threshold, **metrics}


def metric_rank(metrics: dict[str, float]) -> tuple[float, ...]:
    return (
        metrics["fpr_at_recall_95"],
        -metrics["achieved_recall_at_95"],
        -metrics["roc_auc"],
        -metrics["pr_auc"],
        -metrics["threshold_at_recall_95"],
    )


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_verifier(
    train_samples: Sequence[VerifierSample],
    validation_samples: Sequence[VerifierSample],
    parameters: dict[str, Any],
    output: str | Path,
    seed: int,
    save_best: bool,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    set_seed(seed, bool(parameters.get("deterministic", True)))
    device = resolve_device(str(parameters.get("device", "auto")))
    augmentation = {
        "crop_scale_min": float(parameters.get("crop_scale_min", 0.90)),
        "center_jitter": float(parameters.get("center_jitter", 0.03)),
        "photometric_probability": float(
            parameters.get("photometric_probability", 0.50)
        ),
        "photometric_strength": float(
            parameters.get("photometric_strength", 0.10)
        ),
    }
    workers = int(parameters.get("workers", 4))
    batch_size = int(parameters["batch_size"])
    train_loader = DataLoader(
        VerifierDataset(train_samples, train=True, augmentation=augmentation),
        batch_size=batch_size,
        sampler=balanced_sampler(train_samples, seed),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        VerifierDataset(validation_samples, train=False),
        batch_size=int(parameters.get("val_batch_size", 256)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    model = HelmetMicroNeXt().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["lr"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    ema = ModelEMA(model, float(parameters.get("ema_decay", 0.997)))
    epochs = int(parameters["epochs"])
    total_steps = epochs * len(train_loader)
    warmup_steps = round(
        float(parameters.get("warmup_epochs", 3.0)) * len(train_loader)
    )
    minimum_lr = float(parameters["lr"]) * float(
        parameters.get("minimum_lr_ratio", 0.01)
    )
    global_step = 0
    history: list[dict[str, Any]] = []
    best_rank: tuple[float, ...] | None = None
    best_metrics: dict[str, float] | None = None
    label_smoothing = float(parameters.get("label_smoothing", 0.0))
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        count = 0
        for images, labels in train_loader:
            rate = learning_rate(
                global_step,
                total_steps,
                warmup_steps,
                float(parameters["lr"]),
                minimum_lr,
                float(parameters.get("warmup_start_factor", 0.20)),
            )
            for group in optimizer.param_groups:
                group["lr"] = rate
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            smoothed = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, smoothed)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(parameters.get("grad_clip", 5.0))
            )
            optimizer.step()
            ema.update(model)
            total_loss += float(loss.detach()) * len(images)
            count += len(images)
            global_step += 1
        metrics, validation_loss = evaluate(ema.model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(count, 1),
            "validation_loss": validation_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_started,
            **metrics,
        }
        history.append(row)
        rank = metric_rank(metrics)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_metrics = metrics
            if save_best:
                torch.save(
                    {
                        "model": ema.model.state_dict(),
                        "raw_model": model.state_dict(),
                        "parameters": parameters,
                        "epoch": epoch,
                        "metrics": metrics,
                    },
                    output / "best.pt",
                )
        print(
            f"Epoch {epoch}/{epochs}: loss={row['train_loss']:.5f}, "
            f"FPR@R95={100 * metrics['fpr_at_recall_95']:.3f}%, "
            f"ROC-AUC={metrics['roc_auc']:.5f}",
            flush=True,
        )

    final_metrics = history[-1]
    checkpoint = {
        "model": ema.model.state_dict(),
        "raw_model": model.state_dict(),
        "parameters": parameters,
        "epoch": epochs,
        "metrics": final_metrics,
    }
    if save_best:
        torch.save(checkpoint, output / "last.pt")
    write_history(output / "history.csv", history)
    result = {
        "seed": seed,
        "parameters": parameters,
        "model_parameters": trainable_parameters(model),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "training_seconds": time.perf_counter() - training_started,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
    }
    dump_json(result, output / "summary.json")
    del train_loader, validation_loader, model, ema
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
