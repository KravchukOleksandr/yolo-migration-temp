from __future__ import annotations

import csv
import random
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

RGB_MEAN = np.asarray(
    [0.5226514935493469, 0.4618634879589081, 0.4670388400554657],
    dtype=np.float32,
)
RGB_STD = np.asarray(
    [0.2495836615562439, 0.2355685979127884, 0.23572568595409393],
    dtype=np.float32,
)


@dataclass(frozen=True)
class VerifierSample:
    path: Path
    label: int
    source_image: str
    split: str


def read_manifest(path: str | Path, split: str) -> list[VerifierSample]:
    manifest = Path(path).resolve()
    samples: list[VerifierSample] = []
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != split:
                continue
            sample_path = Path(row["crop"])
            if not sample_path.is_absolute():
                sample_path = manifest.parent / sample_path
            samples.append(
                VerifierSample(
                    path=sample_path.resolve(),
                    label=int(row["label"]),
                    source_image=row["source_image"],
                    split=split,
                )
            )
    if not samples:
        raise ValueError(f"No {split} samples in {manifest}")
    return samples


def group_holdout_split(
    samples: Sequence[VerifierSample],
    fraction: float,
    seed: int,
) -> tuple[list[VerifierSample], list[VerifierSample]]:
    if not 0.0 < fraction < 0.5:
        raise ValueError("Holdout fraction must be between 0 and 0.5")
    fit: list[VerifierSample] = []
    holdout: list[VerifierSample] = []
    boundary = int(fraction * 10_000)
    for sample in samples:
        key = f"{seed}:{sample.source_image}".encode()
        bucket = int(sha1(key).hexdigest()[:8], 16) % 10_000
        (holdout if bucket < boundary else fit).append(sample)
    for name, group in (("fit", fit), ("holdout", holdout)):
        labels = {sample.label for sample in group}
        if labels != {0, 1}:
            raise ValueError(f"Internal {name} split does not contain both classes")
    return fit, holdout


def tuning_split(
    samples: Sequence[VerifierSample],
    fraction: float,
    seed: int,
) -> tuple[list[VerifierSample], list[VerifierSample]]:
    return group_holdout_split(samples, fraction, seed)


def _square_crop(
    image: Image.Image,
    train: bool,
    crop_scale_min: float,
    center_jitter: float,
) -> Image.Image:
    width, height = image.size
    maximum = min(width, height)
    if not train:
        left = (width - maximum) // 2
        top = (height - maximum) // 2
        return image.crop((left, top, left + maximum, top + maximum))
    original = (0.10 * width, 0.10 * height, 0.90 * width, 0.90 * height)
    for _ in range(20):
        side = round(random.uniform(crop_scale_min, 1.0) * maximum)
        side = max(1, min(side, maximum))
        shift = center_jitter * maximum
        center_x = width / 2 + random.uniform(-shift, shift)
        center_y = height / 2 + random.uniform(-shift, shift)
        left = round(center_x - side / 2)
        top = round(center_y - side / 2)
        left = min(max(left, 0), width - side)
        top = min(max(top, 0), height - side)
        right = left + side
        bottom = top + side
        covered_x = max(0.0, min(right, original[2]) - max(left, original[0]))
        covered_y = max(0.0, min(bottom, original[3]) - max(top, original[1]))
        covered_x /= max(original[2] - original[0], 1e-6)
        covered_y /= max(original[3] - original[1], 1e-6)
        top_loss = max(0.0, top - original[1]) / max(
            original[3] - original[1], 1e-6
        )
        if covered_x >= 0.90 and covered_y >= 0.90 and top_loss <= 0.05:
            return image.crop((left, top, right, bottom))
    return _square_crop(image, False, crop_scale_min, center_jitter)


def preprocess_image(
    image: Image.Image,
    train: bool = False,
    crop_scale_min: float = 0.90,
    center_jitter: float = 0.03,
    photometric_probability: float = 0.50,
    photometric_strength: float = 0.10,
) -> torch.Tensor:
    image = _square_crop(
        image.convert("RGB"), train, crop_scale_min, center_jitter
    )
    if train and random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    resampling = Image.Resampling.BOX if image.width > 64 else Image.Resampling.BILINEAR
    image = image.resize((64, 64), resampling)
    if train and random.random() < photometric_probability:
        factor = random.uniform(1.0 - photometric_strength, 1.0 + photometric_strength)
        operation = random.randrange(3)
        if operation == 0:
            image = ImageEnhance.Brightness(image).enhance(factor)
        elif operation == 1:
            image = ImageEnhance.Contrast(image).enhance(factor)
        else:
            array = np.asarray(image, dtype=np.float32) / 255.0
            image = Image.fromarray(
                np.uint8(np.clip(np.power(array, factor) * 255.0, 0, 255))
            )
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - RGB_MEAN) / RGB_STD
    chw = np.ascontiguousarray(array.transpose(2, 0, 1))
    return torch.from_numpy(chw)


class VerifierDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        samples: Sequence[VerifierSample],
        train: bool,
        augmentation: dict[str, float] | None = None,
    ) -> None:
        self.samples = list(samples)
        self.train = train
        self.augmentation = augmentation or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = preprocess_image(
                image,
                train=self.train,
                **self.augmentation,
            )
        return tensor, torch.tensor(sample.label, dtype=torch.float32)


def balanced_sampler(
    samples: Sequence[VerifierSample], seed: int
) -> WeightedRandomSampler:
    counts = np.bincount([sample.label for sample in samples], minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training data must contain both classes")
    weights = torch.tensor(
        [1.0 / counts[sample.label] for sample in samples], dtype=torch.double
    )
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=len(samples),
        replacement=True,
        generator=generator,
    )
