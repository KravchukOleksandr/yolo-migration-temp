from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    image: Path
    label: Path
    width: int
    height: int
    boxes: tuple[tuple[float, float, float, float, float], ...]


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )


def label_path(image_path: Path, image_root: Path) -> Path:
    labels_root = image_root.parent / "labels"
    return (labels_root / image_path.relative_to(image_root)).with_suffix(".txt")


def read_record(image_path: Path, image_root: Path) -> ImageRecord:
    with Image.open(image_path) as image:
        width, height = image.size
        image.verify()
    path = label_path(image_path, image_root)
    boxes: list[tuple[float, float, float, float, float]] = []
    if path.exists():
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number}: expected 5 values")
            class_id, x, y, box_width, box_height = map(float, fields)
            if class_id < 0 or any(value < 0 or value > 1 for value in (x, y)):
                raise ValueError(f"{path}:{line_number}: invalid class or center")
            if box_width <= 0 or box_height <= 0 or box_width > 1 or box_height > 1:
                raise ValueError(f"{path}:{line_number}: invalid box size")
            if (
                x - box_width / 2 < 0
                or x + box_width / 2 > 1
                or y - box_height / 2 < 0
                or y + box_height / 2 > 1
            ):
                raise ValueError(f"{path}:{line_number}: box crosses image boundary")
            boxes.append((class_id, x, y, box_width, box_height))
    return ImageRecord(image_path, path, width, height, tuple(boxes))


def box_diagonal(
    box: tuple[float, float, float, float, float], record: ImageRecord
) -> float:
    return math.hypot(box[3] * record.width, box[4] * record.height)
