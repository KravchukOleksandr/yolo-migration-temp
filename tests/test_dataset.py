from pathlib import Path

import pytest
from PIL import Image

from yolo_factory.dataset import box_diagonal, label_path, read_record
from yolo_factory.train import model_path


def test_read_record(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    image_root.mkdir()
    label_root.mkdir()
    image_path = image_root / "frame.jpg"
    Image.new("RGB", (100, 200)).save(image_path)
    (label_root / "frame.txt").write_text("0 0.5 0.5 0.2 0.1\n")

    record = read_record(image_path, image_root)

    assert record.width == 100
    assert record.height == 200
    assert len(record.boxes) == 1
    assert round(box_diagonal(record.boxes[0], record), 4) == 28.2843


def test_nested_label_path(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image = image_root / "camera-1" / "frame.png"

    assert label_path(image, image_root) == (
        tmp_path / "labels" / "camera-1" / "frame.txt"
    )


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("0 nan 0.5 0.2 0.1\n", "non-finite"),
        ("0 0.5 0.5 inf 0.1\n", "non-finite"),
        ("0.5 0.5 0.5 0.2 0.1\n", "invalid class ID"),
        ("3 0.5 0.5 0.2 0.1\n", "unknown class ID 3"),
    ],
)
def test_read_record_rejects_invalid_values(
    tmp_path: Path, line: str, message: str
) -> None:
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    image_root.mkdir()
    label_root.mkdir()
    image_path = image_root / "frame.jpg"
    Image.new("RGB", (100, 200)).save(image_path)
    (label_root / "frame.txt").write_text(line, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_record(image_path, image_root, {0, 1, 2})


def test_read_record_keeps_clippable_boundary_box(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    image_root.mkdir()
    label_root.mkdir()
    image_path = image_root / "frame.jpg"
    Image.new("RGB", (100, 200)).save(image_path)
    (label_root / "frame.txt").write_text("0 0.99 0.5 0.1 0.1\n", encoding="utf-8")

    record = read_record(image_path, image_root, {0})

    assert len(record.boxes) == 1


def test_local_model_path(tmp_path: Path) -> None:
    weights = tmp_path / "yolov8s.pt"
    weights.touch()

    assert model_path("s", tmp_path) == weights


def test_local_yolo26_model_path(tmp_path: Path) -> None:
    weights = tmp_path / "yolo26s.pt"
    weights.touch()

    assert model_path("s", tmp_path, "yolo26") == weights
