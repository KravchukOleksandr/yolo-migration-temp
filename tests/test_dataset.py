from pathlib import Path

from PIL import Image

from yolo_factory.dataset import box_diagonal, label_path, read_record


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

