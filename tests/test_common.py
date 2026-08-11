from pathlib import Path

import yaml

from yolo_factory.common import materialize_dataset_config


def test_dataset_change_invalidates_label_cache(tmp_path: Path) -> None:
    for split in ("train", "val"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels.cache").touch()
    config_path = tmp_path / "data.yaml"
    config = {
        "path": str(tmp_path),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "person"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    materialize_dataset_config(config_path)

    assert not (tmp_path / "train" / "labels.cache").exists()
    assert not (tmp_path / "val" / "labels.cache").exists()
