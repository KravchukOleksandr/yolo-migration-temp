from pathlib import Path

import yaml

from yolo_factory.common import materialize_dataset_config


def test_dataset_change_invalidates_label_cache(tmp_path: Path) -> None:
    for split in ("train", "val"):
        (tmp_path / split / "images").mkdir(parents=True)
    config_path = tmp_path / "data.yaml"
    config = {
        "path": str(tmp_path),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "person"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    materialize_dataset_config(config_path)
    for split in ("train", "val"):
        (tmp_path / split / "labels.cache").touch()

    materialize_dataset_config(config_path)

    assert not (tmp_path / "train" / "labels.cache").exists()
    assert not (tmp_path / "val" / "labels.cache").exists()


def test_nested_source_label_caches_are_invalidated(tmp_path: Path) -> None:
    for split in ("train", "val"):
        images = tmp_path / split / "images"
        labels = tmp_path / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (labels / "sfchd.cache").touch()
        (labels / "shel5k.cache").touch()
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "train/images",
                "val": "val/images",
                "names": {0: "person"},
            }
        ),
        encoding="utf-8",
    )

    materialize_dataset_config(config_path)

    assert not list(tmp_path.rglob("*.cache"))


def test_manifest_label_caches_are_invalidated(tmp_path: Path) -> None:
    factory_image = tmp_path / "factory" / "images" / "frame.jpg"
    external_image = tmp_path / "external" / "sfchd" / "frame.jpg"
    for image in (factory_image, external_image):
        image.parent.mkdir(parents=True)
        image.touch()
    factory_cache = tmp_path / "factory" / "labels.cache"
    external_cache = tmp_path / "external" / "sfchd.cache"
    factory_cache.touch()
    external_cache.touch()
    manifest = tmp_path / "train.txt"
    manifest.write_text(
        f"{factory_image.resolve()}\n{external_image.resolve()}\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "train": str(manifest),
                "val": str(manifest),
                "names": {0: "person"},
            }
        ),
        encoding="utf-8",
    )

    materialize_dataset_config(config_path)

    assert not factory_cache.exists()
    assert not external_cache.exists()


def test_dataset_lists_invalidate_every_label_cache(tmp_path: Path) -> None:
    roots = []
    for name in ("factory", "external"):
        images = tmp_path / name / "images"
        images.mkdir(parents=True)
        (tmp_path / name / "labels.cache").touch()
        roots.append(str(images.relative_to(tmp_path)))
    val = tmp_path / "val" / "images"
    val.mkdir(parents=True)
    (tmp_path / "val" / "labels.cache").touch()
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": roots,
                "val": "val/images",
                "names": {0: "person"},
            }
        ),
        encoding="utf-8",
    )

    materialize_dataset_config(config_path)

    assert not (tmp_path / "factory" / "labels.cache").exists()
    assert not (tmp_path / "external" / "labels.cache").exists()
    assert not (tmp_path / "val" / "labels.cache").exists()


def test_flat_split_directory_invalidates_nested_source_caches(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train"
    val = tmp_path / "val"
    for split in (train, val):
        (split / "source").mkdir(parents=True)
        split.with_suffix(".cache").touch()
        (split / "source.cache").touch()
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "train",
                "val": "val",
                "names": {0: "person"},
            }
        ),
        encoding="utf-8",
    )

    materialize_dataset_config(config_path)

    assert not train.with_suffix(".cache").exists()
    assert not (train / "source.cache").exists()
    assert not val.with_suffix(".cache").exists()
    assert not (val / "source.cache").exists()
