from __future__ import annotations

import json
import platform
import subprocess
from hashlib import sha1
from pathlib import Path
from typing import Any

import yaml

OPERATIONAL_TRAIN_KEYS = {
    "batch",
    "cache",
    "classes",
    "data",
    "deterministic",
    "device",
    "epochs",
    "exist_ok",
    "fraction",
    "imgsz",
    "mode",
    "model",
    "name",
    "patience",
    "plots",
    "profile",
    "project",
    "resume",
    "save",
    "save_period",
    "seed",
    "single_cls",
    "split",
    "task",
    "time",
    "val",
    "verbose",
    "workers",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def dump_json(value: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")


def resolve_dataset_paths(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    return root, config


def split_paths(root: Path, value: str | list[str]) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    paths = [Path(item) for item in values]
    return [path if path.is_absolute() else root / path for path in paths]


def promotable_train_parameters(*values: dict[str, Any]) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for mapping in values:
        parameters.update(
            (name, value)
            for name, value in mapping.items()
            if name not in OPERATIONAL_TRAIN_KEYS
        )
    return parameters


def _label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _manifest_images(path: Path) -> list[Path]:
    parent = path.parent
    images: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        image = Path(value)
        if value.startswith("./"):
            image = parent / value[2:]
        images.append(image)
    return images


def invalidate_label_caches(split_path: Path) -> None:
    if split_path.is_dir():
        labels = _label_path_for_image(split_path / "placeholder.jpg").parent
        labels.with_suffix(".cache").unlink(missing_ok=True)
        if labels.is_dir():
            for cache in labels.rglob("*.cache"):
                cache.unlink()
        return
    if not split_path.is_file():
        return
    cache_paths = {
        _label_path_for_image(image).parent.with_suffix(".cache")
        for image in _manifest_images(split_path)
    }
    for cache in cache_paths:
        cache.unlink(missing_ok=True)


def materialize_dataset_config(config_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    root, config = resolve_dataset_paths(config_path)
    config.pop("download", None)
    config["path"] = str(root)
    digest = sha1(str(config_path).encode()).hexdigest()[:10]
    target = Path("runs/runtime/datasets").resolve() / (
        f"{config_path.stem}-{digest}.yaml"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(config, sort_keys=False)
    for split in ("train", "val"):
        for split_path in split_paths(root, config[split]):
            invalidate_label_caches(split_path)
    target.write_text(content, encoding="utf-8")
    (root / ".yolo-data.yaml").unlink(missing_ok=True)
    return target


def run_metadata() -> dict[str, Any]:
    import torch

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
