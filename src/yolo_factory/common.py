from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml


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


def materialize_dataset_config(config_path: str | Path) -> Path:
    root, config = resolve_dataset_paths(config_path)
    config["path"] = str(root)
    target = root / ".yolo-data.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
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
