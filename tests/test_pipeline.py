from pathlib import Path

import pytest
import yaml

from yolo_factory.pipeline import selected_trial_parameters


def trial_config() -> tuple[dict, dict]:
    tune = {
        "fixed": {"optimizer": "MuSGD", "amp": True},
        "search_space": {"lr0": {}, "mosaic": {}},
    }
    train = {
        "family": "yolo26",
        "model": "s",
        "train": {"classes": [0, 2], "single_cls": False},
    }
    return tune, train


def write_trial(path: Path, classes: list[int]) -> Path:
    path.mkdir()
    values = {
        "model": "/models/yolo26s.pt",
        "classes": classes,
        "single_cls": False,
        "optimizer": "MuSGD",
        "amp": True,
        "lr0": 0.0004,
        "mosaic": 0.95,
        "epochs": 15,
        "imgsz": 960,
    }
    (path / "args.yaml").write_text(
        yaml.safe_dump(values), encoding="utf-8"
    )
    return path


def test_selected_trial_promotes_only_hyperparameters(tmp_path: Path) -> None:
    tune, train = trial_config()
    trial = write_trial(tmp_path / "trial-004", [0, 2])

    source, parameters = selected_trial_parameters(trial, tune, train)

    assert source == trial / "args.yaml"
    assert parameters == {
        "optimizer": "MuSGD",
        "amp": True,
        "lr0": 0.0004,
        "mosaic": 0.95,
    }


def test_selected_trial_rejects_different_classes(tmp_path: Path) -> None:
    tune, train = trial_config()
    trial = write_trial(tmp_path / "trial-004", [0])

    with pytest.raises(ValueError, match="classes"):
        selected_trial_parameters(trial, tune, train)
