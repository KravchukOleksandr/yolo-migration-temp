import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def make_command(*arguments: str) -> str:
    return subprocess.run(
        ["make", "--no-print-directory", "-n", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_default_profile_uses_only_blob_dataset() -> None:
    data = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    train = yaml.safe_load((ROOT / "configs/train_yolo26.yaml").read_text())
    tune = yaml.safe_load((ROOT / "configs/tune_yolo26.yaml").read_text())

    assert data["path"] == "../data"
    assert data["train"] == "train/images"
    assert data["val"] == "val/images"
    assert data["names"][2] == "no-helmet-head"
    assert train["data"] == "data.yaml"
    assert not train.get("extra_validations")
    assert train["model"] == tune["model"] == "s"
    assert train["family"] == tune["family"] == "yolo26"
    assert train["train"]["classes"] == tune["classes"] == [0, 2]
    assert tune["objective_class"] == 2
    assert train["train"]["imgsz"] == tune["imgsz"] == 1280
    assert train["train"]["rect"] is tune["fixed"]["rect"] is True
    assert train["train"]["multi_scale"] == tune["fixed"]["multi_scale"] == 0


def test_make_start_aliases_use_default_profile() -> None:
    expected = make_command("start")
    assert make_command() == expected
    assert make_command("yolo26") == expected
    assert '--data "configs/data.yaml"' in expected
    assert '--train-config "configs/train_yolo26.yaml"' in expected
    assert '--tune-config "configs/tune_yolo26.yaml"' in expected
    assert "external_dataset" not in expected
    assert "data_external" not in expected


def test_make_custom_profile_and_trial_share_parameters() -> None:
    arguments = (
        "DATA_CONFIG=configs/custom-data.yaml",
        "TRAIN_CONFIG=configs/custom-train.yaml",
        "TUNE_CONFIG=configs/custom-tune.yaml",
        "PYTHON=/custom/venv/bin/python",
    )
    for target in ("start", "yolo26-from-trial"):
        command = make_command(target, "TRIAL=custom trial", *arguments)
        assert 'PYTHON="/custom/venv/bin/python"' in command
        assert '--data "configs/custom-data.yaml"' in command
        assert '--train-config "configs/custom-train.yaml"' in command
        assert '--tune-config "configs/custom-tune.yaml"' in command
    assert '--use-trial "custom trial"' in command


def test_make_legacy_and_validation_commands_are_explicit() -> None:
    legacy = make_command("yolov8")
    assert "configs/train.yaml" in legacy
    assert "configs/tune.yaml" in legacy
    validation = make_command("validate", "WEIGHTS=weights/custom.pt")
    assert '--weights "weights/custom.pt"' in validation
    assert "--imgsz 1280 --batch 8" in validation
    assert "--classes 0 2" in validation
