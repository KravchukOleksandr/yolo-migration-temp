import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from yolo_factory import train, tune, validate


def dataset_config(tmp_path: Path) -> Path:
    path = tmp_path / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path / "data"),
                "train": "train/images",
                "val": "val/images",
                "names": {0: "person", 1: "helmet-head", 2: "no-helmet-head"},
            }
        )
    )
    return path


def fake_runtime(calls: list[dict], run_dir: Path):
    class FakeModel:
        def __init__(self, weights: str):
            calls.append({"weights": weights})
            self.trainer = SimpleNamespace(save_dir=run_dir)

        def train(self, **parameters):
            calls.append(parameters)
            return SimpleNamespace(
                results_dict={"metrics/mAP50-95(B)": 0.5},
                box=SimpleNamespace(maps=[0.4, 0.5, 0.6], ap_class_index=[0, 2]),
            )

        def val(self, **parameters):
            calls.append(parameters)
            return SimpleNamespace(results_dict={"map": 0.6}, speed={"inference": 1.2})

    @contextmanager
    def runtime():
        yield FakeModel

    return runtime


def test_resume_needs_only_checkpoint_not_base_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = dataset_config(tmp_path)
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"test checkpoint")
    config = tmp_path / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "family": "yolo26",
                "model": "s",
                "data": str(data),
                "weights_dir": "missing-base-weights",
                "train": {"classes": [0, 2], "single_cls": False, "imgsz": 1280},
            }
        )
    )
    calls = []
    monkeypatch.setattr(train, "offline_runtime", fake_runtime(calls, tmp_path / "run"))
    monkeypatch.setattr(train, "run_metadata", lambda: {})
    monkeypatch.setattr(
        sys, "argv", ["train", "--config", str(config), "--resume", str(checkpoint)]
    )

    train.main()

    assert calls[0]["weights"] == str(checkpoint)
    assert calls[1]["resume"] is True
    assert calls[1]["classes"] == [0, 2]
    assert calls[1]["single_cls"] is False
    assert (tmp_path / "run/run.json").is_file()


@pytest.mark.parametrize("batch", [0, -1])
def test_validation_rejects_nonpositive_batch(batch, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["validate", "--weights", "unused.pt", "--batch", str(batch)]
    )
    with pytest.raises(ValueError, match="positive integer"):
        validate.main()


def test_validation_missing_weights_never_starts_ultralytics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["validate", "--weights", "yolo26s.pt"])
    with pytest.raises(FileNotFoundError, match="Local checkpoint"):
        validate.main()


def test_validation_uses_local_weights_and_keeps_class_ids(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = dataset_config(tmp_path)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"test checkpoint")
    calls = []
    output = tmp_path / "validation"
    monkeypatch.setattr(
        validate, "offline_runtime", fake_runtime(calls, tmp_path / "run")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "--weights",
            str(checkpoint),
            "--data",
            str(data),
            "--classes",
            "0",
            "2",
            "--output",
            str(output),
        ],
    )

    validate.main()

    assert calls[0]["weights"] == str(checkpoint)
    assert calls[1]["classes"] == [0, 2]
    assert calls[1]["imgsz"] == 1280
    assert calls[1]["batch"] == 8
    assert calls[1]["plots"] is True
    assert json.loads((output / "summary.json").read_text())["1280"]["metrics"]


def test_objective_requires_ground_truth_for_requested_class():
    metrics = SimpleNamespace(
        results_dict={"metrics/mAP50-95(B)": 0.9},
        box=SimpleNamespace(maps=[0.9, 0.9, 0.9], ap_class_index=[0]),
    )
    with pytest.raises(ValueError, match="Objective class 2 is absent"):
        tune.objective_value(metrics, 2)
    assert tune.objective_value(metrics, None) == (0.9, 0.9)


def test_objective_uses_original_noncontiguous_class_id():
    metrics = SimpleNamespace(
        results_dict={"metrics/mAP50-95(B)": 0.5},
        box=SimpleNamespace(maps=[0.4, 0.5, 0.6], ap_class_index=[0, 2]),
    )
    assert tune.objective_value(metrics, 2) == (0.6, 0.5)


def test_cli_defaults_match_primary_yolo26_workflow():
    assert train.build_parser().parse_args([]).config == "configs/train_yolo26.yaml"
    assert tune.build_parser().parse_args([]).config == "configs/tune_yolo26.yaml"
    assert validate.build_parser().parse_args(["--weights", "x.pt"]).imgsz == [1280]
