from pathlib import Path

import pytest
import yaml

from yolo_factory import pipeline, subset
from yolo_factory.pipeline import selected_trial_parameters


def trial_config() -> tuple[dict, dict]:
    tune = {
        "fixed": {
            "optimizer": "MuSGD",
            "amp": True,
            "imgsz": 960,
            "batch": 8,
            "patience": 5,
        },
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
        "batch": 8,
        "patience": 5,
    }
    (path / "args.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")
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


def pipeline_config(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "yolo26s.pt").write_bytes(b"test checkpoint")
    for split in ("train", "val"):
        (tmp_path / "data" / split / "images").mkdir(parents=True)
    values = {
        "data.yaml": {
            "path": "../data",
            "train": "train/images",
            "val": "val/images",
            "names": {0: "person", 1: "helmet-head", 2: "no-helmet-head"},
        },
        "train_yolo26.yaml": {
            "family": "yolo26",
            "model": "s",
            "data": "data.yaml",
            "validation_imgsz": [1280],
            "train": {
                "classes": [0, 2],
                "single_cls": False,
                "epochs": 100,
                "imgsz": 1280,
                "batch": 6,
                "rect": True,
                "multi_scale": 0.0,
            },
        },
        "tune_yolo26.yaml": {
            "family": "yolo26",
            "model": "s",
            "classes": [0, 2],
            "single_cls": False,
            "objective_class": 2,
            "data": "../runs/runtime/data_tune_yolo26.yaml",
            "fixed": {"optimizer": "MuSGD", "amp": True},
            "search_space": {"lr0": {}, "mosaic": {}},
        },
    }
    for name, value in values.items():
        (configs / name).write_text(yaml.safe_dump(value), encoding="utf-8")


def test_pipeline_defaults_are_blob_yolo26() -> None:
    args = pipeline.build_parser().parse_args([])

    assert args.data == "configs/data.yaml"
    assert args.train_config == "configs/train_yolo26.yaml"
    assert args.tune_config == "configs/tune_yolo26.yaml"
    assert not args.skip_audit


def test_check_is_non_mutating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline.sys, "argv", ["yolo-pipeline", "--check"])
    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: object())

    pipeline.main()

    assert not (tmp_path / "runs").exists()


def test_check_reports_missing_local_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_config(tmp_path)
    (tmp_path / "weights" / "yolo26s.pt").unlink()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="Missing local weights"):
        pipeline.preflight(pipeline.build_parser().parse_args([]))


def test_check_selected_trial_does_not_require_optuna(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_config(tmp_path)
    trial = write_trial(tmp_path / "trial-004", [0, 2])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        pipeline.importlib.util,
        "find_spec",
        lambda name: None if name == "optuna" else object(),
    )

    pipeline.preflight(pipeline.build_parser().parse_args(["--use-trial", str(trial)]))
    with pytest.raises(RuntimeError, match="optuna"):
        pipeline.preflight(pipeline.build_parser().parse_args([]))


@pytest.mark.parametrize("selected_trial", [False, True])
def test_pipeline_orchestrates_blob_training_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selected_trial: bool
) -> None:
    pipeline_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    arguments = ["yolo-pipeline", "--run-id", "test-run"]
    if selected_trial:
        trial = write_trial(tmp_path / "trial-004", [0, 2])
        arguments.extend(["--use-trial", str(trial)])
    monkeypatch.setattr(pipeline.sys, "argv", arguments)
    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: object())
    calls: list[tuple[str, tuple[object, ...]]] = []

    def run_module(module: str, *args: object) -> None:
        calls.append((module, args))
        if module == "yolo_factory.subset":
            target = Path(str(args[3]))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("names: {0: person}\n", encoding="utf-8")
        if module == "yolo_factory.tune":
            config = yaml.safe_load(Path(str(args[1])).read_text())
            output = Path(config["project"])
            output.mkdir(parents=True)
            (output / "best.json").write_text(
                yaml.safe_dump({"parameters": {"lr0": 0.0002, "optimizer": "MuSGD"}}),
                encoding="utf-8",
            )

    monkeypatch.setattr(pipeline, "run_module", run_module)

    pipeline.main()

    expected = ["audit", "train", "validate"]
    if not selected_trial:
        expected[1:1] = ["subset", "tune"]
    assert [name for name, _ in calls] == [f"yolo_factory.{name}" for name in expected]
    experiment = tmp_path / "runs" / "experiments" / "test-run"
    resolved = yaml.safe_load((experiment / "resolved-train.yaml").read_text())
    assert resolved["data"] == str(tmp_path / "configs" / "data.yaml")
    assert resolved["train"]["epochs"] == 100
    assert resolved["train"]["batch"] == 6
    assert resolved["train"]["imgsz"] == 1280
    assert resolved["train"]["lr0"] == (0.0004 if selected_trial else 0.0002)
    assert resolved["train"]["optimizer"] == "MuSGD"
    validate_args = list(calls[-1][1])
    assert validate_args[validate_args.index("--data") + 1] == (
        tmp_path / "configs" / "data.yaml"
    )
    assert validate_args[-3:] == ["--classes", 0, 2]
    assert (experiment / "selected-trial.yaml").exists() == selected_trial
    assert (experiment / "tune-data.yaml").exists() != selected_trial
    if not selected_trial:
        subset_args = list(calls[1][1])
        assert subset_args[subset_args.index("--output") + 1] == (
            experiment / "tune-data.yaml"
        )
        generated_tune = yaml.safe_load((experiment / "resolved-tune.yaml").read_text())
        assert generated_tune["data"] == str(experiment / "tune-data.yaml")
    assert not (tmp_path / "runs" / "runtime" / ".manifests").exists()


def test_pipeline_subsets_are_isolated_between_experiments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: object())

    def run_module(module: str, *args: object) -> None:
        if module == "yolo_factory.subset":
            with monkeypatch.context() as nested:
                nested.setattr(
                    pipeline.sys,
                    "argv",
                    ["yolo-subset", *map(str, args)],
                )
                subset.main()
        elif module == "yolo_factory.tune":
            config = yaml.safe_load(Path(str(args[1])).read_text())
            output = Path(config["project"])
            output.mkdir(parents=True)
            (output / "best.json").write_text("parameters: {lr0: 0.0002}\n")

    monkeypatch.setattr(pipeline, "run_module", run_module)
    for run_id in ("first", "second"):
        if run_id == "second":
            (tmp_path / "data" / "train" / "images" / "new.jpg").touch()
        monkeypatch.setattr(pipeline.sys, "argv", ["yolo-pipeline", "--run-id", run_id])
        pipeline.main()

    root = tmp_path / "runs" / "experiments"
    first = yaml.safe_load((root / "first" / "tune-data.yaml").read_text())
    second = yaml.safe_load((root / "second" / "tune-data.yaml").read_text())
    assert first["train"] != second["train"]
    assert Path(first["train"]).parent == root / "first" / ".manifests"
    assert Path(second["train"]).parent == root / "second" / ".manifests"
    assert not Path(first["train"]).read_text()
    assert "new.jpg" in Path(second["train"]).read_text()
