import os
from pathlib import Path

import pytest
import yaml

from yolo_factory.common import materialize_dataset_config
from yolo_factory.offline import offline_runtime


def test_offline_runtime_blocks_downloads_and_uses_local_fonts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import requests

    def no_network(*args, **kwargs):
        raise AssertionError("Unexpected network request")

    monkeypatch.setattr(requests, "get", no_network)
    monkeypatch.setattr(requests, "head", no_network)
    with offline_runtime():
        from ultralytics import utils
        from ultralytics.data import utils as data_utils
        from ultralytics.utils import callbacks, checks, downloads, plotting

        assert os.environ["YOLO_OFFLINE"] == "true"
        assert not utils.ONLINE
        assert not checks.AUTOINSTALL
        assert checks.check_pip_update_available() is False
        assert callbacks.add_integration_callbacks(object()) is None
        for module in (checks, data_utils, plotting):
            assert Path(module.check_font("Arial.ttf")).is_file()
        with pytest.raises(ConnectionError, match="Offline mode"):
            downloads.safe_download("https://example.invalid/model.pt")
        with pytest.raises(ConnectionError, match="Offline mode"):
            downloads.attempt_download_asset("yolo26n.pt")
    assert (tmp_path / "runs/runtime/ultralytics/Arial.ttf").is_file()


def test_dataset_materialization_drops_download_instructions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "data.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "path": "data",
                "train": "train/images",
                "val": "val/images",
                "names": {0: "person", 1: "helmet-head", 2: "no-helmet-head"},
                "download": "bash unused-download.sh",
            }
        )
    )
    result = yaml.safe_load(materialize_dataset_config(source).read_text())
    assert "download" not in result
    assert "download" in yaml.safe_load(source.read_text())
    assert result["names"] == {0: "person", 1: "helmet-head", 2: "no-helmet-head"}
