import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from yolo_factory.download import blob_target, build_parser, download_container, main


class FakeContainer:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.etags = dict.fromkeys(files, "initial")
        self.response_etags: dict[str, str] = {}
        self.downloads: list[str] = []
        self.fail: str | None = None

    def list_blobs(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(name=name, size=len(content), etag=self.etags[name])
            for name, content in self.files.items()
        ]

    def download_blob(self, name: str) -> SimpleNamespace:
        self.downloads.append(name)

        def readinto(stream: object) -> None:
            stream.write(self.files[name])
            if name == self.fail:
                raise OSError("simulated interruption")

        return SimpleNamespace(
            readinto=readinto,
            properties=SimpleNamespace(
                etag=self.response_etags.get(name, f'"{self.etags[name]}"')
            ),
        )


class FakeService:
    account_name = "test-account"

    def __init__(self, files: dict[str, bytes]) -> None:
        self.container = FakeContainer(files)

    def get_container_client(self, name: str) -> FakeContainer:
        return self.container


@pytest.fixture
def azure_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    service = FakeService({"frame.jpg": b"image", "frame.txt": b""})
    service.get_container_client = Mock(wraps=service.get_container_client)
    factory = Mock(return_value=service)
    monkeypatch.setitem(
        sys.modules,
        "azure.storage.blob",
        SimpleNamespace(BlobServiceClient=SimpleNamespace(from_connection_string=factory)),
    )
    for name in (
        "AZURE_STORAGE_CONNECTION_STRING",
        "AZURE_TRAIN_CONTAINER",
        "AZURE_VAL_CONTAINER",
    ):
        monkeypatch.delenv(name, raising=False)
    return factory


@pytest.mark.parametrize("split", [None, "all", "train", "val"])
def test_cli_downloads_only_selected_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    azure_client: Mock,
    split: str | None,
) -> None:
    selected = ("train", "val") if split in (None, "all") else (split,)
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "test-connection")
    for name in selected:
        monkeypatch.setenv(f"AZURE_{name.upper()}_CONTAINER", f"test-{name}")
    arguments = ["download", "--output", str(tmp_path), "--workers", "1"]
    if split is not None:
        arguments.extend(["--split", split])
    monkeypatch.setattr(sys, "argv", arguments)
    untouched = tmp_path / "train" / "keep.txt"
    if split == "val":
        untouched.parent.mkdir()
        untouched.write_text("keep training data")

    main()

    azure_client.assert_called_once_with("test-connection")
    calls = azure_client.return_value.get_container_client.call_args_list
    assert [call.args[0] for call in calls] == [f"test-{name}" for name in selected]
    for name in selected:
        assert (tmp_path / name / "images/frame.jpg").read_bytes() == b"image"
        assert (tmp_path / name / "labels/frame.txt").read_bytes() == b""
        assert (tmp_path / name / ".download-state.json").is_file()
    if split == "val":
        assert untouched.read_text() == "keep training data"
        assert list(untouched.parent.iterdir()) == [untouched]
    if split == "train":
        assert not (tmp_path / "val").exists()


@pytest.mark.parametrize(
    "split,missing",
    [
        ("val", "AZURE_STORAGE_CONNECTION_STRING"),
        ("val", "AZURE_VAL_CONTAINER"),
        ("train", "AZURE_TRAIN_CONTAINER"),
        ("all", "AZURE_VAL_CONTAINER"),
    ],
)
def test_missing_required_env_fails_before_connection(
    monkeypatch: pytest.MonkeyPatch, azure_client: Mock, split: str, missing: str
) -> None:
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "test-connection")
    monkeypatch.setenv("AZURE_TRAIN_CONTAINER", "test-train")
    monkeypatch.setenv("AZURE_VAL_CONTAINER", "test-val")
    monkeypatch.delenv(missing)
    monkeypatch.setattr(sys, "argv", ["download", "--split", split])

    with pytest.raises(SystemExit, match=f"Set {missing}"):
        main()

    azure_client.assert_not_called()


def test_invalid_split_is_rejected() -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["--split", "validation"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "name,expected",
    [
        ("frame.jpg", "images/frame.jpg"),
        ("frame.txt", "labels/frame.txt"),
        ("sfchd/frame.jpg", "images/sfchd/frame.jpg"),
        ("sfchd/frame.txt", "labels/sfchd/frame.txt"),
        ("images/sfchd/frame.jpg", "images/sfchd/frame.jpg"),
        ("labels/sfchd/frame.TXT", "labels/sfchd/frame.txt"),
    ],
)
def test_blob_target_keeps_source_directories(name: str, expected: str) -> None:
    assert blob_target(name) == Path(expected)


@pytest.mark.parametrize("name", ["../frame.jpg", "/frame.jpg", "a/../frame.jpg"])
def test_blob_path_traversal_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="Unsafe blob path"):
        blob_target(name)


def test_nested_same_basenames_do_not_collide(tmp_path: Path) -> None:
    files = {
        "sfchd/frame.jpg": b"first image",
        "sfchd/frame.txt": b"0 0.5 0.5 0.4 0.8\n2 0.5 0.2 0.1 0.1\n",
        "shel5k/frame.jpg": b"second image",
        "shel5k/frame.txt": b"0 0.5 0.5 0.4 0.8\n",
    }
    service = FakeService(files)

    download_container(service, "train", tmp_path, workers=2)

    for name, content in files.items():
        assert (tmp_path / blob_target(name)).read_bytes() == content
    assert len(service.container.downloads) == 4


def test_changed_same_size_label_is_refreshed(tmp_path: Path) -> None:
    service = FakeService({"frame.jpg": b"image", "frame.txt": b"0 0.5 0.5 0.1 0.1"})
    download_container(service, "train", tmp_path, workers=1)
    service.container.downloads.clear()

    download_container(service, "train", tmp_path, workers=1)
    assert service.container.downloads == []

    service.container.files["frame.txt"] = b"2 0.5 0.5 0.1 0.1"
    service.container.etags["frame.txt"] = "updated"
    download_container(service, "train", tmp_path, workers=1)

    assert service.container.downloads == ["frame.txt"]
    assert (tmp_path / "labels/frame.txt").read_bytes().startswith(b"2 ")


@pytest.mark.parametrize(
    "listed,returned",
    [
        ("0x8DC123", '"0x8DC123"'),
        ('"0x8DC123"', "0x8DC123"),
        ('"0x8DC123"', '"0x8DC123"'),
        ("0x8DC123", "0x8DC123"),
    ],
)
@pytest.mark.parametrize("label", [b"", b"0 0.5 0.5 0.1 0.1"])
def test_equivalent_azure_etags_allow_download_and_resume(
    tmp_path: Path, listed: str, returned: str, label: bytes
) -> None:
    service = FakeService({"frame.jpg": b"image", "frame.txt": label})
    service.container.etags = dict.fromkeys(service.container.files, listed)
    service.container.response_etags = dict.fromkeys(service.container.files, returned)

    download_container(service, "train", tmp_path, workers=1)

    assert (tmp_path / "images/frame.jpg").read_bytes() == b"image"
    assert (tmp_path / "labels/frame.txt").read_bytes() == label
    state = json.loads((tmp_path / ".download-state.json").read_text())
    assert {record["etag"] for record in state["blobs"].values()} == {"0x8DC123"}
    service.container.downloads.clear()

    download_container(service, "train", tmp_path, workers=1)

    assert service.container.downloads == []


@pytest.mark.parametrize(
    "listed,returned",
    [
        ("version-two", '"version-three"'),
        ("0x8DC123", '"0x8dc123"'),
        ("0x8DC123", 'W/"0x8DC123"'),
    ],
)
def test_real_etag_change_preserves_previous_file_and_state(
    tmp_path: Path, listed: str, returned: str
) -> None:
    service = FakeService({"frame.jpg": b"image", "frame.txt": b"old"})
    download_container(service, "train", tmp_path, workers=1)
    old_state = (tmp_path / ".download-state.json").read_bytes()
    service.container.files["frame.txt"] = b"new"
    service.container.etags["frame.txt"] = listed
    service.container.response_etags["frame.txt"] = returned

    with pytest.raises(RuntimeError, match="Blob changed during download"):
        download_container(service, "train", tmp_path, workers=1)

    assert (tmp_path / "labels/frame.txt").read_bytes() == b"old"
    assert not (tmp_path / "labels/frame.txt.part").exists()
    assert (tmp_path / ".download-state.json").read_bytes() == old_state


def test_quoted_etag_in_previous_state_does_not_force_redownload(
    tmp_path: Path,
) -> None:
    service = FakeService({"frame.jpg": b"image", "frame.txt": b""})
    download_container(service, "train", tmp_path, workers=1)
    state_path = tmp_path / ".download-state.json"
    state = json.loads(state_path.read_text())
    for record in state["blobs"].values():
        record["etag"] = '"initial"'
    state_path.write_text(json.dumps(state))
    service.container.downloads.clear()

    download_container(service, "train", tmp_path, workers=1)

    assert service.container.downloads == []
    state = json.loads(state_path.read_text())
    assert {record["etag"] for record in state["blobs"].values()} == {"initial"}


def test_legacy_files_without_state_are_refreshed(tmp_path: Path) -> None:
    service = FakeService({"frame.jpg": b"image", "frame.txt": b"new"})
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "images/frame.jpg").write_bytes(b"older")
    (tmp_path / "labels/frame.txt").write_bytes(b"old")

    download_container(service, "train", tmp_path, workers=1)

    assert (tmp_path / "images/frame.jpg").read_bytes() == b"image"
    assert (tmp_path / "labels/frame.txt").read_bytes() == b"new"


def test_changed_image_invalidates_only_its_pixel_cache(tmp_path: Path) -> None:
    service = FakeService({"frame.jpg": b"image", "frame.txt": b""})
    download_container(service, "train", tmp_path, workers=1)
    cache = tmp_path / "images/frame.npy"
    other_cache = tmp_path / "images/other.npy"
    cache.write_bytes(b"old pixels")
    other_cache.write_bytes(b"keep")
    service.container.files["frame.jpg"] = b"new image"
    service.container.etags["frame.jpg"] = "updated"

    download_container(service, "train", tmp_path, workers=1)

    assert not cache.exists()
    assert other_cache.read_bytes() == b"keep"


@pytest.mark.parametrize(
    "files",
    [
        {"frame.jpg": b"image"},
        {"frame.txt": b"label"},
        {},
        {"a/frame.jpg": b"image", "b/frame.txt": b"label"},
    ],
)
def test_unpaired_inventory_fails_before_download(
    tmp_path: Path, files: dict[str, bytes]
) -> None:
    service = FakeService(files)
    with pytest.raises(RuntimeError, match="Invalid container inventory"):
        download_container(service, "train", tmp_path, workers=1)
    assert service.container.downloads == []


@pytest.mark.parametrize(
    "files,message",
    [
        (
            {"frame.jpg": b"a", "images/frame.jpg": b"b", "frame.txt": b""},
            "Multiple blobs",
        ),
        (
            {"frame.jpg": b"a", "frame.png": b"b", "frame.txt": b""},
            "Multiple image extensions",
        ),
    ],
)
def test_colliding_inventory_fails_before_download(
    tmp_path: Path, files: dict[str, bytes], message: str
) -> None:
    service = FakeService(files)
    with pytest.raises(ValueError, match=message):
        download_container(service, "train", tmp_path, workers=1)
    assert service.container.downloads == []


def test_stale_local_files_are_preserved_and_reported(tmp_path: Path) -> None:
    service = FakeService({"frame.jpg": b"new", "frame.txt": b""})
    stale = tmp_path / "images/old.jpg"
    stale.parent.mkdir()
    stale.write_bytes(b"keep this")

    with pytest.raises(RuntimeError, match="local dataset files are absent"):
        download_container(service, "train", tmp_path, workers=1)

    assert stale.read_bytes() == b"keep this"
    assert service.container.downloads == []


def test_failed_download_preserves_previous_file(tmp_path: Path) -> None:
    service = FakeService({"frame.jpg": b"image", "frame.txt": b"old"})
    download_container(service, "train", tmp_path, workers=1)
    old_state = (tmp_path / ".download-state.json").read_bytes()
    service.container.files["frame.txt"] = b"new"
    service.container.etags["frame.txt"] = "updated"
    service.container.fail = "frame.txt"

    with pytest.raises(OSError, match="simulated interruption"):
        download_container(service, "train", tmp_path, workers=1)

    assert (tmp_path / "labels/frame.txt").read_bytes() == b"old"
    assert not (tmp_path / "labels/frame.txt.part").exists()
    assert (tmp_path / ".download-state.json").read_bytes() == old_state


def test_initial_download_resumes_completed_files_after_failure(tmp_path: Path) -> None:
    service = FakeService({"frame.jpg": b"image", "frame.txt": b"label"})
    service.container.fail = "frame.txt"

    with pytest.raises(OSError, match="simulated interruption"):
        download_container(service, "train", tmp_path, workers=1)

    state = json.loads((tmp_path / ".download-state.json").read_text())
    assert set(state["blobs"]) == {"frame.jpg"}
    assert (tmp_path / "images/frame.jpg").read_bytes() == b"image"
    assert not (tmp_path / "labels/frame.txt").exists()
    service.container.fail = None
    service.container.downloads.clear()

    download_container(service, "train", tmp_path, workers=1)

    assert service.container.downloads == ["frame.txt"]
    assert (tmp_path / "labels/frame.txt").read_bytes() == b"label"


def test_symlink_escape_is_rejected_before_download(tmp_path: Path) -> None:
    output = tmp_path / "data"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "images").symlink_to(outside, target_is_directory=True)
    service = FakeService({"frame.jpg": b"image", "frame.txt": b""})

    with pytest.raises(ValueError, match="escapes output directory"):
        download_container(service, "train", output, workers=1)

    assert not list(outside.iterdir())
    assert service.container.downloads == []
