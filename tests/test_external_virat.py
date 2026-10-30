import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from yolo_factory.external_virat import (
    extract_frames,
    geometry,
    import_virat,
    track_types,
)


def test_kpf_keeps_only_person_tracks(tmp_path: Path) -> None:
    types = tmp_path / "sample.types.yml"
    types.write_text(
        "- {types: {id1: 7, cset3: {Person: 1.0}}}\n"
        "- {types: {id1: 8, cset3: {Vehicle: 1.0}}}\n",
        encoding="utf-8",
    )
    geom = tmp_path / "sample.geom.yml"
    geom.write_text(
        "- {geom: {id1: 7, ts0: 30, g0: 10 20 30 60, src: truth}}\n"
        "- {geom: {id1: 8, ts0: 30, g0: 50 20 90 60, src: truth}}\n",
        encoding="utf-8",
    )

    frames = geometry(geom, track_types(types))

    assert list(frames) == [30]
    assert frames[30][0]["role"] == "person"
    assert frames[30][0]["xyxy"] == [10.0, 20.0, 30.0, 60.0]
    assert frames[30][0]["verified"] is False


@pytest.mark.parametrize(
    "coordinates",
    ["10 20 nan 60", "10 20 10 60", "10 20 30"],
)
def test_kpf_rejects_invalid_person_geometry(tmp_path: Path, coordinates: str) -> None:
    geom = tmp_path / "sample.geom.yml"
    geom.write_text(
        f"{{geom: {{id1: 7, ts0: 30, g0: {coordinates}, src: truth}}}}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        geometry(geom, {7: "Person"})


def test_extract_frames_does_not_accept_stale_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "frames"
    output.mkdir()
    stale = output / "00000003.jpg"
    stale.write_bytes(b"stale")

    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def get(self, key: int) -> int:
            return 1280 if key == 1 else 720

        def read(self) -> tuple[bool, None]:
            return False, None

        def release(self) -> None:
            pass

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FRAME_WIDTH=1,
        CAP_PROP_FRAME_HEIGHT=2,
        VideoCapture=lambda _: FakeCapture(),
        imwrite=lambda *_: True,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    with pytest.raises(ValueError, match="Video ended"):
        extract_frames(tmp_path / "video.mp4", [3], output)

    assert not stale.exists()


def test_failed_import_invalidates_previous_manifest(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "sample.geom.yml").write_text("{meta: sample}\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    manifest = output / "manifest.jsonl"
    stats = output / "import-stats.json"
    manifest.write_text("stale\n", encoding="utf-8")
    stats.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing types KPF"):
        import_virat(root, output, minimum_gap=30, boxes_per_video=10)

    assert not manifest.exists()
    assert not stats.exists()


def test_import_preserves_reviewed_manifest_and_existing_output(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    output = tmp_path / "output"
    frame = output / "frames" / "sample" / "00000003.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"existing")
    reviewed = output / "reviewed_manifest.jsonl"
    manifest = output / "manifest.jsonl"
    reviewed.write_text("reviewed\n", encoding="utf-8")
    manifest.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="archive the previous review"):
        import_virat(root, output, minimum_gap=30, boxes_per_video=10)

    assert reviewed.read_text(encoding="utf-8") == "reviewed\n"
    assert manifest.read_text(encoding="utf-8") == "existing\n"
    assert frame.read_bytes() == b"existing"
