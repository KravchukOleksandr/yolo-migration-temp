import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

import yolo_factory.external_dataset as external_dataset
from yolo_factory.external_dataset import (
    SourceBox,
    SourceFrame,
    assign_splits,
    build,
    frame_errors,
    has_orphan_head,
    head_coverage_complete,
    load_source,
    object_metadata,
    publish_output,
    split_integrity,
)


def write_image(path: Path, color: str = "white") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color).save(path)


def test_coco_roles_keep_no_helmet_as_class_two(tmp_path: Path) -> None:
    root = tmp_path / "sfchd"
    write_image(root / "images" / "camera_001.jpg")
    annotations = {
        "images": [
            {"id": 1, "file_name": "camera_001.jpg", "width": 100, "height": 80}
        ],
        "categories": [
            {"id": 0, "name": "person"},
            {"id": 4, "name": "head"},
            {"id": 5, "name": "blur_head"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 5, 40, 70]},
            {"id": 2, "image_id": 1, "category_id": 4, "bbox": [20, 5, 10, 10]},
            {"id": 3, "image_id": 1, "category_id": 5, "bbox": [60, 5, 10, 10]},
        ],
    }
    annotation_path = root / "annotations" / "train.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    config = {
        "format": "coco",
        "root": str(root),
        "annotations": ["**/annotations/train.json"],
        "category_roles": {
            "person": "person",
            "head": "no_helmet",
            "blur_head": "unknown",
        },
        "no_helmet_complete": True,
    }

    frames = load_source("sfchd", config)

    assert [box.role for box in frames[0].boxes] == [
        "person",
        "no_helmet",
        "unknown",
    ]
    assert not frames[0].joint_complete


def test_manifest_rejects_image_outside_source_root(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    write_image(tmp_path / "outside.jpg")
    record = {
        "image": "../outside.jpg",
        "source_id": "outside",
        "width": 100,
        "height": 80,
        "group_id": "camera-a",
        "objects": [{"role": "person", "xyxy": [10, 5, 50, 75]}],
    }
    (root / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="image is outside source root"):
        load_source("sample", {"format": "manifest", "root": str(root)})


@pytest.mark.parametrize(
    "location,key",
    [
        ("frame", "person_annotation_complete"),
        ("frame", "no_helmet_annotation_complete"),
        ("object", "verified"),
        ("attributes", "head_review_complete"),
        ("attributes", "person_review_complete"),
    ],
)
def test_manifest_rejects_string_boolean(
    tmp_path: Path, location: str, key: str
) -> None:
    root = tmp_path / "source"
    write_image(root / "frame.jpg")
    record = {
        "image": "frame.jpg",
        "source_id": "frame",
        "width": 100,
        "height": 80,
        "group_id": "camera-a",
        "objects": [{"role": "person", "xyxy": [10, 5, 50, 75]}],
    }
    if location == "frame":
        record[key] = "false"
    elif location == "object":
        record["objects"][0][key] = "false"
    else:
        record["attributes"] = {key: "false"}
    (root / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"{key} must be a boolean"):
        load_source("sample", {"format": "manifest", "root": str(root)})


def test_build_keeps_incomplete_frames_out_of_joint_view(tmp_path: Path) -> None:
    output = tmp_path / "external_dataset"
    raw = output / "raw" / "sample"
    write_image(raw / "complete.jpg", "white")
    write_image(raw / "pending.jpg", "black")
    records = [
        {
            "image": "complete.jpg",
            "source_id": "complete",
            "width": 100,
            "height": 80,
            "group_id": "camera-a",
            "split": "train",
            "person_annotation_complete": True,
            "no_helmet_annotation_complete": True,
            "objects": [
                {"role": "person", "xyxy": [10, 5, 50, 75]},
                {"role": "no_helmet", "xyxy": [20, 5, 30, 15]},
            ],
        },
        {
            "image": "pending.jpg",
            "source_id": "pending",
            "width": 100,
            "height": 80,
            "group_id": "camera-b",
            "split": "train",
            "person_annotation_complete": True,
            "no_helmet_annotation_complete": False,
            "objects": [{"role": "person", "xyxy": [10, 5, 50, 75]}],
        },
    ]
    (raw / "manifest.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    config = {
        "output": str(output),
        "seed": 42,
        "validation_fraction": 0.2,
        "input_size": [1280, 736],
        "sources": {
            "sample": {
                "format": "manifest",
                "root": str(raw),
                "manifest": "manifest.jsonl",
                "train_person_boxes": 2,
                "respect_source_split": True,
            }
        },
    }
    config_path = tmp_path / "external.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    stats = build(config_path)

    flat_images = list((output / "train" / "sample").glob("*.jpg"))
    joint_images = list((output / ".yolo" / "train" / "images").rglob("*.jpg"))
    labels = list((output / "train" / "sample").glob("*.txt"))
    assert len(flat_images) == 2
    assert len(labels) == 2
    assert len(joint_images) == 1
    assert any(line.startswith("2 ") for line in labels[0].read_text().splitlines())
    assert stats["pending_review"] == 1


def test_quota_is_selected_after_reference_deduplication(tmp_path: Path) -> None:
    output = tmp_path / "external_dataset"
    factory = tmp_path / "factory"
    raw = output / "raw" / "sample"
    write_image(factory / "duplicate.jpg", "white")
    write_image(raw / "duplicate.jpg", "white")
    write_image(raw / "new.jpg", "black")
    records = [
        {
            "image": name,
            "source_id": name,
            "width": 100,
            "height": 80,
            "group_id": name,
            "split": "train",
            "person_annotation_complete": True,
            "no_helmet_annotation_complete": True,
            "objects": [{"role": "person", "xyxy": [10, 5, 50, 75]}],
        }
        for name in ("duplicate.jpg", "new.jpg")
    ]
    (raw / "manifest.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    config = {
        "output": str(output),
        "seed": 42,
        "validation_fraction": 0.2,
        "deduplicate_against": [str(factory)],
        "near_duplicate_hamming": -1,
        "sources": {
            "sample": {
                "format": "manifest",
                "root": str(raw),
                "train_person_boxes": 1,
                "respect_source_split": True,
            }
        },
    }
    config_path = tmp_path / "external.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    stats = build(config_path)

    assert stats["sources"]["sample"]["exact_duplicates_removed"] == 1
    assert stats["sources"]["sample"]["train_person"] == 1
    assert list((output / "train" / "sample").glob("*.jpg"))[0].name.startswith("new-")


def test_disputed_frames_are_excluded_before_quota_selection(tmp_path: Path) -> None:
    output = tmp_path / "external_dataset"
    raw = output / "raw" / "sample"
    write_image(raw / "duplicate-boxes.jpg", "black")
    write_image(raw / "unresolved-head.jpg", "gray")
    write_image(raw / "orphan-head.jpg", "blue")
    write_image(raw / "valid.jpg", "white")
    common = {
        "width": 100,
        "height": 80,
        "split": "train",
        "person_annotation_complete": True,
        "no_helmet_annotation_complete": True,
    }
    records = [
        {
            **common,
            "image": "duplicate-boxes.jpg",
            "source_id": "duplicate-boxes",
            "group_id": "camera-a",
            "objects": [
                {"role": "person", "xyxy": [10, 5, 50, 75]},
                {"role": "person", "xyxy": [10, 5, 50, 75]},
            ],
        },
        {
            **common,
            "image": "unresolved-head.jpg",
            "source_id": "unresolved-head",
            "group_id": "camera-b",
            "objects": [{"role": "person", "xyxy": [10, 5, 50, 75]}],
        },
        {
            **common,
            "image": "valid.jpg",
            "source_id": "valid",
            "group_id": "camera-c",
            "objects": [
                {"role": "person", "xyxy": [10, 5, 50, 75]},
                {"role": "helmet_present", "xyxy": [20, 5, 30, 15]},
            ],
        },
        {
            **common,
            "image": "orphan-head.jpg",
            "source_id": "orphan-head",
            "group_id": "camera-d",
            "objects": [
                {"role": "person", "xyxy": [10, 5, 50, 75]},
                {"role": "helmet_present", "xyxy": [70, 5, 80, 15]},
            ],
        },
    ]
    (raw / "manifest.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    config_path = tmp_path / "external.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output": str(output),
                "validation_fraction": 0.2,
                "near_duplicate_hamming": -1,
                "sources": {
                    "sample": {
                        "format": "manifest",
                        "root": str(raw),
                        "train_person_boxes": 1,
                        "respect_source_split": True,
                        "head_evidence_roles": ["no_helmet", "helmet_present"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    stats = build(config_path)

    selected_images = list((output / "train" / "sample").glob("*.jpg"))
    excluded = [
        json.loads(line)
        for line in (output / "manifests" / "excluded.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    excluded_by_id = {item["source_id"]: item for item in excluded}
    selected_manifest = json.loads(
        (output / "manifests" / "manifest.jsonl").read_text(encoding="utf-8")
    )
    selected_objects = [
        json.loads(line)
        for line in (output / "manifests" / "objects.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(selected_images) == 1
    assert selected_images[0].name.startswith("valid-")
    assert stats["sources"]["sample"]["qa_excluded_images"] == 3
    assert stats["sources"]["sample"]["train_person"] == 1
    assert "duplicate_box" in excluded_by_id["duplicate-boxes"]["reason"]
    assert (
        excluded_by_id["unresolved-head"]["reason"] == "unresolved_person_head_status"
    )
    assert excluded_by_id["orphan-head"]["reason"] == "head_without_person_annotation"
    assert selected_manifest["assurance"] == "source_annotations_automated_qa"
    assert all(not item["verified"] for item in selected_objects)


def test_manual_image_exclusion_is_recorded_before_quota_selection(
    tmp_path: Path,
) -> None:
    output = tmp_path / "external_dataset"
    raw = output / "raw" / "sample"
    write_image(raw / "manual.jpg", "black")
    write_image(raw / "replacement.jpg", "white")
    records = [
        {
            "image": image,
            "source_id": source_id,
            "width": 100,
            "height": 80,
            "group_id": source_id,
            "split": "train",
            "person_annotation_complete": True,
            "no_helmet_annotation_complete": True,
            "objects": [
                {"role": "person", "xyxy": [10, 5, 50, 75]},
                {"role": "helmet_present", "xyxy": [20, 5, 30, 15]},
            ],
        }
        for image, source_id in (
            ("manual.jpg", "manual"),
            ("replacement.jpg", "replacement"),
        )
    ]
    (raw / "manifest.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    config_path = tmp_path / "external.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output": str(output),
                "validation_fraction": 0.2,
                "near_duplicate_hamming": -1,
                "sources": {
                    "sample": {
                        "format": "manifest",
                        "root": str(raw),
                        "train_person_boxes": 1,
                        "respect_source_split": True,
                        "head_evidence_roles": ["helmet_present"],
                        "exclude_images": {"manual.jpg": "known_reflective_copy"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    stats = build(config_path)

    selected_images = list((output / "train" / "sample").glob("*.jpg"))
    excluded = [
        json.loads(line)
        for line in (output / "manifests" / "excluded.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(selected_images) == 1
    assert selected_images[0].name.startswith("replacement-")
    assert stats["sources"]["sample"]["train_person"] == 1
    assert excluded[0]["source_id"] == "manual"
    assert excluded[0]["qa_errors"] == ["manual_exclusion:known_reflective_copy"]


def test_grouped_split_never_separates_video_frames(tmp_path: Path) -> None:
    frames = [
        SourceFrame(
            source="video",
            source_id=f"{group}:{index}",
            image=tmp_path / f"{group}-{index}.jpg",
            width=100,
            height=80,
            group_id=group,
            source_split=None,
        )
        for group in ("video-a", "video-b")
        for index in range(3)
    ]

    assign_splits(frames, 0.2, 42)

    assert {frame.source_split for frame in frames} == {"train", "val"}
    assert all(
        len({frame.source_split for frame in frames if frame.group_id == group}) == 1
        for group in ("video-a", "video-b")
    )


@pytest.mark.parametrize(
    "box",
    [
        (float("nan"), 5.0, 50.0, 75.0),
        (10.0, 5.0, float("inf"), 75.0),
        (50.0, 5.0, 10.0, 75.0),
    ],
)
def test_frame_errors_rejects_nonfinite_or_invalid_boxes(
    tmp_path: Path, box: tuple[float, float, float, float]
) -> None:
    image = tmp_path / "frame.jpg"
    write_image(image)
    frame = SourceFrame(
        source="sample",
        source_id="frame",
        image=image,
        width=100,
        height=80,
        group_id="camera-a",
        source_split="train",
        boxes=[SourceBox("person", "person", box)],
    )

    assert "invalid_box:0:person" in frame_errors(frame)


def test_frame_errors_handles_wrong_sized_box_without_crashing(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.jpg"
    write_image(image)
    frame = SourceFrame(
        source="sample",
        source_id="frame",
        image=image,
        width=100,
        height=80,
        group_id="camera-a",
        source_split="train",
        boxes=[
            SourceBox("person", "person", (10.0, 5.0, 50.0)),
            SourceBox("person", "person", (10.0, 5.0, 50.0, 75.0)),
        ],
    )

    assert "invalid_box:0:person" in frame_errors(frame)


@pytest.mark.parametrize("width,height", [(0, 80), (-1, 80), (100, 0)])
def test_frame_errors_rejects_invalid_dimensions(
    tmp_path: Path, width: int, height: int
) -> None:
    image = tmp_path / "frame.jpg"
    write_image(image)
    frame = SourceFrame(
        source="sample",
        source_id="frame",
        image=image,
        width=width,
        height=height,
        group_id="camera-a",
        source_split="train",
        boxes=[SourceBox("person", "person", (10.0, 5.0, 50.0, 75.0))],
    )

    errors = frame_errors(frame)

    assert any(error.startswith("image_size_mismatch:") for error in errors)
    assert "invalid_box:0:person" in errors


def test_frame_errors_rejects_duplicate_boxes_at_iou_threshold(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.jpg"
    write_image(image)
    frame = SourceFrame(
        source="sample",
        source_id="frame",
        image=image,
        width=100,
        height=80,
        group_id="camera-a",
        source_split="train",
        boxes=[
            SourceBox("person", "person", (0.0, 0.0, 10.0, 10.0)),
            SourceBox("person", "person", (0.0, 0.0, 8.0, 10.0)),
        ],
    )

    assert "duplicate_box:0:1:person" in frame_errors(frame)


def test_head_coverage_requires_distinct_evidence_per_person(tmp_path: Path) -> None:
    person_boxes = [
        SourceBox("person", "person", (0.0, 0.0, 100.0, 100.0)),
        SourceBox("person", "person", (40.0, 0.0, 100.0, 100.0)),
    ]
    shared_head = SourceBox("helmet", "helmet_present", (50.0, 0.0, 60.0, 10.0))
    frame = SourceFrame(
        source="sample",
        source_id="frame",
        image=tmp_path / "unused.jpg",
        width=100,
        height=100,
        group_id="camera-a",
        source_split="train",
        boxes=person_boxes + [shared_head],
    )

    assert not head_coverage_complete(frame, {"helmet_present"})

    frame.boxes.append(SourceBox("helmet", "helmet_present", (10.0, 0.0, 20.0, 10.0)))
    assert head_coverage_complete(frame, {"helmet_present"})


@pytest.mark.parametrize(
    "head",
    [
        SourceBox("helmet", "helmet_indicator", (70.0, 5.0, 80.0, 15.0)),
        SourceBox("head_with_helmet", "helmet_present", (20.0, 5.0, 30.0, 15.0)),
    ],
)
def test_orphan_check_ignores_isolated_helmet_and_accepts_head_inside_person(
    tmp_path: Path, head: SourceBox
) -> None:
    frame = SourceFrame(
        source="sample",
        source_id="frame",
        image=tmp_path / "unused.jpg",
        width=100,
        height=80,
        group_id="camera-a",
        source_split="train",
        boxes=[
            SourceBox("person", "person", (10.0, 0.0, 50.0, 75.0)),
            head,
        ],
    )

    assert not has_orphan_head(frame)


@pytest.mark.parametrize("role", ["no_helmet", "helmet_present"])
def test_orphan_check_flags_strong_head_without_person(
    tmp_path: Path, role: str
) -> None:
    frame = SourceFrame(
        source="sample",
        source_id="frame",
        image=tmp_path / "unused.jpg",
        width=100,
        height=80,
        group_id="camera-a",
        source_split="train",
        boxes=[
            SourceBox("person", "person", (10.0, 0.0, 50.0, 75.0)),
            SourceBox(role, role, (70.0, 5.0, 80.0, 15.0)),
        ],
    )

    assert has_orphan_head(frame)


def test_projected_object_sizes_distinguish_square_and_widescreen() -> None:
    box = SourceBox("head", "no_helmet", (0.0, 0.0, 100.0, 100.0))
    square = SourceFrame(
        source="sample",
        source_id="square",
        image=Path("square.jpg"),
        width=1000,
        height=1000,
        group_id="square",
        source_split="train",
        boxes=[box],
    )
    widescreen = SourceFrame(
        source="sample",
        source_id="widescreen",
        image=Path("widescreen.jpg"),
        width=1280,
        height=720,
        group_id="widescreen",
        source_split="train",
        boxes=[box],
    )

    square_size = object_metadata(square, box, "square", [1280, 736], 1280)
    widescreen_size = object_metadata(widescreen, box, "widescreen", [1280, 736], 1280)

    assert square_size["deployment_letterbox_size"] == pytest.approx([73.6, 73.6])
    assert square_size["rect_baseline_size"] == pytest.approx([128.0, 128.0])
    assert widescreen_size["deployment_letterbox_size"] == pytest.approx(
        widescreen_size["rect_baseline_size"]
    )


def test_assign_splits_rejects_conflicting_preassigned_group(
    tmp_path: Path,
) -> None:
    frames = [
        SourceFrame(
            source="sample",
            source_id=f"frame-{split}",
            image=tmp_path / f"frame-{split}.jpg",
            width=100,
            height=80,
            group_id="same-video",
            source_split=split,
        )
        for split in ("train", "val")
    ]

    with pytest.raises(ValueError, match="both train and val"):
        assign_splits(frames, 0.2, 42)


def test_split_integrity_rejects_video_across_splits() -> None:
    items = [
        {
            "source": "sample",
            "split": split,
            "group_id": f"group-{split}",
            "video_id": "same-video",
            "camera_id": None,
            "sha256": f"sha-{split}",
            "pixel_sha256": f"pixels-{split}",
        }
        for split in ("train", "val")
    ]

    with pytest.raises(ValueError, match="Cross-split overlap for sample: video_id"):
        split_integrity(items)


def test_build_preserves_existing_export_when_staging_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "external_dataset"
    sentinel = output / "train" / "sample" / "keep.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("existing export", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    config_path = tmp_path / "external.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output": str(output),
                "validation_fraction": 0.2,
                "sources": {
                    "sample": {
                        "root": str(source),
                        "train_person_boxes": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fail_staging(config: dict, stage: Path) -> dict:
        (stage / "train").mkdir()
        (stage / "train" / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("candidate failed")

    monkeypatch.setattr(external_dataset, "build_staged", fail_staging)

    with pytest.raises(RuntimeError, match="candidate failed"):
        build(config_path)

    assert sentinel.read_text(encoding="utf-8") == "existing export"
    assert not (output / "train" / "partial.txt").exists()


def test_build_rejects_source_root_containing_output(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "external_dataset"
    config_path = tmp_path / "external.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output": str(output),
                "validation_fraction": 0.2,
                "sources": {
                    "sample": {
                        "format": "auto",
                        "root": str(tmp_path),
                        "train_person_boxes": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Source root overlaps managed export"):
        build(config_path)

    assert not output.exists()


def test_first_strict_build_does_not_publish_when_source_is_unavailable(
    tmp_path: Path,
) -> None:
    output = tmp_path / "external_dataset"
    config_path = tmp_path / "external.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output": str(output),
                "validation_fraction": 0.2,
                "sources": {
                    "sample": {
                        "format": "manifest",
                        "root": str(tmp_path / "missing-source"),
                        "train_person_boxes": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Sources unavailable: sample"):
        build(config_path)

    assert not (output / "stats.json").exists()
    assert not (output / "train").exists()
    assert not (output / "val").exists()
    assert not (output / ".yolo").exists()
    assert not (output / "manifests").exists()


def test_build_preserves_existing_export_when_previous_source_is_blocked(
    tmp_path: Path,
) -> None:
    output = tmp_path / "external_dataset"
    sentinel = output / "train" / "sample" / "keep.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("existing export", encoding="utf-8")
    stats = {"sources": {"sample": {"status": "selected", "train_person": 1}}}
    (output / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    config_path = tmp_path / "external.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output": str(output),
                "validation_fraction": 0.2,
                "sources": {
                    "sample": {
                        "format": "manifest",
                        "root": str(tmp_path / "missing-source"),
                        "train_person_boxes": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Previous export preserved"):
        build(config_path)

    assert sentinel.read_text(encoding="utf-8") == "existing export"
    assert json.loads((output / "stats.json").read_text(encoding="utf-8")) == stats


def test_successful_rebuild_removes_old_yolo_cache(tmp_path: Path) -> None:
    output = tmp_path / "external_dataset"
    raw = output / "raw" / "sample"
    write_image(raw / "frame.jpg")
    record = {
        "image": "frame.jpg",
        "source_id": "frame",
        "width": 100,
        "height": 80,
        "group_id": "camera-a",
        "split": "train",
        "person_annotation_complete": True,
        "no_helmet_annotation_complete": True,
        "objects": [{"role": "person", "xyxy": [10, 5, 50, 75]}],
    }
    (raw / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    config_path = tmp_path / "external.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output": str(output),
                "validation_fraction": 0.2,
                "near_duplicate_hamming": -1,
                "sources": {
                    "sample": {
                        "format": "manifest",
                        "root": str(raw),
                        "train_person_boxes": 1,
                        "respect_source_split": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    build(config_path)
    cache = output / ".yolo" / "train" / "labels.cache"
    cache.write_text("stale cache", encoding="utf-8")

    build(config_path)

    assert not cache.exists()


@pytest.mark.parametrize(
    "unsafe_path",
    ["../outside.txt", "/tmp/outside.txt", "raw/unmanaged.txt"],
)
def test_publish_rejects_unsafe_paths_in_generated_manifest(
    tmp_path: Path, unsafe_path: str
) -> None:
    output = tmp_path / "external_dataset"
    manifests = output / "manifests"
    manifests.mkdir(parents=True)
    generated = manifests / "generated_files.txt"
    generated.write_text(f"{unsafe_path}\n", encoding="utf-8")
    stage = tmp_path / "stage"
    (stage / "train").mkdir(parents=True)
    (stage / "train" / "replacement.txt").write_text("new", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe generated path"):
        publish_output(stage, output)

    assert generated.read_text(encoding="utf-8") == f"{unsafe_path}\n"
    assert not (output / "train" / "replacement.txt").exists()
