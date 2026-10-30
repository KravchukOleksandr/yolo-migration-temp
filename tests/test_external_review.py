import json
from pathlib import Path

import pytest

from yolo_factory.external_review import apply_reviews


def write_jsonl(path: Path, *items: dict) -> None:
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in items), encoding="utf-8"
    )


def untracked_record(source_id: str = "video:10") -> dict:
    return {
        "source_id": source_id,
        "width": 100,
        "height": 80,
        "objects": [{"role": "person", "xyxy": [0, 0, 20, 50]}],
        "person_annotation_complete": True,
        "no_helmet_annotation_complete": False,
    }


def run_review(tmp_path: Path, record: dict, review: dict) -> tuple[dict, dict]:
    manifest = tmp_path / "manifest.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    output = tmp_path / "reviewed_manifest.jsonl"
    write_jsonl(manifest, record)
    write_jsonl(reviews, review)
    stats = apply_reviews(manifest, reviews, output)
    return stats, json.loads(output.read_text())


def test_unverified_positive_remains_incomplete(tmp_path: Path) -> None:
    stats, record = run_review(
        tmp_path,
        untracked_record(),
        {
            "source_id": "video:10",
            "complete": True,
            "heads": [
                {
                    "status": "helmet_absent",
                    "person_index": 0,
                    "xyxy": [5, 0, 12, 8],
                    "verified": False,
                }
            ],
        },
    )

    assert not record["no_helmet_annotation_complete"]
    assert not record["attributes"]["head_review_complete"]
    assert record["objects"][1]["role"] == "unknown"
    assert stats["no_helmet"] == 0


def test_unverified_helmet_present_remains_incomplete(tmp_path: Path) -> None:
    _, record = run_review(
        tmp_path,
        untracked_record(),
        {
            "source_id": "video:10",
            "complete": True,
            "heads": [
                {
                    "status": "helmet_present",
                    "person_index": 0,
                    "xyxy": [5, 0, 12, 8],
                    "verified": False,
                }
            ],
        },
    )

    assert not record["no_helmet_annotation_complete"]
    assert record["objects"][1]["role"] == "unknown"


@pytest.mark.parametrize(
    "status, has_box",
    [
        ("helmet_absent", True),
        ("helmet_present", True),
        ("head_not_visible", False),
    ],
)
def test_resolved_status_requires_explicit_verification(
    tmp_path: Path, status: str, has_box: bool
) -> None:
    head = {"status": status, "person_index": 0}
    if has_box:
        head["xyxy"] = [5, 0, 12, 8]

    _, record = run_review(
        tmp_path,
        untracked_record(),
        {
            "source_id": "video:10",
            "complete": True,
            "heads": [head],
        },
    )

    assert not record["no_helmet_annotation_complete"]
    assert not record["attributes"]["head_review_complete"]
    assert not record["attributes"]["head_review_decisions"][0]["verified"]


def test_verified_untracked_review_uses_person_index(tmp_path: Path) -> None:
    stats, record = run_review(
        tmp_path,
        untracked_record(),
        {
            "source_id": "video:10",
            "complete": True,
            "person_review_complete": True,
            "reviewer": "manual",
            "heads": [
                {
                    "status": "helmet_absent",
                    "person_index": 0,
                    "xyxy": [5, 0, 12, 8],
                    "verified": True,
                }
            ],
        },
    )

    assert record["no_helmet_annotation_complete"]
    assert record["person_annotation_complete"]
    assert record["attributes"]["head_review_complete"]
    assert record["attributes"]["person_review_complete"]
    assert record["objects"][1]["role"] == "no_helmet"
    assert record["objects"][1]["person_index"] == 0
    assert stats["complete"] == 1
    assert stats["no_helmet"] == 1


def test_tracked_review_preserves_not_visible_decision(tmp_path: Path) -> None:
    record = {
        "source_id": "video:20",
        "width": 100,
        "height": 80,
        "objects": [
            {"role": "person", "track_id": 7, "xyxy": [0, 0, 20, 50]},
            {"role": "person", "track_id": 9, "xyxy": [30, 0, 50, 50]},
        ],
    }
    review = {
        "source_id": "video:20",
        "complete": True,
        "heads": [
            {
                "status": "helmet_present",
                "person_track_id": 7,
                "xyxy": [5, 0, 12, 8],
                "verified": True,
            },
            {
                "status": "head_not_visible",
                "person_track_id": 9,
                "verified": True,
            },
        ],
    }

    _, reviewed = run_review(tmp_path, record, review)

    assert reviewed["no_helmet_annotation_complete"]
    assert len(reviewed["objects"]) == 3
    decisions = reviewed["attributes"]["head_review_decisions"]
    assert decisions[1] == {
        "status": "head_not_visible",
        "verified": True,
        "person_track_id": "9",
    }
    assert "person_review_complete" not in reviewed["attributes"]


def test_complete_review_requires_one_decision_per_person(tmp_path: Path) -> None:
    record = {
        "source_id": "video:25",
        "objects": [
            {"role": "person", "xyxy": [0, 0, 20, 50]},
            {"role": "person", "xyxy": [30, 0, 50, 50]},
        ],
    }
    review = {
        "source_id": "video:25",
        "complete": True,
        "heads": [
            {
                "status": "helmet_present",
                "person_index": 0,
                "xyxy": [5, 0, 12, 8],
                "verified": True,
            }
        ],
    }
    manifest = tmp_path / "manifest.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    write_jsonl(manifest, record)
    write_jsonl(reviews, review)

    with pytest.raises(ValueError, match="complete review is missing people: \\[1\\]"):
        apply_reviews(manifest, reviews, tmp_path / "output.jsonl")


@pytest.mark.parametrize(
    "heads, message",
    [
        (
            [
                {
                    "status": "helmet_present",
                    "person_track_id": 999,
                    "xyxy": [1, 1, 5, 5],
                    "verified": True,
                }
            ],
            "unknown person",
        ),
        (
            [
                {
                    "status": "helmet_present",
                    "person_track_id": 7,
                    "xyxy": [1, 1, 5, 5],
                    "verified": True,
                },
                {
                    "status": "head_not_visible",
                    "person_track_id": 7,
                    "verified": True,
                },
            ],
            "duplicate head decision",
        ),
    ],
)
def test_invalid_track_associations_are_rejected(
    tmp_path: Path, heads: list[dict], message: str
) -> None:
    record = {
        "source_id": "video:30",
        "objects": [{"role": "person", "track_id": 7, "xyxy": [0, 0, 20, 50]}],
    }
    manifest = tmp_path / "manifest.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    write_jsonl(manifest, record)
    write_jsonl(reviews, {"source_id": "video:30", "complete": True, "heads": heads})

    with pytest.raises(ValueError, match=message):
        apply_reviews(manifest, reviews, tmp_path / "output.jsonl")


@pytest.mark.parametrize(
    "box, message",
    [
        ([float("nan"), 0, 10, 10], "finite"),
        ([5, 5, 5, 10], "positive area"),
        ([-1, 0, 10, 10], "outside"),
        ([0, 0, 101, 10], "exceeds image width"),
    ],
)
def test_invalid_head_boxes_are_rejected(
    tmp_path: Path, box: list[float], message: str
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    write_jsonl(manifest, untracked_record())
    write_jsonl(
        reviews,
        {
            "source_id": "video:10",
            "heads": [
                {
                    "status": "helmet_absent",
                    "person_index": 0,
                    "xyxy": box,
                    "verified": True,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match=message):
        apply_reviews(manifest, reviews, tmp_path / "output.jsonl")


def test_not_visible_rejects_fabricated_box(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    write_jsonl(manifest, untracked_record())
    write_jsonl(
        reviews,
        {
            "source_id": "video:10",
            "heads": [
                {
                    "status": "head_not_visible",
                    "person_index": 0,
                    "xyxy": [1, 1, 5, 5],
                    "verified": True,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="must not contain xyxy"):
        apply_reviews(manifest, reviews, tmp_path / "output.jsonl")


@pytest.mark.parametrize("duplicate_in", ["manifest", "reviews"])
def test_duplicate_source_ids_are_rejected(tmp_path: Path, duplicate_in: str) -> None:
    manifest = tmp_path / "manifest.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    record = untracked_record()
    review = {"source_id": "video:10", "heads": []}
    write_jsonl(manifest, record, *([record] if duplicate_in == "manifest" else []))
    write_jsonl(reviews, review, *([review] if duplicate_in == "reviews" else []))

    with pytest.raises(ValueError, match="duplicate source_id"):
        apply_reviews(manifest, reviews, tmp_path / "output.jsonl")
