from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

RESOLVED_STATUSES = {"helmet_absent", "helmet_present", "head_not_visible"}
SUPPORTED_STATUSES = RESOLVED_STATUSES | {"head_status_unknown"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def indexed_by_source_id(
    items: list[dict[str, Any]], path: Path
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line_number, item in enumerate(items, start=1):
        if "source_id" not in item:
            raise ValueError(f"{path}:{line_number}: source_id is required")
        source_id = str(item["source_id"])
        if source_id in result:
            raise ValueError(f"{path}: duplicate source_id: {source_id}")
        result[source_id] = item
    return result


def frame_dimension(record: dict[str, Any], name: str) -> float | None:
    if name not in record:
        return None
    value = record[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{record['source_id']}: {name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{record['source_id']}: {name} must be positive and finite")
    return value


def validated_box(
    head: dict[str, Any], source_id: str, width: float | None, height: float | None
) -> list[float]:
    values = head.get("xyxy")
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"{source_id}: head xyxy must contain four numbers")
    invalid_type = any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    )
    if invalid_type:
        raise ValueError(f"{source_id}: head xyxy must be numeric")
    box = [float(value) for value in values]
    if not all(math.isfinite(value) for value in box):
        raise ValueError(f"{source_id}: head xyxy must be finite")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{source_id}: head xyxy must have positive area")
    if x1 < 0 or y1 < 0:
        raise ValueError(f"{source_id}: head xyxy is outside the image")
    if width is not None and x2 > width:
        raise ValueError(f"{source_id}: head xyxy exceeds image width")
    if height is not None and y2 > height:
        raise ValueError(f"{source_id}: head xyxy exceeds image height")
    return box


def verified_value(head: dict[str, Any], source_id: str) -> bool:
    value = head.get("verified", False)
    if not isinstance(value, bool):
        raise ValueError(f"{source_id}: head verified must be boolean")
    return value


def head_object(
    head: dict[str, Any],
    source_id: str = "review",
    width: float | None = None,
    height: float | None = None,
) -> dict[str, Any] | None:
    status = head.get("status")
    if status not in SUPPORTED_STATUSES:
        raise ValueError(f"{source_id}: unsupported head status: {status}")
    verified = verified_value(head, source_id)
    if status == "head_not_visible":
        if "xyxy" in head:
            raise ValueError(f"{source_id}: head_not_visible must not contain xyxy")
        return None
    box = validated_box(head, source_id, width, height)
    role = {
        "helmet_absent": "no_helmet" if verified else "unknown",
        "helmet_present": "helmet_present" if verified else "unknown",
        "head_status_unknown": "unknown",
    }[str(status)]
    result = {
        "role": role,
        "source_category": status,
        "xyxy": box,
        "annotation_method": head.get("annotation_method", "review"),
        "verified": verified,
    }
    if "person_track_id" in head:
        result["track_id"] = head["person_track_id"]
    if "person_index" in head:
        result["person_index"] = head["person_index"]
    return result


def association_mode(
    persons: list[dict[str, Any]], source_id: str
) -> tuple[str, set[str | int]]:
    if not persons:
        return "none", set()
    track_ids = [person.get("track_id") for person in persons]
    if all(track_id is not None for track_id in track_ids):
        normalized = [str(track_id) for track_id in track_ids]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{source_id}: person track IDs must be unique")
        return "track", set(normalized)
    if all(track_id is None for track_id in track_ids):
        return "index", set(range(len(persons)))
    raise ValueError(f"{source_id}: person track IDs are only partially populated")


def head_association(
    head: dict[str, Any], mode: str, expected: set[str | int], source_id: str
) -> str | int:
    if mode == "track":
        if "person_index" in head or "person_track_id" not in head:
            raise ValueError(f"{source_id}: each head requires person_track_id")
        value: str | int = str(head["person_track_id"])
    elif mode == "index":
        value = head.get("person_index")
        if (
            "person_track_id" in head
            or isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise ValueError(f"{source_id}: each head requires integer person_index")
    else:
        raise ValueError(f"{source_id}: a frame without people cannot contain heads")
    if value not in expected:
        raise ValueError(f"{source_id}: head references an unknown person: {value}")
    return value


def normalized_decision(
    head: dict[str, Any],
    association: str | int,
    width: float | None,
    height: float | None,
    source_id: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    item = head_object(head, source_id, width, height)
    status = str(head["status"])
    verified = verified_value(head, source_id)
    decision: dict[str, Any] = {"status": status, "verified": verified}
    association_name = (
        "person_track_id" if isinstance(association, str) else "person_index"
    )
    decision[association_name] = association
    if status != "head_not_visible":
        decision["xyxy"] = list(item["xyxy"]) if item is not None else None
    if "annotation_method" in head:
        decision["annotation_method"] = head["annotation_method"]
    resolved = status in RESOLVED_STATUSES and verified
    return decision, item, resolved


def explicit_boolean(item: dict[str, Any], name: str, source_id: str) -> bool | None:
    if name not in item:
        return None
    value = item[name]
    if not isinstance(value, bool):
        raise ValueError(f"{source_id}: {name} must be boolean")
    return value


def apply_reviews(manifest: Path, reviews_path: Path, output: Path) -> dict[str, int]:
    records = read_jsonl(manifest)
    records_by_id = indexed_by_source_id(records, manifest)
    reviews = indexed_by_source_id(read_jsonl(reviews_path), reviews_path)
    unknown = sorted(set(reviews) - set(records_by_id))
    if unknown:
        raise ValueError(f"Review source IDs are absent from manifest: {unknown[:5]}")
    reviewed = 0
    complete = 0
    positives = 0
    result: list[dict[str, Any]] = []
    for record in records:
        source_id = str(record["source_id"])
        review = reviews.get(source_id)
        if review is None:
            result.append(record)
            continue
        reviewed += 1
        objects = [
            item
            for item in record.get("objects", [])
            if item.get("role") not in {"no_helmet", "helmet_present", "unknown"}
        ]
        persons = [item for item in objects if item.get("role") == "person"]
        mode, expected = association_mode(persons, source_id)
        review_heads = review.get("heads", [])
        if not isinstance(review_heads, list):
            raise ValueError(f"{source_id}: heads must be a list")
        width = frame_dimension(record, "width")
        height = frame_dimension(record, "height")
        decisions: list[dict[str, Any]] = []
        heads: list[dict[str, Any]] = []
        associations: set[str | int] = set()
        all_resolved = True
        for head in review_heads:
            if not isinstance(head, dict):
                raise ValueError(f"{source_id}: every head decision must be an object")
            association = head_association(head, mode, expected, source_id)
            if association in associations:
                raise ValueError(
                    f"{source_id}: duplicate head decision for {association}"
                )
            associations.add(association)
            decision, item, resolved = normalized_decision(
                head, association, width, height, source_id
            )
            decisions.append(decision)
            all_resolved &= resolved
            if item is not None:
                heads.append(item)
                positives += item["role"] == "no_helmet"
        requested_complete = explicit_boolean(review, "complete", source_id) is True
        if requested_complete and associations != expected:
            missing = sorted(expected - associations, key=str)
            raise ValueError(
                f"{source_id}: complete review is missing people: {missing}"
            )
        is_complete = requested_complete and associations == expected and all_resolved
        record["objects"] = objects + heads
        record["no_helmet_annotation_complete"] = is_complete
        attributes = record.setdefault("attributes", {})
        attributes["head_review"] = review.get("reviewer", "unknown")
        attributes["head_review_complete"] = is_complete
        attributes["head_review_decisions"] = decisions
        person_review = explicit_boolean(review, "person_review_complete", source_id)
        if person_review is not None:
            attributes["person_review_complete"] = person_review
            record["person_annotation_complete"] = person_review
        complete += is_complete
        result.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in result),
        encoding="utf-8",
    )
    temporary.replace(output)
    return {
        "records": len(records),
        "reviewed": reviewed,
        "complete": complete,
        "no_helmet": positives,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply verified head reviews")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or args.manifest.parent / "reviewed_manifest.jsonl"
    stats = apply_reviews(args.manifest, args.reviews, output)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
