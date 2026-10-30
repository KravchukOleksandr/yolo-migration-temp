from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from .common import dump_json, load_yaml
from .dataset import IMAGE_SUFFIXES

PERSON_CLASS = 0
NO_HELMET_CLASS = 2
JOINT_ROLES = {"person", "no_helmet"}
HEAD_ROLES = {"no_helmet", "helmet_present", "unknown"}
AUXILIARY_ROLES = {"helmet_indicator"}


@dataclass(frozen=True)
class SourceBox:
    source_category: str
    role: str
    xyxy: tuple[float, float, float, float]
    track_id: str | None = None
    annotation_method: str = "source"
    verified: bool = False


@dataclass
class SourceFrame:
    source: str
    source_id: str
    image: Path
    width: int
    height: int
    group_id: str
    source_split: str | None
    boxes: list[SourceBox] = field(default_factory=list)
    person_annotation_complete: bool = True
    no_helmet_annotation_complete: bool = False
    camera_id: str | None = None
    video_id: str | None = None
    frame_index: int | None = None
    sha256: str | None = None
    pixel_sha256: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def person_count(self) -> int:
        return sum(box.role == "person" for box in self.boxes)

    @property
    def no_helmet_count(self) -> int:
        return sum(box.role == "no_helmet" for box in self.boxes)

    @property
    def joint_complete(self) -> bool:
        return self.person_annotation_complete and self.no_helmet_annotation_complete


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def stable_number(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8)).getdata())
    result = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            result = (result << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return result


def infer_group(source: str, source_id: str) -> str:
    path = Path(source_id)
    stem = path.stem
    prefix = re.sub(r"(?:[_-](?:frame)?\d+)+$", "", stem, flags=re.I)
    parent = path.parent.as_posix()
    if stem.isdigit():
        key = parent if parent != "." else "flat-numeric-sequence"
    else:
        key = f"{parent}/{prefix or stem}" if parent != "." else prefix or stem
    return f"{source}:{key}"


def infer_frame_index(source_id: str) -> int | None:
    match = re.search(r"(?:^|[_-])(\d+)$", Path(source_id).stem)
    return int(match.group(1)) if match else None


def frame_group(source: str, source_id: str, config: dict[str, Any]) -> str:
    if config.get("group_mode") == "image":
        return f"{source}:{source_id}"
    if config.get("group_mode") == "camera":
        camera = infer_camera(source_id)
        if camera is None:
            raise ValueError(f"Cannot infer camera ID from {source_id}")
        return f"{source}:{camera}"
    return infer_group(source, source_id)


def infer_camera(source_id: str) -> str | None:
    match = re.match(r"\d+\.\d+\.\d+\.\d+(?=_)", Path(source_id).stem)
    return match.group(0) if match else None


def infer_split(path: Path) -> str | None:
    parts = {normalized_name(part) for part in path.parts}
    stem = normalized_name(path.stem)
    if "train" in parts or stem.startswith("train"):
        return "train"
    if parts & {"val", "valid", "validation", "test"} or stem.startswith(
        ("val", "test")
    ):
        return "val"
    return None


def image_index(root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.suffix.casefold() in IMAGE_SUFFIXES:
            result[path.name].append(path)
            result[str(path.relative_to(root))].append(path)
    return result


def find_image(index: dict[str, list[Path]], name: str) -> Path:
    matches = index.get(name) or index.get(Path(name).name) or []
    unique = sorted(set(matches))
    if not unique:
        raise FileNotFoundError(f"Image referenced by annotations is missing: {name}")
    if len(unique) > 1:
        exact = [path for path in unique if path.as_posix().endswith(name)]
        if len(exact) == 1:
            return exact[0]
        raise ValueError(f"Image name is ambiguous: {name}")
    return unique[0]


def category_roles(config: dict[str, Any]) -> dict[str, str]:
    return {
        normalized_name(name): role
        for name, role in config.get("category_roles", {}).items()
    }


def coco_frames(name: str, config: dict[str, Any], root: Path) -> list[SourceFrame]:
    index = image_index(root / config.get("image_root", "."))
    roles = category_roles(config)
    frames: list[SourceFrame] = []
    patterns = config.get("annotations", ["**/*.json"])
    annotation_paths = sorted(
        {path for pattern in patterns for path in root.glob(pattern)}
    )
    if not annotation_paths:
        raise FileNotFoundError(f"{name}: no COCO annotation files under {root}")
    seen: set[tuple[Path, int]] = set()
    for annotation_path in annotation_paths:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not {"images", "annotations", "categories"} <= payload.keys():
            continue
        categories = {
            int(item["id"]): normalized_name(str(item["name"]))
            for item in payload["categories"]
        }
        by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload["annotations"]:
            by_image[int(annotation["image_id"])].append(annotation)
        split = infer_split(annotation_path)
        for item in payload["images"]:
            image_id = int(item["id"])
            key = (Path(item["file_name"]), image_id)
            if key in seen:
                continue
            seen.add(key)
            try:
                image_path = find_image(index, str(item["file_name"]))
            except FileNotFoundError as error:
                if not config.get("allow_missing_images", False):
                    raise
                config.setdefault("_load_warnings", []).append(str(error))
                continue
            boxes: list[SourceBox] = []
            complete = bool(config.get("no_helmet_complete", False))
            for annotation in by_image.get(image_id, []):
                category = categories[int(annotation["category_id"])]
                role = roles.get(category, "ignore")
                if role == "unknown":
                    complete = False
                if role not in JOINT_ROLES | HEAD_ROLES | AUXILIARY_ROLES:
                    continue
                x, y, width, height = map(float, annotation["bbox"])
                boxes.append(
                    SourceBox(
                        source_category=category,
                        role=role,
                        xyxy=(x, y, x + width, y + height),
                        track_id=str(annotation.get("track_id"))
                        if annotation.get("track_id") is not None
                        else None,
                    )
                )
            source_id = str(item["file_name"])
            frames.append(
                SourceFrame(
                    source=name,
                    source_id=source_id,
                    image=image_path,
                    width=int(item["width"]),
                    height=int(item["height"]),
                    group_id=frame_group(name, source_id, config),
                    source_split=split,
                    boxes=boxes,
                    frame_index=infer_frame_index(source_id),
                    camera_id=infer_camera(source_id),
                    no_helmet_annotation_complete=complete,
                )
            )
    return frames


def voc_frames(name: str, config: dict[str, Any], root: Path) -> list[SourceFrame]:
    index = image_index(root / config.get("image_root", "."))
    roles = category_roles(config)
    patterns = config.get("annotations", ["**/*.xml"])
    paths = sorted({path for pattern in patterns for path in root.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"{name}: no Pascal VOC annotations under {root}")
    frames: list[SourceFrame] = []
    for annotation_path in paths:
        tree = ET.parse(annotation_path)
        root_node = tree.getroot()
        filename = root_node.findtext("filename") or f"{annotation_path.stem}.jpg"
        try:
            image_path = find_image(index, filename)
        except FileNotFoundError as error:
            if not config.get("allow_missing_images", False):
                raise
            config.setdefault("_load_warnings", []).append(str(error))
            continue
        size = root_node.find("size")
        if size is None:
            with Image.open(image_path) as image:
                width, height = image.size
        else:
            width = int(size.findtext("width", "0"))
            height = int(size.findtext("height", "0"))
        boxes: list[SourceBox] = []
        complete = bool(config.get("no_helmet_complete", False))
        for node in root_node.findall("object"):
            category = normalized_name(node.findtext("name", ""))
            role = roles.get(category, "ignore")
            if role == "unknown":
                complete = False
            if role not in JOINT_ROLES | HEAD_ROLES | AUXILIARY_ROLES:
                continue
            bounds = node.find("bndbox")
            if bounds is None:
                continue
            xyxy = tuple(
                float(bounds.findtext(field, "0"))
                for field in ("xmin", "ymin", "xmax", "ymax")
            )
            boxes.append(SourceBox(category, role, xyxy))
        source_id = str(image_path.relative_to(root))
        frames.append(
            SourceFrame(
                source=name,
                source_id=source_id,
                image=image_path,
                width=width,
                height=height,
                group_id=frame_group(name, source_id, config),
                source_split=infer_split(annotation_path),
                boxes=boxes,
                frame_index=infer_frame_index(source_id),
                no_helmet_annotation_complete=complete,
            )
        )
    return frames


def manifest_frames(name: str, config: dict[str, Any], root: Path) -> list[SourceFrame]:
    reviewed = root / "reviewed_manifest.jsonl"
    manifest = (
        reviewed
        if reviewed.is_file()
        else root / config.get("manifest", "manifest.jsonl")
    )
    if not manifest.is_file():
        raise FileNotFoundError(
            f"{name}: expected normalized manifest at {manifest}. "
            "See README.md for the schema."
        )
    frames: list[SourceFrame] = []
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        image_path = (root / item["image"]).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError(f"{manifest}:{line_number}: image is outside source root")
        if not image_path.is_file():
            raise FileNotFoundError(f"{manifest}:{line_number}: {image_path}")
        group_id = item.get("group_id")
        if not group_id:
            raise ValueError(f"{manifest}:{line_number}: group_id is required")
        boxes = [
            SourceBox(
                source_category=str(box.get("source_category", box["role"])),
                role=str(box["role"]),
                xyxy=tuple(map(float, box["xyxy"])),
                track_id=str(box["track_id"])
                if box.get("track_id") is not None
                else None,
                annotation_method=str(box.get("annotation_method", "source")),
                verified=boolean_field(box, "verified", False),
            )
            for box in item.get("objects", [])
        ]
        attributes = dict(item.get("attributes", {}))
        for field_name in ("head_review_complete", "person_review_complete"):
            if field_name in attributes:
                boolean_field(attributes, field_name, False)
        frames.append(
            SourceFrame(
                source=name,
                source_id=str(item.get("source_id", item["image"])),
                image=image_path,
                width=int(item["width"]),
                height=int(item["height"]),
                group_id=str(group_id),
                source_split=item.get("split"),
                boxes=boxes,
                person_annotation_complete=boolean_field(
                    item, "person_annotation_complete", True
                ),
                no_helmet_annotation_complete=boolean_field(
                    item,
                    "no_helmet_annotation_complete",
                    config.get("no_helmet_complete", False),
                ),
                camera_id=item.get("camera_id"),
                video_id=item.get("video_id"),
                frame_index=item.get("frame_index"),
                attributes=attributes,
            )
        )
    return frames


def boolean_field(item: dict[str, Any], key: str, default: bool) -> bool:
    value = item.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def load_source(name: str, config: dict[str, Any]) -> list[SourceFrame]:
    root = Path(config["root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{name}: source directory is missing: {root}")
    loaders = {"coco": coco_frames, "voc": voc_frames, "manifest": manifest_frames}
    kind = config["format"]
    if kind == "auto":
        manifest = root / config.get("manifest", "manifest.jsonl")
        if manifest.is_file():
            kind = "manifest"
        elif any(root.rglob("*.xml")):
            kind = "voc"
        else:
            kind = "coco"
    if kind not in loaders:
        raise ValueError(f"{name}: unsupported source format: {kind}")
    frames = loaders[kind](name, config, root)
    if kind != "manifest" and (root / "reviewed_manifest.jsonl").is_file():
        reviewed = manifest_frames(name, config, root)
        overrides = {frame.source_id: frame for frame in reviewed}
        if len(overrides) != len(reviewed):
            raise ValueError("Duplicate source ID in reviewed manifest")
        unexpected = overrides.keys() - {frame.source_id for frame in frames}
        if unexpected:
            raise ValueError(f"Unknown reviewed source IDs: {sorted(unexpected)[:5]}")
        frames = [overrides.get(frame.source_id, frame) for frame in frames]
    seen: set[str] = set()
    for frame in frames:
        if frame.source_id in seen:
            raise ValueError(f"Duplicate source ID: {frame.source_id}")
        seen.add(frame.source_id)
        if kind != "manifest":
            frame.group_id = frame_group(name, frame.source_id, config)
        if any(box.role == "unknown" for box in frame.boxes):
            frame.no_helmet_annotation_complete = False
        if frame.camera_id is None:
            frame.camera_id = infer_camera(frame.source_id)
        if frame.video_id is None and config.get("group_mode") != "image":
            frame.attributes.setdefault(
                "inferred_sequence_id", infer_group(name, frame.source_id)
            )
    return [frame for frame in frames if frame.person_count]


def assign_splits(frames: list[SourceFrame], val_fraction: float, seed: int) -> None:
    if not 0 < val_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    observed: dict[str, set[str]] = defaultdict(set)
    for frame in frames:
        if frame.source_split:
            if frame.source_split not in {"train", "val"}:
                raise ValueError(f"Unsupported split: {frame.source_split}")
            observed[frame.group_id].add(frame.source_split)
    if any(len(splits) > 1 for splits in observed.values()):
        raise ValueError("A source group occurs in both train and val")
    fixed_groups = {frame.group_id for frame in frames if frame.source_split}
    assigned = {
        frame.group_id: frame.source_split for frame in frames if frame.source_split
    }
    for frame in frames:
        if frame.source_split:
            continue
        existing = assigned.get(frame.group_id)
        if existing:
            frame.source_split = existing
            continue
        threshold = int(val_fraction * 10_000)
        value = stable_number(frame.group_id, seed) % 10_000
        split = "val" if value < threshold else "train"
        assigned[frame.group_id] = split
        frame.source_split = split
    groups: dict[str, list[SourceFrame]] = defaultdict(list)
    for frame in frames:
        groups[frame.group_id].append(frame)
    mutable = [group_id for group_id in groups if group_id not in fixed_groups]
    if len(groups) == 1 and mutable:
        for frame in groups[mutable[0]]:
            frame.source_split = "train"
        return
    split_groups = {
        split: [
            group_id
            for group_id, group in groups.items()
            if group[0].source_split == split
        ]
        for split in ("train", "val")
    }
    if len(groups) > 1 and not split_groups["val"] and mutable:
        group_id = min(mutable, key=lambda value: stable_number(value, seed))
        for frame in groups[group_id]:
            frame.source_split = "val"
    elif len(groups) > 1 and not split_groups["train"] and mutable:
        group_id = max(mutable, key=lambda value: stable_number(value, seed))
        for frame in groups[group_id]:
            frame.source_split = "train"


def filter_temporal_neighbors(
    frames: list[SourceFrame], minimum_gap: int
) -> list[SourceFrame]:
    if minimum_gap <= 0:
        return frames
    grouped: dict[str, list[SourceFrame]] = defaultdict(list)
    passthrough: list[SourceFrame] = []
    for frame in frames:
        if frame.frame_index is None:
            passthrough.append(frame)
        else:
            grouped[sequence_id(frame)].append(frame)
    kept = list(passthrough)
    for group in grouped.values():
        previous: int | None = None
        for frame in sorted(group, key=lambda item: int(item.frame_index or 0)):
            current = int(frame.frame_index or 0)
            if previous is None or current - previous >= minimum_gap:
                kept.append(frame)
                previous = current
    return kept


def filter_near_neighbors(
    frames: list[SourceFrame], maximum_distance: int
) -> tuple[list[SourceFrame], int]:
    if maximum_distance < 0:
        return frames, 0
    grouped: dict[str, list[SourceFrame]] = defaultdict(list)
    for frame in frames:
        grouped[sequence_id(frame)].append(frame)
    kept: list[SourceFrame] = []
    removed = 0
    for group in grouped.values():
        previous_hash: int | None = None
        previous_frame: SourceFrame | None = None
        ordered = sorted(
            group,
            key=lambda item: (
                item.frame_index is None,
                item.frame_index if item.frame_index is not None else 0,
                item.source_id,
            ),
        )
        for frame in ordered:
            current_hash = difference_hash(frame.image)
            if (
                previous_hash is not None
                and (current_hash ^ previous_hash).bit_count() <= maximum_distance
                and previous_frame is not None
                and same_objects(previous_frame, frame)
            ):
                removed += 1
                continue
            kept.append(frame)
            previous_hash = current_hash
            previous_frame = frame
    return kept, removed


def existing_hashes(paths: list[str]) -> tuple[dict[str, str], list[str]]:
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for value in paths:
        root = Path(value).resolve()
        if not root.is_dir():
            missing.append(value)
            continue
        for path in root.rglob("*"):
            if path.suffix.casefold() not in IMAGE_SUFFIXES:
                continue
            hashes.setdefault(file_sha256(path), str(path))
    return hashes, missing


def sequence_id(frame: SourceFrame) -> str:
    return (
        frame.video_id or frame.attributes.get("inferred_sequence_id") or frame.group_id
    )


def filter_exact_duplicates(
    frames: list[SourceFrame], hashes: dict[str, str]
) -> tuple[list[SourceFrame], list[dict[str, str]]]:
    kept: list[SourceFrame] = []
    removed: list[dict[str, str]] = []
    for frame in frames:
        frame.sha256 = file_sha256(frame.image)
        pixel_key = f"pixels:{frame.pixel_sha256}" if frame.pixel_sha256 else None
        duplicate = hashes.get(frame.sha256) or hashes.get(pixel_key)
        if duplicate:
            removed.append(
                {
                    "source": frame.source,
                    "source_id": frame.source_id,
                    "reason": f"exact_duplicate_of:{duplicate}",
                }
            )
            continue
        hashes[frame.sha256] = f"{frame.source}:{frame.source_id}"
        if pixel_key:
            hashes[pixel_key] = f"{frame.source}:{frame.source_id}"
        kept.append(frame)
    return kept, removed


def candidate_record(frame: SourceFrame) -> dict[str, Any]:
    return {
        "source": frame.source,
        "source_id": frame.source_id,
        "image": str(frame.image),
        "width": frame.width,
        "height": frame.height,
        "group_id": frame.group_id,
        "camera_id": frame.camera_id,
        "video_id": frame.video_id,
        "frame_index": frame.frame_index,
        "split": frame.source_split,
        "person_annotation_complete": frame.person_annotation_complete,
        "no_helmet_annotation_complete": frame.no_helmet_annotation_complete,
        "attributes": frame.attributes,
        "objects": [
            {
                "source_category": box.source_category,
                "role": box.role,
                "xyxy": list(box.xyxy),
                "track_id": box.track_id,
                "annotation_method": box.annotation_method,
                "verified": box.verified,
            }
            for box in frame.boxes
        ],
    }


def select_for_quota(
    frames: list[SourceFrame], target: int, seed: int
) -> tuple[list[SourceFrame], int]:
    buckets: dict[str, list[SourceFrame]] = defaultdict(list)
    for frame in frames:
        buckets[frame.group_id].append(frame)
    for group in buckets.values():
        group.sort(
            key=lambda frame: stable_number(f"{frame.source}:{frame.source_id}", seed)
        )
    group_ids = sorted(buckets, key=lambda value: stable_number(value, seed))
    ordered: list[SourceFrame] = []
    while any(buckets.values()):
        for group_id in group_ids:
            if buckets[group_id]:
                ordered.append(buckets[group_id].pop())
    previous = [-2] * (target + 1)
    previous[0] = -1
    chosen_index = [-1] * (target + 1)
    for index, frame in enumerate(ordered):
        count = frame.person_count
        if count <= 0 or count > target:
            continue
        for total in range(target, count - 1, -1):
            if previous[total] == -2 and previous[total - count] != -2:
                previous[total] = total - count
                chosen_index[total] = index
    achieved = max(index for index, value in enumerate(previous) if value != -2)
    selected: list[SourceFrame] = []
    cursor = achieved
    while cursor:
        index = chosen_index[cursor]
        selected.append(ordered[index])
        cursor = previous[cursor]
    return sorted(selected, key=lambda frame: frame.source_id), achieved


def clip_box(box: SourceBox, frame: SourceFrame) -> SourceBox | None:
    if (
        len(box.xyxy) != 4
        or not all(math.isfinite(value) for value in box.xyxy)
        or frame.width <= 0
        or frame.height <= 0
    ):
        return None
    x1, y1, x2, y2 = box.xyxy
    x1 = min(max(x1, 0.0), float(frame.width))
    y1 = min(max(y1, 0.0), float(frame.height))
    x2 = min(max(x2, 0.0), float(frame.width))
    y2 = min(max(y2, 0.0), float(frame.height))
    if x2 <= x1 or y2 <= y1:
        return None
    return SourceBox(
        source_category=box.source_category,
        role=box.role,
        xyxy=(x1, y1, x2, y2),
        track_id=box.track_id,
        annotation_method=box.annotation_method,
        verified=box.verified,
    )


def target_class(role: str) -> int | None:
    return {"person": PERSON_CLASS, "no_helmet": NO_HELMET_CLASS}.get(role)


def yolo_line(box: SourceBox, frame: SourceFrame) -> str | None:
    class_id = target_class(box.role)
    clipped = clip_box(box, frame)
    if class_id is None or clipped is None:
        return None
    x1, y1, x2, y2 = clipped.xyxy
    x = (x1 + x2) / (2 * frame.width)
    y = (y1 + y2) / (2 * frame.height)
    width = (x2 - x1) / frame.width
    height = (y2 - y1) / frame.height
    return f"{class_id} {x:.8f} {y:.8f} {width:.8f} {height:.8f}"


def intersection_over_union(first: SourceBox, second: SourceBox) -> float:
    ax1, ay1, ax2, ay2 = first.xyxy
    bx1, by1, bx2, by2 = second.xyxy
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def same_objects(first: SourceFrame, second: SourceFrame) -> bool:
    if (first.width, first.height) != (second.width, second.height):
        return False
    remaining = list(second.boxes)
    if len(first.boxes) != len(remaining):
        return False
    for box in first.boxes:
        matches = [
            index
            for index, other in enumerate(remaining)
            if box.role == other.role and intersection_over_union(box, other) >= 0.95
        ]
        if not matches:
            return False
        remaining.pop(matches[0])
    return True


def frame_errors(frame: SourceFrame) -> list[str]:
    errors: list[str] = []
    try:
        with Image.open(frame.image) as image:
            actual_size = image.size
            image.load()
            frame.pixel_sha256 = hashlib.sha256(
                str(image.size).encode() + image.convert("RGB").tobytes()
            ).hexdigest()
    except (OSError, ValueError) as error:
        return [f"unreadable_image:{error}"]
    if actual_size != (frame.width, frame.height):
        errors.append(
            f"image_size_mismatch:{actual_size[0]}x{actual_size[1]}"
            f"!={frame.width}x{frame.height}"
        )
    if frame.width <= 0 or frame.height <= 0:
        errors.append("invalid_dimensions")
    for index, box in enumerate(frame.boxes):
        if box.role not in JOINT_ROLES | HEAD_ROLES | AUXILIARY_ROLES:
            errors.append(f"invalid_role:{index}:{box.role}")
        if clip_box(box, frame) is None:
            errors.append(f"invalid_box:{index}:{box.role}")
    for first_index, first in enumerate(frame.boxes):
        if clip_box(first, frame) is None:
            continue
        for second_index in range(first_index + 1, len(frame.boxes)):
            second = frame.boxes[second_index]
            if (
                first.role == second.role
                and clip_box(second, frame) is not None
                and intersection_over_union(first, second) >= 0.8
            ):
                errors.append(
                    f"duplicate_box:{first_index}:{second_index}:{first.role}"
                )
    return errors


def head_coverage_complete(frame: SourceFrame, evidence_roles: set[str]) -> bool:
    people = [box for box in frame.boxes if box.role == "person"]
    heads = [box for box in frame.boxes if box.role in evidence_roles]
    owners: dict[int, int] = {}

    def assign(person_index: int, visited: set[int]) -> bool:
        person = people[person_index]
        px1, py1, px2, py2 = person.xyxy
        for index, head in enumerate(heads):
            if index in visited:
                continue
            x1, y1, x2, y2 = head.xyxy
            area = (x2 - x1) * (y2 - y1)
            overlap = max(0, min(px2, x2) - max(px1, x1)) * max(
                0, min(py2, y2) - max(py1, y1)
            )
            if area <= 0 or overlap / area < 0.5:
                continue
            visited.add(index)
            if index not in owners or assign(owners[index], visited):
                owners[index] = person_index
                return True
        return False

    return all(assign(index, set()) for index in range(len(people)))


def has_orphan_head(frame: SourceFrame) -> bool:
    people = [box for box in frame.boxes if box.role == "person"]
    for head in frame.boxes:
        if head.role not in {"no_helmet", "helmet_present"}:
            continue
        x1, y1, x2, y2 = head.xyxy
        area = (x2 - x1) * (y2 - y1)
        covered = False
        for person in people:
            px1, py1, px2, py2 = person.xyxy
            overlap = max(0, min(px2, x2) - max(px1, x1)) * max(
                0, min(py2, y2) - max(py1, y1)
            )
            if area > 0 and overlap / area >= 0.5:
                covered = True
                break
        if not covered:
            return True
    return False


def safe_stem(frame: SourceFrame) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(frame.source_id).stem)
    suffix = hashlib.sha1(frame.source_id.encode()).hexdigest()[:8]
    return f"{stem}-{suffix}"


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def object_metadata(
    frame: SourceFrame,
    box: SourceBox,
    image_id: str,
    input_size: list[int],
    training_imgsz: int = 1280,
) -> dict[str, Any]:
    x1, y1, x2, y2 = box.xyxy
    input_width, input_height = input_size
    gain = min(input_width / frame.width, input_height / frame.height)
    object_key = json.dumps(
        [image_id, box.source_category, box.role, box.xyxy, box.track_id]
    )
    clipped = clip_box(box, frame)
    return {
        "object_id": hashlib.sha1(object_key.encode()).hexdigest()[:16],
        "image_id": image_id,
        "source_category": box.source_category,
        "target_class": target_class(box.role),
        "role": box.role,
        "head_status": {
            "no_helmet": "helmet_absent",
            "helmet_present": "helmet_present",
            "unknown": "head_status_unknown",
        }.get(box.role),
        "source_xyxy": list(box.xyxy),
        "target_xyxy": (
            list(clipped.xyxy)
            if target_class(box.role) is not None and clipped is not None
            else None
        ),
        "source_size": [x2 - x1, y2 - y1],
        "deployment_letterbox_size": [(x2 - x1) * gain, (y2 - y1) * gain],
        "rect_baseline_size": [
            (x2 - x1) * training_imgsz / max(frame.width, frame.height),
            (y2 - y1) * training_imgsz / max(frame.width, frame.height),
        ],
        "track_id": box.track_id,
        "annotation_method": box.annotation_method,
        "verified": box.verified,
    }


def write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in items),
        encoding="utf-8",
    )


def numeric_summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return round(ordered[index], 3)

    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "p10": percentile(0.1),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "max": round(ordered[-1], 3),
        "mean": round(statistics.fmean(ordered), 3),
    }


def object_statistics(items: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in sorted({str(item["role"]) for item in items}):
        role_items = [item for item in items if item["role"] == role]
        source_width = [float(item["source_size"][0]) for item in role_items]
        source_height = [float(item["source_size"][1]) for item in role_items]
        result[role] = {
            "source_width": numeric_summary(source_width),
            "source_height": numeric_summary(source_height),
        }
        for kind in ("deployment_letterbox", "rect_baseline"):
            widths = [float(item[f"{kind}_size"][0]) for item in role_items]
            heights = [float(item[f"{kind}_size"][1]) for item in role_items]
            result[role][f"{kind}_width"] = numeric_summary(widths)
            result[role][f"{kind}_height"] = numeric_summary(heights)
            if role in HEAD_ROLES:
                result[role][f"{kind}_height_le_12"] = sum(
                    value <= 12 for value in heights
                )
    return result


def draw_preview(frame: SourceFrame, destination: Path) -> None:
    with Image.open(frame.image).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        colors = {
            "person": "#00ff66",
            "no_helmet": "#ff3030",
            "helmet_present": "#2080ff",
            "helmet_indicator": "#00d8ff",
            "unknown": "#ffd000",
        }
        for box in frame.boxes:
            if box.role not in colors:
                continue
            draw.rectangle(box.xyxy, outline=colors[box.role], width=3)
        image.thumbnail((480, 320))
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, quality=88)


def split_integrity(items: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in sorted({item["source"] for item in items}):
        source_items = [item for item in items if item["source"] == source]
        stats: dict[str, Any] = {}
        for key in ("group_id", "video_id", "camera_id", "sha256", "pixel_sha256"):
            values = {
                split: {
                    item[key]
                    for item in source_items
                    if item["split"] == split and item.get(key)
                }
                for split in ("train", "val")
            }
            overlap = values["train"] & values["val"]
            stats[key] = {
                "train": len(values["train"]),
                "val": len(values["val"]),
                "overlap": len(overlap),
            }
            if overlap and key != "camera_id":
                raise ValueError(f"Cross-split overlap for {source}: {key}")
        result[source] = stats
    return result


MANAGED_OUTPUTS = {
    "train",
    "val",
    ".yolo",
    "manifests",
    "previews",
    "provenance_and_licenses",
    "stats.json",
    "class_mapping.json",
    "build-config.yaml",
    "qa_report.md",
    "README.md",
}


def publish_output(stage: Path, output: Path) -> None:
    generated = output / "manifests" / "generated_files.txt"
    known = {"manifests/generated_files.txt"}
    if generated.is_file():
        known.update(line for line in generated.read_text().splitlines() if line)
    for value in known:
        path = Path(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.parts[0] not in MANAGED_OUTPUTS
        ):
            raise ValueError(f"Unsafe generated path: {value}")
    for name in MANAGED_OUTPUTS:
        target = output / name
        if target.is_symlink():
            raise ValueError(f"Managed output must not be a symlink: {target}")
        paths = target.rglob("*") if target.is_dir() else [target]
        for path in paths:
            if not (path.is_file() or path.is_symlink()):
                continue
            relative = path.relative_to(output).as_posix()
            is_cache = path.suffix in {".cache", ".npy"} and name in {
                "train",
                "val",
                ".yolo",
            }
            if relative not in known and not is_cache:
                raise ValueError(f"Refusing to overwrite untracked export file: {path}")
    with tempfile.TemporaryDirectory(prefix=".previous-", dir=output) as previous:
        backup = Path(previous)
        moved: list[str] = []
        installed: list[str] = []
        try:
            for name in sorted(MANAGED_OUTPUTS):
                target = output / name
                if target.exists():
                    target.replace(backup / name)
                    moved.append(name)
                replacement = stage / name
                if replacement.exists():
                    replacement.replace(target)
                    installed.append(name)
        except BaseException:
            for name in reversed(installed):
                (output / name).replace(stage / name)
            for name in reversed(moved):
                (backup / name).replace(output / name)
            raise


def build(config_path: Path, allow_partial: bool = False) -> dict[str, Any]:
    import fcntl

    config = load_yaml(config_path)
    output = Path(config.get("output", "external_dataset")).resolve()
    if output == Path.cwd() or output in Path.cwd().parents:
        raise ValueError("Output must be a dedicated dataset directory")
    if not 0 < float(config.get("validation_fraction", 0.2)) < 1:
        raise ValueError("validation_fraction must be between zero and one")
    for name, source in config["sources"].items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Unsafe source name: {name}")
        if int(source["train_person_boxes"]) <= 0:
            raise ValueError(f"Invalid train quota for {name}")
        root = Path(source["root"]).resolve()
        if (
            root == output
            or root in output.parents
            or any(
                (output / key) == root or (output / key) in root.parents
                for key in MANAGED_OUTPUTS
            )
        ):
            raise ValueError(f"Source root overlaps managed export: {root}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".build.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another dataset build is running") from error
        with tempfile.TemporaryDirectory(prefix=".build-", dir=output) as temporary:
            stage = Path(temporary)
            stats = build_staged(config, stage)
            previous = output / "stats.json"
            if previous.is_file():
                old_sources = load_yaml(previous).get("sources", {})
                lost = [
                    name
                    for name in stats["blocked_sources"]
                    if old_sources.get(name, {}).get("train_person", 0)
                ]
                if lost:
                    raise ValueError(
                        f"Previously available sources are now blocked: {lost}. "
                        "Previous export preserved."
                    )
            if stats["blocked_sources"] and not allow_partial:
                names = ", ".join(sorted(stats["blocked_sources"]))
                raise ValueError(
                    f"Sources unavailable: {names}; "
                    "use --allow-partial to publish available data"
                )
            publish_output(stage, output)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return stats


def build_staged(config: dict[str, Any], output: Path) -> dict[str, Any]:
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (output / ".yolo" / split / kind).mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    val_fraction = float(config.get("validation_fraction", 0.2))
    input_size = list(
        config.get("deployment_canvas", config.get("input_size", [1280, 736]))
    )
    training_imgsz = int(config.get("training_imgsz", 1280))
    if (
        len(input_size) != 2
        or any(not isinstance(size, int) or size <= 0 for size in input_size)
        or training_imgsz <= 0
    ):
        raise ValueError("Image sizes must be positive integers")
    selected: list[tuple[str, SourceFrame]] = []
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {}
    blocked: dict[str, str] = {}
    hashes, missing_dedup_roots = existing_hashes(
        list(config.get("deduplicate_against", []))
    )
    reference_hash_count = len(hashes)
    duplicate_rows: list[dict[str, str]] = []
    for offset, (name, source_config) in enumerate(config["sources"].items()):
        print(f"[{name}] Loading source annotations", flush=True)
        try:
            frames = load_source(name, source_config)
        except (FileNotFoundError, ValueError, ET.ParseError) as error:
            blocked[name] = str(error)
            source_stats[name] = {"status": "blocked", "reason": str(error)}
            continue
        candidate_count = len(frames)
        eligible: list[SourceFrame] = []
        rejected: Counter[str] = Counter()
        evidence_roles = set(source_config.get("head_evidence_roles", []))
        manual_exclusions = source_config.get("exclude_images", {})
        print(
            f"[{name}] Checking {candidate_count} candidate images "
            "before quota selection",
            flush=True,
        )
        for index, frame in enumerate(frames):
            errors = frame_errors(frame)
            if not errors and evidence_roles and has_orphan_head(frame):
                errors.append("head_without_person_annotation")
            if (
                not errors
                and evidence_roles
                and not frame.attributes.get("head_review_complete", False)
            ):
                if not head_coverage_complete(frame, evidence_roles):
                    errors.append("unresolved_person_head_status")
            if (
                source_config.get("selection_requires_joint_complete", False)
                and not frame.joint_complete
            ):
                errors.append("incomplete_annotations")
            if reason := manual_exclusions.get(frame.image.name):
                errors.append(f"manual_exclusion:{reason}")
            record = candidate_record(frame)
            record["qa_errors"] = errors
            candidates.append(record)
            if errors:
                excluded.append({**record, "reason": ";".join(errors)})
                rejected.update(error.split(":", 1)[0] for error in errors)
            else:
                eligible.append(frame)
            if (index + 1) % 1000 == 0:
                print(
                    f"[{name}] Checked {index + 1}/{candidate_count}; "
                    f"excluded {index + 1 - len(eligible)}",
                    flush=True,
                )
        frames = eligible
        if not source_config.get("respect_source_split", False):
            for frame in frames:
                if frame.source_split:
                    frame.attributes.setdefault("original_split", frame.source_split)
                frame.source_split = None
        qa_eligible_count = len(frames)
        frames = filter_temporal_neighbors(
            frames, int(source_config.get("min_frame_gap", 0))
        )
        temporal_removed = qa_eligible_count - len(frames)
        frames, near_removed = filter_near_neighbors(
            frames, int(config.get("near_duplicate_hamming", -1))
        )
        frames, source_duplicates = filter_exact_duplicates(frames, hashes)
        duplicate_rows.extend(source_duplicates)
        deduplicated_count = len(frames)
        complete_candidate_count = sum(frame.joint_complete for frame in frames)
        if source_config.get("selection_requires_joint_complete", False):
            frames = [frame for frame in frames if frame.joint_complete]
        assign_splits(frames, val_fraction, seed + offset)
        train_target = int(source_config["train_person_boxes"])
        val_target = round(train_target * val_fraction / (1 - val_fraction))
        train, train_count = select_for_quota(
            [frame for frame in frames if frame.source_split == "train"],
            train_target,
            seed + offset,
        )
        val, val_count = select_for_quota(
            [frame for frame in frames if frame.source_split == "val"],
            val_target,
            seed + 100 + offset,
        )
        selected.extend(("train", frame) for frame in train)
        selected.extend(("val", frame) for frame in val)
        source_stats[name] = {
            "status": "selected",
            "candidate_images": candidate_count,
            "qa_excluded_images": candidate_count - qa_eligible_count,
            "exclusion_reasons": dict(rejected),
            "eligible_images_after_dedup": deduplicated_count,
            "eligible_images_for_selection": len(frames),
            "joint_complete_candidates": complete_candidate_count,
            "temporal_neighbors_removed": temporal_removed,
            "near_neighbors_removed": near_removed,
            "exact_duplicates_removed": len(source_duplicates),
            "train_target_person": train_target,
            "train_person": train_count,
            "train_deficit_person": train_target - train_count,
            "val_target_person": val_target,
            "val_person": val_count,
            "val_deficit_person": val_target - val_count,
            "group_mode": source_config.get("group_mode", "manifest_or_sequence"),
            "license": source_config.get("license", "unknown"),
            "version": source_config.get("version", "unknown"),
            "load_warnings": list(source_config.get("_load_warnings", [])),
        }
        print(
            f"[{name}] Selected train={train_count}/{train_target}, "
            f"val={val_count}/{val_target} person boxes",
            flush=True,
        )

    manifest_items: list[dict[str, Any]] = []
    object_items: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = [
        {key: item[key] for key in ("source", "source_id", "image", "reason")}
        for item in excluded
    ]
    generated: list[Path] = []
    split_lists: dict[str, list[str]] = defaultdict(list)
    joint_lists: dict[str, list[str]] = defaultdict(list)
    for split, frame in selected:
        image_hash = frame.sha256 or file_sha256(frame.image)
        frame_ready = frame.joint_complete
        stem = safe_stem(frame)
        suffix = frame.image.suffix.casefold()
        flat_dir = output / split / frame.source
        image_target = flat_dir / f"{stem}{suffix}"
        label_target = flat_dir / f"{stem}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frame.image, image_target)
        lines = [line for box in frame.boxes if (line := yolo_line(box, frame))]
        label_target.write_text("\n".join(lines) + ("\n" if lines else ""))
        generated.extend((image_target, label_target))
        image_id = f"{frame.source}:{stem}"
        relative_image = str(image_target.relative_to(output))
        split_lists[split].append(relative_image)
        item = {
            "image_id": image_id,
            "source": frame.source,
            "source_version": config["sources"][frame.source].get("version", "unknown"),
            "usage_terms": config["sources"][frame.source].get("license", "unknown"),
            "source_id": frame.source_id,
            "split": split,
            "path": relative_image,
            "group_id": frame.group_id,
            "camera_id": frame.camera_id,
            "video_id": frame.video_id,
            "frame_index": frame.frame_index,
            "width": frame.width,
            "height": frame.height,
            "sha256": image_hash,
            "pixel_sha256": frame.pixel_sha256,
            "person_count": frame.person_count,
            "no_helmet_count": frame.no_helmet_count,
            "person_annotation_complete": frame.person_annotation_complete,
            "no_helmet_annotation_complete": frame.no_helmet_annotation_complete,
            "joint_ready": frame_ready,
            "assurance": (
                "individually_reviewed"
                if frame.attributes.get("head_review_complete")
                and frame.attributes.get("person_review_complete")
                else "source_annotations_automated_qa"
            ),
            "qa_errors": [],
            "attributes": frame.attributes,
        }
        manifest_items.append(item)
        object_items.extend(
            object_metadata(frame, box, image_id, input_size, training_imgsz)
            for box in frame.boxes
        )
        if frame_ready:
            yolo_image = (
                output / ".yolo" / split / "images" / frame.source / image_target.name
            )
            yolo_label = (
                output / ".yolo" / split / "labels" / frame.source / label_target.name
            )
            link_file(image_target, yolo_image)
            link_file(label_target, yolo_label)
            generated.extend((yolo_image, yolo_label))
            joint_lists[split].append(relative_image)
        else:
            unknown = sum(box.role == "unknown" for box in frame.boxes)
            pending.append(
                {
                    "source": frame.source,
                    "source_id": frame.source_id,
                    "split": split,
                    "person_count": frame.person_count,
                    "unknown_heads": unknown,
                    "reason": "incomplete_annotations",
                }
            )

    manifests = output / "manifests"
    write_jsonl(manifest_items, manifests / "manifest.jsonl")
    write_jsonl(object_items, manifests / "objects.jsonl")
    generated.extend((manifests / "manifest.jsonl", manifests / "objects.jsonl"))
    for name, items in (
        ("candidates", candidates),
        ("excluded", excluded),
        ("duplicates", duplicate_rows),
    ):
        path = manifests / f"{name}.jsonl"
        write_jsonl(items, path)
        generated.append(path)
    for name in config["sources"]:
        path = manifests / f"candidates_{name}.jsonl"
        write_jsonl([item for item in candidates if item["source"] == name], path)
        generated.append(path)
    for split in ("train", "val"):
        selected_path = manifests / f"selected_{split}.txt"
        joint_path = manifests / f"{split}_joint_complete.txt"
        selected_path.write_text("\n".join(split_lists[split]) + "\n")
        joint_path.write_text("\n".join(joint_lists[split]) + "\n")
        generated.extend((selected_path, joint_path))
    partial_train = manifests / "train_requires_partial_labels.txt"
    partial_train.write_text(
        "".join(
            f"{item['path']}\n"
            for item in manifest_items
            if item["split"] == "train" and not item["joint_ready"]
        ),
        encoding="utf-8",
    )
    generated.append(partial_train)
    pending_path = manifests / "pending_review.csv"
    fields = sorted({key for item in pending for key in item}) or ["reason"]
    with pending_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pending)
    generated.append(pending_path)

    class_mapping = {
        name: source_config.get("category_roles", {})
        for name, source_config in config["sources"].items()
    }
    dump_json(class_mapping, output / "class_mapping.json")
    generated.append(output / "class_mapping.json")
    counts = Counter()
    for item in manifest_items:
        prefix = f"{item['split']}_{item['source']}"
        counts[f"{prefix}_images"] += 1
        counts[f"{prefix}_person"] += item["person_count"]
        counts[f"{prefix}_no_helmet"] += item["no_helmet_count"]
        if item["joint_ready"]:
            counts[f"{prefix}_joint_images"] += 1
            counts[f"{prefix}_joint_person"] += item["person_count"]
    stats = {
        "sources": source_stats,
        "counts": dict(sorted(counts.items())),
        "blocked_sources": blocked,
        "pending_review": len(pending),
        "deduplication": {
            "reference_images": reference_hash_count,
            "missing_reference_roots": missing_dedup_roots,
            "exact_duplicates_removed": len(duplicate_rows),
        },
        "geometry": {
            "deployment_canvas": input_size,
            "training_imgsz": training_imgsz,
            "rect_baseline": (
                "long edge resized to imgsz; aspect preserved; no augmentation"
            ),
            "deployment_letterbox": (
                "aspect preserved inside fixed canvas; not the training tensor shape"
            ),
        },
        "class_ids": {"person": PERSON_CLASS, "no_helmet_head": NO_HELMET_CLASS},
        "object_sizes": object_statistics(object_items),
        "head_statuses": dict(
            sorted(
                Counter(
                    item["head_status"] for item in object_items if item["head_status"]
                ).items()
            )
        ),
        "qa_errors": dict(
            sorted(
                Counter(
                    error.split(":", 1)[0]
                    for item in manifest_items
                    for error in item["qa_errors"]
                ).items()
            )
        ),
        "exclusions": {
            "images": len(excluded),
            "reasons": dict(
                Counter(
                    error.split(":", 1)[0]
                    for item in excluded
                    for error in item["qa_errors"]
                )
            ),
        },
        "assurance": dict(Counter(item["assurance"] for item in manifest_items)),
        "clipped_target_boxes": sum(
            item["target_xyxy"] is not None
            and item["target_xyxy"] != item["source_xyxy"]
            for item in object_items
        ),
        "split_integrity": split_integrity(manifest_items),
        "coverage": {
            "night_person": sum(
                item["person_count"]
                for item in manifest_items
                if item["source"] == "night_surveillance"
            ),
            "far_person": "not_classified: calibrate thresholds on factory examples",
            "very_far_person": (
                "not_classified: calibrate thresholds on factory examples"
            ),
            "reflective_vest": "unknown unless source metadata or review confirms it",
            "dark_helmet": "unknown unless source metadata or review confirms it",
        },
    }
    train_items = [item for item in manifest_items if item["split"] == "train"]
    stats["readiness"] = {
        "selected_person": sum(item["person_count"] for item in train_items),
        "joint_ready_person": sum(
            item["person_count"] for item in train_items if item["joint_ready"]
        ),
        "requires_partial_labels_person": sum(
            item["person_count"] for item in train_items if not item["joint_ready"]
        ),
        "blocked_target_person": sum(
            int(config["sources"][name]["train_person_boxes"]) for name in blocked
        ),
        "train_target_deficit_person": sum(
            int(source["train_person_boxes"]) for source in config["sources"].values()
        )
        - sum(item["person_count"] for item in train_items),
    }
    dump_json(stats, output / "stats.json")
    generated.append(output / "stats.json")
    resolved_config = output / "build-config.yaml"
    resolved_config.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    generated.append(resolved_config)
    provenance = output / "provenance_and_licenses"
    for name, source_config in config["sources"].items():
        path = provenance / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# {name}\n\n"
            f"- Root: `{source_config['root']}`\n"
            f"- Version: `{source_config.get('version', 'unknown')}`\n"
            f"- Source: {source_config.get('url', 'unknown')}\n"
            f"- Declared terms: `{source_config.get('license', 'unknown')}`\n"
            f"- Build status: `{source_stats[name]['status']}`\n\n"
            "The source files are not redistributed by this repository.\n",
            encoding="utf-8",
        )
        generated.append(path)
    readiness = stats["readiness"]
    qa_path = output / "qa_report.md"
    qa_path.write_text(
        "# External dataset QA\n\n"
        f"- Selected train person boxes: {readiness['selected_person']}\n"
        f"- Joint-ready train person boxes: {readiness['joint_ready_person']}\n"
        "- Train person boxes requiring partial-label support or review: "
        f"{readiness['requires_partial_labels_person']}\n"
        f"- Blocked target person boxes: {readiness['blocked_target_person']}\n"
        f"- Pending review rows: {len(pending)}\n\n"
        f"- Selected-frame QA errors: {sum(stats['qa_errors'].values())}\n"
        f"- Excluded candidate frames: {len(excluded)}\n"
        "- Individually reviewed frames: "
        f"{stats['assurance'].get('individually_reviewed', 0)}\n\n"
        "Only `joint_ready=true` images are linked into `.yolo/`. "
        "This means training-eligible under the configured source completeness "
        "policy plus automatic checks, not manually verified by us. Native "
        "annotations retain `verified=false`. Disputed frames are excluded before "
        "quota selection; inspect `manifests/excluded.jsonl` and "
        "`manifests/pending_review.csv`.\n\n"
        "SFCHD sequence IDs are inferred from filenames; they are not documented "
        "video identities. Zero group overlap does not prove camera or episode "
        "independence. See `stats.json:split_integrity` for camera overlap.\n\n"
        "Size statistics distinguish fixed-canvas deployment letterboxing from "
        "the rect-training long-edge baseline. Augmentation is not included.\n",
        encoding="utf-8",
    )
    generated.append(qa_path)
    package_readme = output / "README.md"
    package_readme.write_text(
        "# Prepared external dataset\n\n"
        "`train/<source>` and `val/<source>` contain image/YOLO-label pairs. "
        "Class 0 is person and class 2 is a head without a safety helmet.\n\n"
        "The hidden `.yolo/` directory contains only frames whose person and "
        "no-helmet annotations are accepted as complete under the configured "
        "source policy and automated checks, not necessarily individually reviewed. "
        "It is the only view used by the "
        "standard joint trainer. See `stats.json`, `qa_report.md`, and "
        "`manifests/pending_review.csv` for readiness details.\n",
        encoding="utf-8",
    )
    generated.append(package_readme)
    for (split, source), frames in _preview_groups(selected).items():
        for index, frame in enumerate(frames[:12]):
            preview = output / "previews" / split / source / f"{index:02d}.jpg"
            draw_preview(frame, preview)
            generated.append(preview)
    for (category, source), frames in _qa_preview_groups(
        [frame for _, frame in selected], training_imgsz
    ).items():
        for index, frame in enumerate(frames[:12]):
            preview = (
                output / "previews" / "qa" / category / source / f"{index:02d}.jpg"
            )
            draw_preview(frame, preview)
            generated.append(preview)
    generated_manifest = manifests / "generated_files.txt"
    relative_generated = sorted(
        str(path.relative_to(output)) for path in generated if path.exists()
    )
    generated_manifest.write_text("\n".join(relative_generated) + "\n")
    return stats


def _preview_groups(
    selected: list[tuple[str, SourceFrame]],
) -> dict[tuple[str, str], list[SourceFrame]]:
    groups: dict[tuple[str, str], list[SourceFrame]] = defaultdict(list)
    for split, frame in selected:
        groups[(split, frame.source)].append(frame)
    return groups


def _qa_preview_groups(
    frames: list[SourceFrame], training_imgsz: int
) -> dict[tuple[str, str], list[SourceFrame]]:
    groups: dict[tuple[str, str], list[SourceFrame]] = defaultdict(list)
    for frame in frames:
        categories: set[str] = set()
        if frame.no_helmet_count:
            categories.add("no-helmet")
        if any(box.role == "helmet_present" for box in frame.boxes):
            categories.add("helmet-present")
        if any(box.role == "unknown" for box in frame.boxes):
            categories.add("unknown-head")
        if not frame.joint_complete:
            categories.add("requires-review")
        gain = training_imgsz / max(frame.width, frame.height)
        if any(
            box.role == "no_helmet" and (box.xyxy[3] - box.xyxy[1]) * gain <= 12
            for box in frame.boxes
        ):
            categories.add("small-no-helmet")
        if frame.source == "night_surveillance":
            categories.add("night")
        for category in categories:
            groups[(category, frame.source)].append(frame)
    return groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the external YOLO dataset")
    parser.add_argument("--config", default="configs/external_dataset.yaml")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write available sources even when another source is missing",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build(Path(args.config).resolve(), allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
