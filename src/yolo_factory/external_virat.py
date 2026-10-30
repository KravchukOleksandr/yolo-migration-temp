from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def finite_number(value: object, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: expected a number, got {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{context}: expected a finite number, got {value!r}")
    return number


def nonnegative_integer(value: object, context: str) -> int:
    number = finite_number(value, context)
    if number < 0 or not number.is_integer():
        raise ValueError(f"{context}: expected a non-negative integer, got {value!r}")
    return int(number)


def kpf_records(path: Path, key: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            value = yaml.safe_load(line)
        except yaml.YAMLError as error:
            raise ValueError(f"{path}:{line_number}: invalid KPF YAML") from error
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected a KPF mapping")
            record = item.get(key)
            if isinstance(record, dict):
                records.append(record)
            elif key in {"types", "geom"} and key not in item:
                records.append(item)
            elif record is not None:
                raise ValueError(f"{path}:{line_number}: malformed {key} record")
    return records


def track_types(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for record in kpf_records(path, "types"):
        track = record.get("id1")
        classes = record.get("cset3", {})
        if track is None or not isinstance(classes, dict) or not classes:
            continue
        track_id = nonnegative_integer(track, f"{path}: track ID")
        scores = {
            str(label): finite_number(score, f"{path}: score for {label}")
            for label, score in classes.items()
        }
        result[track_id] = max(scores, key=scores.get)
    return result


def geometry(path: Path, types: dict[int, str]) -> dict[int, list[dict[str, Any]]]:
    frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in kpf_records(path, "geom"):
        track = record.get("id1")
        frame = record.get("ts0")
        coordinates = record.get("g0")
        if track is None:
            continue
        track_id = nonnegative_integer(track, f"{path}: track ID")
        if "person" not in types.get(track_id, "").casefold():
            continue
        if frame is None or coordinates is None:
            raise ValueError(f"{path}: incomplete geometry for person track {track_id}")
        frame_id = nonnegative_integer(frame, f"{path}: frame index")
        if isinstance(coordinates, str):
            values = coordinates.split()
        else:
            try:
                values = list(coordinates)
            except TypeError as error:
                raise ValueError(
                    f"{path}: malformed geometry for person track {track_id}"
                ) from error
        if len(values) != 4:
            raise ValueError(
                f"{path}: expected four coordinates for person track {track_id}"
            )
        xyxy = [
            finite_number(value, f"{path}: geometry for person track {track_id}")
            for value in values
        ]
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            raise ValueError(f"{path}: invalid box for person track {track_id}: {xyxy}")
        frames[frame_id].append(
            {
                "role": "person",
                "source_category": types[track_id],
                "xyxy": xyxy,
                "track_id": track_id,
                "annotation_method": "virat-kpf",
                "verified": False,
            }
        )
    return frames


def matching_file(root: Path, stem: str, suffixes: set[str]) -> Path:
    candidates = [
        path
        for path in root.rglob("*")
        if path.suffix.casefold() in suffixes
        and (path.stem == stem or path.stem.startswith(stem))
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one file for {stem}, found {len(candidates)} under {root}"
        )
    return candidates[0]


def annotation_stem(path: Path) -> str:
    name = path.name
    for suffix in (".geom.yml", ".geom.yaml"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def sample_frames(
    frames: dict[int, list[dict[str, Any]]], minimum_gap: int, limit: int
) -> list[int]:
    selected: list[int] = []
    boxes = 0
    previous: int | None = None
    for frame in sorted(frames):
        if previous is not None and frame - previous < minimum_gap:
            continue
        selected.append(frame)
        boxes += len(frames[frame])
        previous = frame
        if boxes >= limit:
            break
    return selected


def extract_frames(video: Path, indices: list[int], output: Path) -> tuple[int, int]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required to extract VIRAT frames") from error
    if any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in indices
    ):
        raise ValueError("Frame indices must be non-negative integers")
    wanted = set(indices)
    maximum = max(indices, default=-1)
    output.mkdir(parents=True, exist_ok=True)
    targets = {index: output / f"{index:08d}.jpg" for index in wanted}
    for target in targets.values():
        target.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {video}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid video dimensions {width}x{height}: {video}")
        current = 0
        while current <= maximum:
            ok, image = capture.read()
            if not ok:
                break
            if current in wanted and not cv2.imwrite(str(targets[current]), image):
                raise OSError(f"Cannot write extracted frame: {targets[current]}")
            current += 1
        missing = [index for index, target in targets.items() if not target.is_file()]
        if missing:
            raise ValueError(
                f"Video ended before frames {sorted(missing)[:5]} in {video}"
            )
        return width, height
    except Exception:
        for target in targets.values():
            target.unlink(missing_ok=True)
        raise
    finally:
        capture.release()


def import_virat(
    root: Path,
    output: Path,
    minimum_gap: int,
    boxes_per_video: int,
) -> dict[str, int]:
    if minimum_gap < 0:
        raise ValueError("Minimum frame gap must be non-negative")
    if boxes_per_video <= 0:
        raise ValueError("Boxes per video must be positive")
    reviewed_manifest = output / "reviewed_manifest.jsonl"
    if reviewed_manifest.exists():
        raise FileExistsError(
            f"Refusing to overwrite reviewed VIRAT data at {reviewed_manifest}. "
            "Use a new --output directory or archive the previous review first."
        )
    geometries = sorted(root.rglob("*.geom.yml")) + sorted(root.rglob("*.geom.yaml"))
    if not geometries:
        raise FileNotFoundError(f"No .geom.yml files found under {root}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    stats_path = output / "import-stats.json"
    manifest.unlink(missing_ok=True)
    stats_path.unlink(missing_ok=True)
    records: list[dict[str, Any]] = []
    total_boxes = 0
    for geometry_path in geometries:
        stem = annotation_stem(geometry_path)
        type_candidates = list(root.rglob(f"{stem}*.types.yml"))
        type_candidates += list(root.rglob(f"{stem}*.types.yaml"))
        if len(type_candidates) != 1:
            raise FileNotFoundError(f"Missing types KPF for {geometry_path}")
        types_path = type_candidates[0]
        video = matching_file(root, stem, {".mp4", ".avi", ".mov", ".mkv"})
        boxes_by_frame = geometry(geometry_path, track_types(types_path))
        selected = sample_frames(boxes_by_frame, minimum_gap, boxes_per_video)
        frame_dir = output / "frames" / stem
        width, height = extract_frames(video, selected, frame_dir)
        for frame_index in selected:
            objects = boxes_by_frame[frame_index]
            records.append(
                {
                    "image": str(
                        (frame_dir / f"{frame_index:08d}.jpg").relative_to(output)
                    ),
                    "source_id": f"{stem}:{frame_index}",
                    "width": width,
                    "height": height,
                    "group_id": stem,
                    "video_id": stem,
                    "frame_index": frame_index,
                    "person_annotation_complete": True,
                    "no_helmet_annotation_complete": False,
                    "objects": objects,
                }
            )
            total_boxes += len(objects)
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    stats = {"videos": len(geometries), "frames": len(records), "person": total_boxes}
    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import DIVA VIRAT KPF annotations")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("external_dataset/raw/virat")
    )
    parser.add_argument("--min-frame-gap", type=int, default=30)
    parser.add_argument("--boxes-per-video", type=int, default=600)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = import_virat(
        args.root.resolve(),
        args.output.resolve(),
        args.min_frame_gap,
        args.boxes_per_video,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
