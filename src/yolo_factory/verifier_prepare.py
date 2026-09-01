from __future__ import annotations

import argparse
import csv
import math
from hashlib import sha1
from pathlib import Path

from PIL import Image

from .common import dump_json, load_yaml, resolve_dataset_paths
from .dataset import image_files, read_record

Rectangle = tuple[float, float, float, float]


def intersection(first: Rectangle, second: Rectangle) -> Rectangle | None:
    value = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    return value if value[2] > value[0] and value[3] > value[1] else None


def union_area(rectangles: list[Rectangle]) -> float:
    if not rectangles:
        return 0.0
    xs = sorted({value for rectangle in rectangles for value in rectangle[::2]})
    total = 0.0
    for left, right in zip(xs[:-1], xs[1:], strict=True):
        intervals = sorted(
            (rectangle[1], rectangle[3])
            for rectangle in rectangles
            if rectangle[0] < right and rectangle[2] > left
        )
        if not intervals:
            continue
        covered = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                covered += end - start
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        covered += end - start
        total += (right - left) * covered
    return total


def absolute_box(box: tuple[float, ...], width: int, height: int) -> Rectangle:
    _, center_x, center_y, box_width, box_height = box
    pixel_width = box_width * width
    pixel_height = box_height * height
    center_x *= width
    center_y *= height
    return (
        center_x - pixel_width / 2,
        center_y - pixel_height / 2,
        center_x + pixel_width / 2,
        center_y + pixel_height / 2,
    )


def expand_box(box: Rectangle, scale: float) -> Rectangle:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    width = (box[2] - box[0]) * scale
    height = (box[3] - box[1]) * scale
    return (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare helmet verifier crops")
    parser.add_argument("--config", default="configs/verifier.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    data_path = Path(config["data"])
    if not data_path.is_absolute():
        data_path = config_path.parent / data_path
    root, data = resolve_dataset_paths(data_path)
    settings = config["prepare"]
    output = Path(settings["output"])
    if not output.is_absolute():
        output = (config_path.parent / output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    class_mapping = {
        int(settings.get("helmet_class", 1)): ("helmet", 0),
        int(settings.get("no_helmet_class", 2)): ("no_helmet", 1),
    }
    scale = float(settings.get("scale", 1.25))
    minimum_side = float(settings.get("minimum_scaled_side", 16.0))
    minimum_ratio = float(settings.get("minimum_aspect_ratio", 0.72))
    maximum_ratio = float(settings.get("maximum_aspect_ratio", 1.39))
    opposite_limit = float(settings.get("opposite_overlap", 0.10))
    same_limit = float(settings.get("same_overlap", 0.25))
    rows: list[dict[str, object]] = []
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for split in ("train", "val"):
        image_root = root / data[split]
        paths = image_files(image_root)
        print(f"Preparing {split}: {len(paths)} images", flush=True)
        for image_index, image_path in enumerate(paths, start=1):
            record = read_record(image_path, image_root)
            targets: list[tuple[int, int, Rectangle]] = []
            for index, box in enumerate(record.boxes):
                if not box[0].is_integer():
                    raise ValueError(
                        f"{record.label}: class ID must be an integer: {box[0]}"
                    )
                class_id = int(box[0])
                if class_id in class_mapping:
                    targets.append(
                        (
                            index,
                            class_id,
                            absolute_box(box, record.width, record.height),
                        )
                    )
            if not targets:
                continue
            with Image.open(image_path) as source:
                source = source.convert("RGB")
                relative = image_path.relative_to(image_root).as_posix()
                digest = sha1(relative.encode()).hexdigest()[:12]
                for box_index, class_id, box in targets:
                    expanded = expand_box(box, scale)
                    width = expanded[2] - expanded[0]
                    height = expanded[3] - expanded[1]
                    if min(width, height) < minimum_side:
                        reject("too_small")
                        continue
                    if (
                        expanded[0] < 0
                        or expanded[1] < 0
                        or expanded[2] > record.width
                        or expanded[3] > record.height
                    ):
                        reject("out_of_bounds")
                        continue
                    left = math.floor(expanded[0])
                    top = math.floor(expanded[1])
                    right = math.ceil(expanded[2])
                    bottom = math.ceil(expanded[3])
                    ratio = (right - left) / (bottom - top)
                    if ratio < minimum_ratio or ratio > maximum_ratio:
                        reject("aspect_ratio")
                        continue
                    crop_area = width * height
                    same: list[Rectangle] = []
                    opposite: list[Rectangle] = []
                    for other_index, other_class, other_box in targets:
                        if other_index == box_index:
                            continue
                        clipped = intersection(expanded, other_box)
                        if clipped is None:
                            continue
                        (same if other_class == class_id else opposite).append(clipped)
                    opposite_rejected = (
                        union_area(opposite) / crop_area > opposite_limit
                    )
                    same_rejected = union_area(same) / crop_area > same_limit
                    if opposite_rejected or same_rejected:
                        if opposite_rejected and same_rejected:
                            reject("both_overlaps")
                        elif opposite_rejected:
                            reject("opposite_overlap")
                        else:
                            reject("same_overlap")
                        continue
                    class_name, label = class_mapping[class_id]
                    destination = output / split / class_name
                    destination.mkdir(parents=True, exist_ok=True)
                    crop_path = destination / f"{digest}-{box_index:04d}.png"
                    source.crop((left, top, right, bottom)).save(crop_path)
                    rows.append(
                        {
                            "split": split,
                            "label": label,
                            "class_name": class_name,
                            "crop": crop_path.relative_to(output).as_posix(),
                            "source_image": relative,
                            "source_class": class_id,
                        }
                    )
            if image_index % 1000 == 0:
                print(f"Prepared {image_index}/{len(paths)} {split} images", flush=True)

    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "split",
            "label",
            "class_name",
            "crop",
            "source_image",
            "source_class",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['split']}/{row['class_name']}"
        counts[key] = counts.get(key, 0) + 1
    dump_json(
        {"counts": counts, "rejected": rejected, "manifest": manifest},
        output / "summary.json",
    )
    print(f"Verifier crops written to {output}")


if __name__ == "__main__":
    main()
