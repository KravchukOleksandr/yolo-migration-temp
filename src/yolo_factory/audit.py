from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from .common import dump_json, resolve_dataset_paths
from .dataset import box_diagonal, image_files, read_record


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def scan(image_root: Path, csv_path: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    diagonals: list[float] = []
    errors: list[str] = []
    missing_labels = 0
    empty_labels = 0
    for image_path in image_files(image_root):
        try:
            record = read_record(image_path, image_root)
        except (OSError, ValueError) as error:
            errors.append(str(error))
            continue
        if not record.label.exists():
            missing_labels += 1
        elif not record.boxes:
            empty_labels += 1
        image_diagonals = [box_diagonal(box, record) for box in record.boxes]
        diagonals.extend(image_diagonals)
        rows.append(
            {
                "image": str(image_path),
                "width": record.width,
                "height": record.height,
                "boxes": len(record.boxes),
                "min_box_diagonal": min(image_diagonals, default=""),
                "max_box_diagonal": max(image_diagonals, default=""),
            }
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    return {
        "images": len(rows),
        "boxes": len(diagonals),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "errors": errors[:100],
        "box_diagonal": {
            "mean": statistics.fmean(diagonals) if diagonals else None,
            "std": statistics.pstdev(diagonals) if len(diagonals) > 1 else None,
            "min": min(diagonals, default=None),
            "p10": percentile(diagonals, 0.10),
            "p50": percentile(diagonals, 0.50),
            "p90": percentile(diagonals, 0.90),
            "max": max(diagonals, default=None),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a YOLO detection dataset")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--output", default="runs/dataset-audit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root, config = resolve_dataset_paths(args.data)
    output = Path(args.output)
    report: dict[str, dict[str, object]] = {}
    for split in ("train", "val"):
        image_root = root / config[split]
        report[split] = scan(image_root, output / f"{split}.csv")
    dump_json(report, output / "summary.json")
    print(f"Audit written to {output}")
    invalid = any(
        split_report["errors"] or split_report["missing_labels"]
        for split_report in report.values()
    )
    if invalid:
        raise SystemExit("Dataset audit failed; inspect summary.json")


if __name__ == "__main__":
    main()
