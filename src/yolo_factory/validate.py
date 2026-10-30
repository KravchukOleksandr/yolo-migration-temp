from __future__ import annotations

import argparse
from pathlib import Path

from .common import dump_json, materialize_dataset_config
from .offline import offline_runtime
from .train import local_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate at deployment resolutions")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[1280])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--classes", nargs="+", type=int)
    parser.add_argument("--output", default="runs/validation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch <= 0:
        raise ValueError("Validation --batch must be a positive integer")
    if any(size <= 0 for size in args.imgsz):
        raise ValueError("Validation --imgsz values must be positive integers")
    weights = local_checkpoint(args.weights)
    output = Path(args.output).resolve()
    data = materialize_dataset_config(args.data)
    summary: dict[str, object] = {}
    with offline_runtime() as model_type:
        model = model_type(str(weights))
        for image_size in args.imgsz:
            parameters = {
                "data": str(data),
                "imgsz": image_size,
                "batch": args.batch,
                "project": str(output),
                "name": str(image_size),
                "plots": True,
            }
            if args.classes is not None:
                parameters["classes"] = args.classes
            metrics = model.val(**parameters)
            summary[str(image_size)] = {
                "metrics": metrics.results_dict,
                "speed_ms": metrics.speed,
            }
    dump_json(summary, output / "summary.json")


if __name__ == "__main__":
    main()
