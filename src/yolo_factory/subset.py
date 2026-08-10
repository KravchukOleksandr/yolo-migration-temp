from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from .common import resolve_dataset_paths
from .dataset import image_files


def choose(paths: list[Path], limit: int, seed: int) -> list[Path]:
    if limit >= len(paths):
        return paths
    generator = random.Random(seed)
    return sorted(generator.sample(paths, limit))


def write_manifest(paths: list[Path], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{path.resolve()}\n" for path in paths)
    target.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic tuning manifests"
    )
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--output", default="configs/data_tune.yaml")
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--val-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root, config = resolve_dataset_paths(args.data)
    output = Path(args.output).resolve()
    manifest_dir = output.parent / ".manifests"
    train = choose(image_files(root / config["train"]), args.train_size, args.seed)
    val = choose(image_files(root / config["val"]), args.val_size, args.seed + 1)
    train_manifest = manifest_dir / "tune_train.txt"
    val_manifest = manifest_dir / "tune_val.txt"
    write_manifest(train, train_manifest)
    write_manifest(val, val_manifest)
    subset_config = {
        "train": str(train_manifest),
        "val": str(val_manifest),
        "names": config["names"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(subset_config, sort_keys=False), encoding="utf-8")
    print(f"Created subset with {len(train)} train and {len(val)} val images")


if __name__ == "__main__":
    main()
