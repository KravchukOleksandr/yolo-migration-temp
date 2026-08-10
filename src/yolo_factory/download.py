from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContainerClient

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def download_blob(
    container: ContainerClient, name: str, size: int, destination: Path
) -> None:
    suffix = Path(name).suffix.lower()
    category = "labels" if suffix == ".txt" else "images"
    target = destination / category / Path(name).name
    if target.exists() and target.stat().st_size == size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(f"{target.suffix}.part")
    with partial.open("wb") as stream:
        container.download_blob(name).readinto(stream)
    partial.replace(target)


def download_container(
    service: BlobServiceClient,
    container_name: str,
    destination: Path,
    workers: int,
) -> None:
    container = service.get_container_client(container_name)
    blobs = [
        (blob.name, blob.size)
        for blob in container.list_blobs()
        if Path(blob.name).suffix.lower() in IMAGE_SUFFIXES | {".txt"}
    ]
    print(f"Downloading {len(blobs)} files from {container_name}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_blob, container, name, size, destination)
            for name, size in blobs
        ]
        for index, future in enumerate(futures, start=1):
            future.result()
            if index % 1000 == 0 or index == len(futures):
                print(f"{container_name}: {index}/{len(futures)}")
    image_stems = {path.stem for path in (destination / "images").glob("*")}
    label_stems = {path.stem for path in (destination / "labels").glob("*.txt")}
    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems
    if missing_labels or missing_images:
        raise RuntimeError(
            f"{container_name}: {len(missing_labels)} images without labels, "
            f"{len(missing_images)} labels without images"
        )


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Set {name}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download datasets from Azure Blob")
    parser.add_argument("--output", default="data")
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    connection_string = required_env("AZURE_STORAGE_CONNECTION_STRING")
    train_container = required_env("AZURE_TRAIN_CONTAINER")
    val_container = required_env("AZURE_VAL_CONTAINER")
    service = BlobServiceClient.from_connection_string(connection_string)
    output = Path(args.output)
    download_container(service, train_container, output / "train", args.workers)
    download_container(service, val_container, output / "val", args.workers)


if __name__ == "__main__":
    main()
