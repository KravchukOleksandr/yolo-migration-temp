from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from azure.storage.blob import BlobServiceClient, ContainerClient

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def normalize_etag(value: object | None) -> str | None:
    if value is None:
        return None
    etag = str(value).strip()
    # Azure listing and download responses may differ in ETag quoting.
    if len(etag) >= 2 and etag.startswith('"') and etag.endswith('"'):
        return etag[1:-1]
    return etag


def blob_target(name: str) -> Path:
    source = PurePosixPath(name)
    if source.is_absolute() or ".." in source.parts or "\\" in name:
        raise ValueError(f"Unsafe blob path: {name}")
    category = "labels" if source.suffix.lower() == ".txt" else "images"
    parts = source.parts
    if parts and parts[0] == category:
        parts = parts[1:]
    if not parts:
        raise ValueError(f"Empty blob path: {name}")
    target = Path(category, *parts)
    return target.with_suffix(".txt") if category == "labels" else target


def download_blob(
    container: ContainerClient,
    name: str,
    size: int,
    destination: Path,
    etag: str | None = None,
    previous: dict[str, Any] | None = None,
) -> bool:
    relative = blob_target(name)
    target = destination / relative
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"Blob target escapes output directory: {name}")
    etag = normalize_etag(etag)
    if previous is not None:
        previous = {**previous, "etag": normalize_etag(previous.get("etag"))}
    current = {"etag": etag, "size": size, "target": relative.as_posix()}
    if etag and previous == current and target.is_file():
        if target.stat().st_size == size:
            return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(f"{target.suffix}.part")
    if partial.is_symlink():
        raise ValueError(f"Partial download must not be a symlink: {partial}")
    try:
        with partial.open("wb") as stream:
            downloader = container.download_blob(name)
            actual_etag = normalize_etag(
                getattr(getattr(downloader, "properties", None), "etag", None)
            )
            if etag and actual_etag is not None and actual_etag != etag:
                raise RuntimeError(
                    f"Blob changed during download; retry: {name} "
                    f"(expected ETag {etag!r}, received {actual_etag!r})"
                )
            downloader.readinto(stream)
        if partial.stat().st_size != size:
            raise OSError(f"Downloaded size does not match listing: {name}")
        partial.replace(target)
        if relative.parts[0] == "images":
            target.with_suffix(".npy").unlink(missing_ok=True)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return True


def validate_inventory(blobs: list[dict[str, Any]], destination: Path) -> None:
    targets: set[Path] = set()
    resolved: set[Path] = set()
    images: set[Path] = set()
    labels: set[Path] = set()
    for blob in blobs:
        relative = blob_target(blob["name"])
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError(f"Blob target escapes output directory: {blob['name']}")
        if relative in targets or target in resolved:
            raise ValueError(f"Multiple blobs map to one output file: {relative}")
        targets.add(relative)
        resolved.add(target)
        stem = Path(*relative.parts[1:]).with_suffix("")
        if relative.parts[0] == "images":
            if stem in images:
                raise ValueError(f"Multiple image extensions share one label: {stem}")
            images.add(stem)
        else:
            labels.add(stem)
    if not images or images != labels:
        raise RuntimeError(
            f"Invalid container inventory: {len(images)} images, "
            f"{len(images - labels)} images without labels, "
            f"{len(labels - images)} labels without images"
        )
    extras = [
        path.relative_to(destination)
        for category in ("images", "labels")
        for path in (destination / category).rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES | {".txt"}
        and path.relative_to(destination) not in targets
    ]
    if extras:
        raise RuntimeError(
            f"{destination}: {len(extras)} local dataset files are absent from the "
            "container. Use a clean output directory or move stale files first; "
            "nothing has been deleted."
        )


def save_download_state(state: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(".json.tmp")
    if temporary.is_symlink():
        raise ValueError(f"Temporary state must not be a symlink: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def download_container(
    service: BlobServiceClient,
    container_name: str,
    destination: Path,
    workers: int,
) -> None:
    if workers <= 0:
        raise ValueError("workers must be positive")
    container = service.get_container_client(container_name)
    blobs = [
        {
            "name": blob.name,
            "size": blob.size,
            "etag": normalize_etag(getattr(blob, "etag", None)),
        }
        for blob in container.list_blobs()
        if Path(blob.name).suffix.lower() in IMAGE_SUFFIXES | {".txt"}
    ]
    validate_inventory(blobs, destination)
    state_path = destination / ".download-state.json"
    if state_path.is_symlink():
        raise ValueError(f"Download state must not be a symlink: {state_path}")
    previous: dict[str, Any] = {}
    account = getattr(service, "account_name", None)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("version") != 1:
            raise ValueError(f"Unsupported download state: {state_path}")
        if state.get("container") == container_name and state.get("account") == account:
            previous = state["blobs"]
    records = {
        blob["name"]: {
            "etag": blob["etag"],
            "size": blob["size"],
            "target": blob_target(blob["name"]).as_posix(),
        }
        for blob in blobs
    }
    state = {
        "version": 1,
        "account": account,
        "container": container_name,
        "blobs": {name: record for name, record in previous.items() if name in records},
    }
    print(f"Downloading {len(blobs)} files from {container_name}", flush=True)
    downloaded = 0
    futures: dict[Future[bool], str] = {}
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            try:
                for blob in blobs:
                    future = executor.submit(
                        download_blob,
                        container,
                        blob["name"],
                        blob["size"],
                        destination,
                        blob["etag"],
                        previous.get(blob["name"]),
                    )
                    futures[future] = blob["name"]
                for index, future in enumerate(as_completed(futures), start=1):
                    downloaded += future.result()
                    name = futures[future]
                    state["blobs"][name] = records[name]
                    if index % 1000 == 0 or index == len(futures):
                        save_download_state(state, state_path)
                        print(f"{container_name}: {index}/{len(futures)}", flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    finally:
        for future, name in futures.items():
            if future.done() and not future.cancelled() and future.exception() is None:
                state["blobs"][name] = records[name]
        save_download_state(state, state_path)
    print(
        f"{container_name}: downloaded {downloaded}, "
        f"unchanged {len(blobs) - downloaded}",
        flush=True,
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
    from azure.storage.blob import BlobServiceClient

    connection_string = required_env("AZURE_STORAGE_CONNECTION_STRING")
    train_container = required_env("AZURE_TRAIN_CONTAINER")
    val_container = required_env("AZURE_VAL_CONTAINER")
    service = BlobServiceClient.from_connection_string(connection_string)
    output = Path(args.output)
    download_container(service, train_container, output / "train", args.workers)
    download_container(service, val_container, output / "val", args.workers)


if __name__ == "__main__":
    main()
