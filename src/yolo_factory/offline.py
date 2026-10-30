from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch


@contextmanager
def offline_runtime() -> Iterator[Any]:
    """Keep Ultralytics assets and integrations local to this process."""
    import matplotlib

    runtime = Path("runs/runtime/ultralytics").resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    for name in ("Arial.ttf", "Arial.Unicode.ttf"):
        target = runtime / name
        if not target.is_file():
            shutil.copyfile(font, target)

    with ExitStack() as stack:
        stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "YOLO_OFFLINE": "true",
                    "YOLO_AUTOINSTALL": "false",
                    "YOLO_CONFIG_DIR": str(runtime),
                },
            )
        )
        from ultralytics import YOLO, utils
        from ultralytics.data import utils as data_utils
        from ultralytics.utils import callbacks, checks, downloads, plotting

        def no_download(*args: Any, **kwargs: Any) -> None:
            raise ConnectionError("Offline mode: automatic asset downloads disabled")

        for name in ("safe_download", "get_github_assets"):
            stack.enter_context(patch.object(downloads, name, no_download))
        for module in (checks, data_utils, plotting):
            stack.enter_context(patch.object(module, "check_font", return_value=font))
        stack.enter_context(patch.object(utils, "ONLINE", False))
        stack.enter_context(patch.object(checks, "ONLINE", False))
        stack.enter_context(patch.object(checks, "AUTOINSTALL", False))
        stack.enter_context(
            patch.object(checks, "check_pip_update_available", return_value=False)
        )
        stack.enter_context(
            patch.object(callbacks, "add_integration_callbacks", return_value=None)
        )
        yield YOLO
