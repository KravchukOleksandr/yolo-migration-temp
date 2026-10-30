import os
import subprocess
import time
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "start_pipeline.sh"


def fake_python(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-python"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "for argument in \"$@\"; do\n"
        "  if [[ \"${argument}\" == --check ]]; then\n"
        "    printf 'checked\\n'\n"
        "    exit \"${CHECK_EXIT:-0}\"\n"
        "  fi\n"
        "done\n"
        "printf 'unbuffered=%s\\n' \"${PYTHONUNBUFFERED:-}\"\n"
        "printf 'arg=%s\\n' \"$@\"\n"
        "printf 'done\\n'\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def test_launcher_honors_python_and_publishes_latest(tmp_path: Path) -> None:
    executable = fake_python(tmp_path)
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--use-trial", "custom trial/args.yaml"],
        cwd=tmp_path,
        env={**os.environ, "PYTHON": str(executable), "RUN_PREFIX": "custom"},
        check=True,
        text=True,
        capture_output=True,
    )

    run_dir = Path((tmp_path / "runs" / "LATEST.txt").read_text().strip())
    assert run_dir.is_dir()
    assert run_dir.name.startswith("custom-")
    assert (run_dir / "pipeline.pid").read_text().strip().isdigit()
    assert "Pipeline started in the background" in result.stdout
    log = run_dir / "pipeline.log"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        content = log.read_text() if log.exists() else ""
        if content.endswith("done\n"):
            break
        time.sleep(0.02)
    assert "unbuffered=1" in content
    assert "arg=custom trial/args.yaml" in content
    assert "arg=--skip-audit" not in content
    assert "arg=--check" not in content


def test_launcher_does_not_create_run_after_failed_check(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHON": str(fake_python(tmp_path)),
            "CHECK_EXIT": "7",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode == 7
    assert not (tmp_path / "runs").exists()


def test_launcher_check_only_does_not_start_background_process(tmp_path: Path) -> None:
    subprocess.run(
        ["bash", str(LAUNCHER), "--check"],
        cwd=tmp_path,
        env={**os.environ, "PYTHON": str(fake_python(tmp_path))},
        check=True,
        text=True,
        capture_output=True,
    )

    assert not (tmp_path / "runs").exists()


def test_launcher_rejects_run_id_override(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--run-id", "different-folder"],
        cwd=tmp_path,
        env={**os.environ, "PYTHON": str(fake_python(tmp_path))},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert not (tmp_path / "runs").exists()
