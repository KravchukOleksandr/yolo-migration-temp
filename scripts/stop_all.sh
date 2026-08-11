#!/usr/bin/env bash
set -euo pipefail

found=false

for pid_file in runs/experiments/*/pipeline.pid; do
    [[ -f "${pid_file}" ]] || continue
    pipeline_pid="$(<"${pid_file}")"
    [[ "${pipeline_pid}" =~ ^[0-9]+$ ]] || continue
    if kill -0 "${pipeline_pid}" 2>/dev/null; then
        found=true
        echo "Stopping process group ${pipeline_pid}"
        kill -TERM -- "-${pipeline_pid}" 2>/dev/null || true
    fi
done

pattern='python[^ ]* -m yolo_factory\.(pipeline|train|tune|validate)'
if pgrep -f "${pattern}" >/dev/null; then
    found=true
    pkill -TERM -f "${pattern}" || true
fi

if [[ "${found}" == false ]]; then
    echo "No active YOLO Factory processes found."
    exit 0
fi

sleep 3

for pid_file in runs/experiments/*/pipeline.pid; do
    [[ -f "${pid_file}" ]] || continue
    pipeline_pid="$(<"${pid_file}")"
    [[ "${pipeline_pid}" =~ ^[0-9]+$ ]] || continue
    if kill -0 "${pipeline_pid}" 2>/dev/null; then
        echo "Force stopping process group ${pipeline_pid}"
        kill -KILL -- "-${pipeline_pid}" 2>/dev/null || true
    fi
done

pkill -KILL -f "${pattern}" 2>/dev/null || true
echo "All YOLO Factory training and validation processes were stopped."
