#!/usr/bin/env bash
set -euo pipefail

run_id="$(date -u +%Y%m%d-%H%M%S)-$$"
run_dir="runs/experiments/${run_id}"
log_file="${run_dir}/pipeline.log"
pid_file="${run_dir}/pipeline.pid"

mkdir -p "${run_dir}"
nohup setsid python3 -m yolo_factory.pipeline --skip-audit --run-id "${run_id}" "$@" \
    >"${log_file}" 2>&1 </dev/null &
pipeline_pid=$!
printf '%s\n' "${pipeline_pid}" >"${pid_file}"

echo "Pipeline started in the background."
echo "PID: ${pipeline_pid}"
echo "Log: ${log_file}"
echo "Follow: tail -f ${log_file}"
