#!/usr/bin/env bash
set -euo pipefail

run_id="$(date -u +%Y%m%d-%H%M%S)"
log_dir="runs/launcher"
log_file="${log_dir}/${run_id}.log"
pid_file="${log_dir}/${run_id}.pid"

mkdir -p "${log_dir}"
nohup python3 -m yolo_factory.pipeline --skip-audit "$@" \
    >"${log_file}" 2>&1 </dev/null &
pipeline_pid=$!
printf '%s\n' "${pipeline_pid}" >"${pid_file}"

echo "Pipeline started in the background."
echo "PID: ${pipeline_pid}"
echo "Log: ${log_file}"
echo "Follow: tail -f ${log_file}"
