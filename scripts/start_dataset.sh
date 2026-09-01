#!/usr/bin/env bash
set -euo pipefail

run_id="dataset-$(date -u +%Y%m%d-%H%M%S)-$$"
run_dir="runs/dataset-preparation/${run_id}"
log_file="${run_dir}/dataset.log"
pid_file="${run_dir}/pipeline.pid"

mkdir -p "${run_dir}"
nohup setsid bash scripts/prepare_dataset.sh \
    >"${log_file}" 2>&1 </dev/null &
pipeline_pid=$!
printf '%s\n' "${pipeline_pid}" >"${pid_file}"
printf '%s\n' "$(realpath "${run_dir}")" >runs/LATEST_DATASET.txt

echo "Dataset preparation started in the background."
echo "PID: ${pipeline_pid}"
echo "Log: ${log_file}"
echo "Follow: tail -f ${log_file}"
