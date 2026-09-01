#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/start_verifier.sh /path/to/detector.pt"
    exit 2
fi

detector="$1"
shift
config="${VERIFIER_CONFIG:-configs/verifier.yaml}"
if [[ ! -f "${detector}" ]]; then
    echo "Detector checkpoint does not exist: ${detector}"
    exit 2
fi
run_id="verifier-$(date -u +%Y%m%d-%H%M%S)-$$"
run_dir="runs/verifier/${run_id}"
log_file="${run_dir}/pipeline.log"
pid_file="${run_dir}/pipeline.pid"

mkdir -p "${run_dir}"
nohup setsid python3 -m yolo_factory.verifier_pipeline \
    --config "${config}" \
    --detector "${detector}" \
    --run-id "${run_id}" \
    "$@" >"${log_file}" 2>&1 </dev/null &
pipeline_pid=$!
printf '%s\n' "${pipeline_pid}" >"${pid_file}"

echo "Verifier pipeline started in the background."
echo "PID: ${pipeline_pid}"
echo "Log: ${log_file}"
echo "Follow: tail -f ${log_file}"
