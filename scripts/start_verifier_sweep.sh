#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: bash scripts/start_verifier_sweep.sh detector.pt verifier.pt"
    exit 2
fi

detector="$1"
verifier="$2"
config="${VERIFIER_CONFIG:-configs/verifier.yaml}"
retention="${SAFE_TP_RETENTION:-0.995}"
python_cmd="${PYTHON:-python3}"
run_id="sweep-$(date -u +%Y%m%d-%H%M%S)-$$"
run_dir="runs/verifier-sweeps/${run_id}"
log_file="${run_dir}/sweep.log"
pid_file="${run_dir}/pipeline.pid"

mkdir -p "${run_dir}"
nohup setsid "${python_cmd}" -m yolo_factory.verifier_sweep \
    --config "${config}" \
    --detector "${detector}" \
    --verifier "${verifier}" \
    --retention "${retention}" \
    --output "${run_dir}" \
    >"${log_file}" 2>&1 </dev/null &
pipeline_pid=$!
printf '%s\n' "${pipeline_pid}" >"${pid_file}"
printf '%s\n' "$(realpath "${run_dir}")" >runs/LATEST_VERIFIER_SWEEP.txt

echo "Verifier proposal sweep started in the background."
echo "PID: ${pipeline_pid}"
echo "Log: ${log_file}"
echo "Follow: tail -f ${log_file}"
