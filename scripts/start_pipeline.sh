#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-python3}"
run_prefix="${RUN_PREFIX:-yolo26}"
[[ "${run_prefix}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "RUN_PREFIX must contain only letters, digits, dots, underscores or hyphens." >&2
    exit 2
}
for argument in "$@"; do
    if [[ "${argument}" == --run-id || "${argument}" == --run-id=* ]]; then
        echo "The launcher assigns the run ID; set RUN_PREFIX instead." >&2
        exit 2
    fi
done
command -v nohup >/dev/null
command -v setsid >/dev/null
"${python_bin}" -m yolo_factory.pipeline --check "$@"
for argument in "$@"; do
    [[ "${argument}" != --check ]] || exit 0
done

run_id="${run_prefix}-$(date -u +%Y%m%d-%H%M%S)-$$"
run_dir="runs/experiments/${run_id}"
log_file="${run_dir}/pipeline.log"
pid_file="${run_dir}/pipeline.pid"

mkdir -p "${run_dir}"
export PYTHONUNBUFFERED=1
nohup setsid "${python_bin}" -m yolo_factory.pipeline --run-id "${run_id}" "$@" \
    >"${log_file}" 2>&1 </dev/null &
pipeline_pid=$!
printf '%s\n' "${pipeline_pid}" >"${pid_file}"
printf '%s\n' "${PWD}/${run_dir}" >runs/LATEST.txt

echo "Pipeline started in the background."
echo "PID: ${pipeline_pid}"
echo "Log: ${log_file}"
echo "Follow: tail -f ${log_file}"
