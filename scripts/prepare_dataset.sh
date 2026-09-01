#!/usr/bin/env bash
set -euo pipefail

python_cmd="${PYTHON:-python3}"
data_config="${DATA_CONFIG:-configs/data.yaml}"
data_output="${DATA_OUTPUT:-data}"
audit_output="${AUDIT_OUTPUT:-runs/audit}"
download_workers="${DOWNLOAD_WORKERS:-16}"
verifier_config="${VERIFIER_CONFIG:-configs/verifier.yaml}"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

echo "Stage 1/3: downloading train and validation containers"
"${python_cmd}" -m yolo_factory.download \
    --output "${data_output}" \
    --workers "${download_workers}"

echo "Stage 2/3: auditing the downloaded YOLO dataset"
"${python_cmd}" -m yolo_factory.audit \
    --data "${data_config}" \
    --output "${audit_output}"

echo "Stage 3/3: preparing helmet verifier crops"
"${python_cmd}" -m yolo_factory.verifier_prepare \
    --config "${verifier_config}"

echo "Dataset preparation completed."
