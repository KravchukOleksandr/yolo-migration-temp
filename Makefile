.DEFAULT_GOAL := start

.PHONY: start yolo26 yolo26-from-trial dataset verifier-prepare verifier verifier-validate stop setup download audit subset tune train validate pipeline test lint

PYTHON ?= python3
TRAIN_CONFIG ?= configs/train_yolo26.yaml
TUNE_CONFIG ?= configs/tune_yolo26.yaml
VERIFIER_CONFIG ?= configs/verifier.yaml
DATA_CONFIG ?= configs/data.yaml
DATA_OUTPUT ?= data
AUDIT_OUTPUT ?= runs/audit
DOWNLOAD_WORKERS ?= 16
VERIFIER_DATASET ?= data/verifier/manifest.csv
DETECTOR ?= weights/detector.pt
LATEST_VERIFIER := $(shell cat runs/LATEST_VERIFIER.txt 2>/dev/null)
VERIFIER_MODEL ?= $(if $(LATEST_VERIFIER),$(LATEST_VERIFIER)/train/last.pt,)
VERIFIER_OUTPUT ?= $(if $(LATEST_VERIFIER),$(LATEST_VERIFIER)/yolo-validation-manual,runs/verifier/manual-validation)

start:
	bash scripts/start_pipeline.sh

yolo26:
	RUN_PREFIX=yolo26 bash scripts/start_pipeline.sh \
		--train-config configs/train_yolo26.yaml \
		--tune-config configs/tune_yolo26.yaml

yolo26-from-trial:
	@test -n "$(TRIAL)" || (echo "Usage: make yolo26-from-trial TRIAL=path/to/trial-004"; exit 2)
	RUN_PREFIX=yolo26-trial bash scripts/start_pipeline.sh \
		--train-config "$(TRAIN_CONFIG)" \
		--tune-config "$(TUNE_CONFIG)" \
		--use-trial "$(TRIAL)"

dataset:
	PYTHON="$(PYTHON)" \
	DATA_CONFIG="$(DATA_CONFIG)" \
	DATA_OUTPUT="$(DATA_OUTPUT)" \
	AUDIT_OUTPUT="$(AUDIT_OUTPUT)" \
	DOWNLOAD_WORKERS="$(DOWNLOAD_WORKERS)" \
	VERIFIER_CONFIG="$(VERIFIER_CONFIG)" \
		bash scripts/start_dataset.sh

verifier-prepare:
	$(PYTHON) -m yolo_factory.verifier_prepare --config "$(VERIFIER_CONFIG)"

verifier:
	@test -f "$(VERIFIER_DATASET)" || (echo "Prepared dataset not found; run: make dataset"; exit 2)
	VERIFIER_CONFIG="$(VERIFIER_CONFIG)" bash scripts/start_verifier.sh \
		"$(DETECTOR)" --skip-prepare

verifier-validate:
	@test -f "$(DETECTOR)" || (echo "Detector not found: $(DETECTOR)"; exit 2)
	@test -n "$(VERIFIER_MODEL)" || (echo "No completed verifier run found"; exit 2)
	@test -f "$(VERIFIER_MODEL)" || (echo "Verifier not found: $(VERIFIER_MODEL)"; exit 2)
	$(PYTHON) -m yolo_factory.verifier_validate \
		--config "$(VERIFIER_CONFIG)" \
		--detector "$(DETECTOR)" \
		--verifier "$(VERIFIER_MODEL)" \
		--output "$(VERIFIER_OUTPUT)"

stop:
	bash scripts/stop_all.sh

setup:
	$(PYTHON) -m pip install -e '.[dev,tune]'

download:
	$(PYTHON) -m yolo_factory.download --output "$(DATA_OUTPUT)" \
		--workers "$(DOWNLOAD_WORKERS)"

audit:
	$(PYTHON) -m yolo_factory.audit --data "$(DATA_CONFIG)" \
		--output "$(AUDIT_OUTPUT)"

subset:
	yolo-subset --data configs/data.yaml --output runs/runtime/data_tune.yaml

tune:
	yolo-tune --config configs/tune.yaml

train:
	yolo-train --config configs/train.yaml

validate:
	yolo-validate --weights runs/train/yolov8s/weights/best.pt

pipeline:
	yolo-pipeline

test:
	pytest -q

lint:
	ruff check .
