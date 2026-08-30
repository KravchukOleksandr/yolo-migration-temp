.DEFAULT_GOAL := start

.PHONY: start yolo26 yolo26-from-trial stop setup download audit subset tune train validate pipeline test lint

PYTHON ?= python3
TRAIN_CONFIG ?= configs/train_yolo26.yaml
TUNE_CONFIG ?= configs/tune_yolo26.yaml

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

stop:
	bash scripts/stop_all.sh

setup:
	$(PYTHON) -m pip install -e '.[dev,tune]'

download:
	yolo-download

audit:
	yolo-audit --data configs/data.yaml --output runs/audit

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
