.DEFAULT_GOAL := start

.PHONY: start yolo26 stop setup download audit subset tune train validate pipeline test lint

PYTHON ?= python3

start:
	bash scripts/start_pipeline.sh

yolo26:
	RUN_PREFIX=yolo26 bash scripts/start_pipeline.sh \
		--train-config configs/train_yolo26.yaml \
		--tune-config configs/tune_yolo26.yaml

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
