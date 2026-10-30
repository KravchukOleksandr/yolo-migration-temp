.DEFAULT_GOAL := start

.PHONY: start yolo26 yolov8 check yolo26-from-trial external-dataset external-import-virat external-apply-review external-audit stop setup download audit subset tune train validate pipeline test lint

PYTHON ?= python3
DATA_CONFIG ?= configs/data.yaml
TRAIN_CONFIG ?= configs/train_yolo26.yaml
TUNE_CONFIG ?= configs/tune_yolo26.yaml
TUNE_DATA ?= runs/runtime/data_tune_yolo26.yaml
RUN_PREFIX ?= yolo26
VAL_OUTPUT ?= runs/validation/standalone
VAL_IMGSZ ?= 1280
VAL_BATCH ?= 8
VAL_CLASSES ?= 0 2
PIPELINE_ARGS = --data "$(DATA_CONFIG)" --train-config "$(TRAIN_CONFIG)" --tune-config "$(TUNE_CONFIG)"

start yolo26:
	PYTHON="$(PYTHON)" RUN_PREFIX="$(RUN_PREFIX)" bash scripts/start_pipeline.sh $(PIPELINE_ARGS)

yolov8:
	PYTHON="$(PYTHON)" RUN_PREFIX=yolov8 bash scripts/start_pipeline.sh \
		--data "$(DATA_CONFIG)" \
		--train-config configs/train.yaml \
		--tune-config configs/tune.yaml

check:
	$(PYTHON) -m yolo_factory.pipeline $(PIPELINE_ARGS) --check

yolo26-from-trial:
	@test -n "$(TRIAL)" || (echo "Usage: make yolo26-from-trial TRIAL=path/to/trial-004"; exit 2)
	PYTHON="$(PYTHON)" RUN_PREFIX=yolo26-trial bash scripts/start_pipeline.sh \
		$(PIPELINE_ARGS) \
		--use-trial "$(TRIAL)"

external-dataset:
	$(PYTHON) -m yolo_factory.external_dataset --config configs/external_dataset.yaml

external-import-virat:
	@test -n "$(VIRAT_ROOT)" || (echo "Usage: make external-import-virat VIRAT_ROOT=/path/to/virat"; exit 2)
	$(PYTHON) -m yolo_factory.external_virat --root "$(VIRAT_ROOT)"

external-apply-review:
	@test -n "$(MANIFEST)" -a -n "$(REVIEWS)" || (echo "Usage: make external-apply-review MANIFEST=... REVIEWS=..."; exit 2)
	$(PYTHON) -m yolo_factory.external_review --manifest "$(MANIFEST)" --reviews "$(REVIEWS)" $(if $(REVIEW_OUTPUT),--output "$(REVIEW_OUTPUT)")

external-audit:
	$(PYTHON) -m yolo_factory.audit --data configs/data_external_val.yaml --output runs/audit/external

stop:
	bash scripts/stop_all.sh

setup:
	$(PYTHON) -m pip install -e '.[dev,tune]'

download:
	$(PYTHON) -m yolo_factory.download

audit:
	$(PYTHON) -m yolo_factory.audit --data "$(DATA_CONFIG)" --output runs/audit

subset:
	$(PYTHON) -m yolo_factory.subset --data "$(DATA_CONFIG)" --output "$(TUNE_DATA)"

tune:
	$(PYTHON) -m yolo_factory.tune --config "$(TUNE_CONFIG)"

train:
	$(PYTHON) -m yolo_factory.train --config "$(TRAIN_CONFIG)"

validate:
	@test -n "$(WEIGHTS)" || (echo "Usage: make validate WEIGHTS=path/to/best.pt"; exit 2)
	$(PYTHON) -m yolo_factory.validate --weights "$(WEIGHTS)" \
		--data "$(DATA_CONFIG)" --imgsz $(VAL_IMGSZ) --batch $(VAL_BATCH) \
		--classes $(VAL_CLASSES) --output "$(VAL_OUTPUT)"

pipeline:
	$(PYTHON) -m yolo_factory.pipeline $(PIPELINE_ARGS)

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
