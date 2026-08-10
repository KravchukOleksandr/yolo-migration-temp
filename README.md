# YOLOv8 Person Detection

A compact training repository for `yolov8n`, `yolov8s`, and `yolov8m`. It is
designed for a 16 GB NVIDIA T4, flat Azure Blob containers, mixed source image
sizes, and inference at both 640 and 1280 pixels.

## 1. Expected Azure data

Use one container for training and one for validation. Both containers may be
flat. Every image must have a YOLO label with the same stem:

```text
frame_0001.jpg
frame_0001.txt
frame_0002.png
frame_0002.txt
```

Each label line must use this format:

```text
class_id x_center y_center width height
```

All coordinates must be normalized to the `[0, 1]` range. The current dataset
configuration contains one class: `person` with ID `0`.

## 2. Create the Python environment

Python 3.10 or 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If pip requires a proxy, export it before installation:

```bash
export PIP_PROXY='http://user:password@proxy.example:3128'
python3 -m pip install --proxy "$PIP_PROXY" --upgrade pip
python3 -m pip install --proxy "$PIP_PROXY" -e '.[dev,tune]'
```

Pip reads `PIP_PROXY` automatically, so the shorter installation command also
works after the variable has been exported:

```bash
make setup
```

Do not save a proxy password in the repository. Prefer a proxy URL without
embedded credentials when your environment supports it.

## 3. Configure Azure Storage

Create the local environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
AZURE_STORAGE_CONNECTION_STRING='paste-full-connection-string-here'
AZURE_TRAIN_CONTAINER='train-container-name'
AZURE_VAL_CONTAINER='validation-container-name'
```

Keep the connection string in single quotes because it contains semicolons.
The `.env` file is ignored by Git and must never be committed.

Load the variables and download the data:

```bash
set -a
source .env
set +a
make download
```

The downloader creates the local structure automatically:

```text
data/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

It supports interrupted downloads, uses 16 workers by default, and verifies
that every image has a matching label. To change concurrency:

```bash
yolo-download --workers 8
```

## 4. Audit the dataset

Run the audit before the first experiment:

```bash
make audit
```

The report is written to `runs/dataset-audit/` and contains:

- image and box counts;
- missing or empty labels;
- invalid normalized coordinates;
- image dimensions;
- box diagonal mean, deviation, percentiles, minimum, and maximum;
- one CSV row per image.

Fix all reported errors before training. Empty labels are allowed only when they
intentionally represent background images.

## 5. Train a baseline

The default configuration trains `yolov8s` at a base size of 1024 with
multi-scale augmentation. Batch size is selected automatically for the GPU.

```bash
make train
```

Select another model size without editing YAML:

```bash
yolo-train --config configs/train.yaml --model n
yolo-train --config configs/train.yaml --model m
```

Start with `yolov8s`. Use `yolov8n` as the speed baseline and test `yolov8m`
only if its accuracy improvement justifies its memory and latency cost.

Resume an interrupted run:

```bash
yolo-train \
  --config configs/train.yaml \
  --resume runs/train/yolov8s/weights/last.pt
```

Training settings are stored in `configs/train.yaml`. Ultralytics saves losses,
metrics, plots, timings, and `best.pt`/`last.pt` under `runs/train/`.

## 6. Tune hyperparameters

Create a deterministic subset first. By default it contains up to 10,000 train
and 2,000 validation images:

```bash
make subset
```

Change its size when needed:

```bash
yolo-subset \
  --data configs/data.yaml \
  --output configs/data_tune.yaml \
  --train-size 8000 \
  --val-size 2000 \
  --seed 42
```

Run Optuna tuning:

```bash
make tune
```

The default search runs 20 trials of 30 epochs at 960 pixels. It tunes learning
rate, weight decay, mosaic, scale, translation, and mixup. Edit
`configs/tune.yaml` to change the trial count, epochs, model, or image size.
The best parameters are saved to `runs/tune/best.json`.

## 7. Tune, train, and validate with one command

After the dataset has been downloaded, run the complete pipeline:

```bash
make pipeline
```

This command performs the following steps automatically:

1. Audits the full train and validation datasets.
2. Creates deterministic tuning subsets.
3. Runs Optuna hyperparameter search.
4. Applies the best parameters to the full training configuration.
5. Trains the selected model on the full dataset.
6. Validates `best.pt` at 640, 960, and 1280 pixels.

Each execution creates an isolated timestamped directory:

```text
runs/pipeline/<timestamp>/
├── dataset-audit/
├── resolved-train.yaml
├── train/
│   └── weights/
│       ├── best.pt
│       └── last.pt
└── validation/
    ├── 640/
    ├── 960/
    ├── 1280/
    └── summary.json
```

Skip the repeated audit when the dataset has already passed it:

```bash
yolo-pipeline --skip-audit
```

The pipeline may take a long time with the default 20 tuning trials. For an
initial smoke test, set `trials: 2`, `epochs: 3` in `configs/tune.yaml` and
temporarily reduce `epochs` in `configs/train.yaml`.

## 8. Validate an existing checkpoint

Validate any checkpoint at all deployment resolutions:

```bash
yolo-validate \
  --weights runs/train/yolov8s/weights/best.pt \
  --imgsz 640 960 1280
```

Results are saved under `runs/validation/`. Compare recall, `mAP50-95`, and
inference time at 640 and 1280 before choosing the production resolution.

## 9. Quality checks

```bash
make lint
make test
```

Important configuration files:

- `configs/data.yaml` — local train/validation paths and class names;
- `configs/train.yaml` — full training settings;
- `configs/tune.yaml` — tuning budget and tuning dataset;
- `.env` — private Azure credentials, never committed.
