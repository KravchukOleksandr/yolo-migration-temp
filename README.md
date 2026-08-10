# YOLOv8 Person Detection

A compact training repository for `yolov8n`, `yolov8s`, and `yolov8m`. It is
designed for a 16 GB NVIDIA T4, flat Azure Blob containers, mixed source image
sizes, and inference at both 640 and 1280 pixels.

The default model is `yolov8n`, optimized for fast iteration on a T4.

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

## 4. Add local pretrained weights

The training code never downloads model weights. Copy the required files into
the local `weights/` directory:

```text
weights/
├── yolov8n.pt
├── yolov8s.pt
└── yolov8m.pt
```

Only the selected model is required. The default configuration uses
`weights/yolov8n.pt`. Weight files are ignored by Git.

Verify the file before starting a long run:

```bash
ls -lh weights/yolov8n.pt
```

If the selected file is missing, training stops with a clear error and does not
attempt a network download. Change `weights_dir` in the training and tuning
configurations only when the files must be stored elsewhere.

## 5. Audit the dataset

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

## 6. Train a baseline

The default configuration trains `yolov8n` at a base size of 1024 with
multi-scale augmentation and an explicit batch size of 12 for a 16 GB T4.

```bash
make train
```

Select another model size without editing YAML:

```bash
yolo-train --config configs/train.yaml --model n
yolo-train --config configs/train.yaml --model m
```

Start with `yolov8n` for fast iteration. Test `yolov8s` or `yolov8m` only if
their accuracy improvement justifies the additional training and inference cost.

Resume an interrupted run:

```bash
yolo-train \
  --config configs/train.yaml \
  --resume runs/train/yolov8n/weights/last.pt
```

Training settings are stored in `configs/train.yaml`. Ultralytics saves losses,
metrics, plots, timings, and `best.pt`/`last.pt` under `runs/train/`.

## 7. Tune hyperparameters

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

The default fast search runs 8 trials of 15 epochs at 960 pixels. It tunes learning
rate, weight decay, mosaic, scale, translation, and mixup. Edit
`configs/tune.yaml` to change the trial count, epochs, model, or image size.
The best parameters are saved to `runs/tune/best.json`.

## 8. Tune, train, and validate with one command

After the dataset has been downloaded and audited, start the complete pipeline
in the background:

```bash
make
```

`make start` is equivalent. The launcher uses `nohup`, so the process continues
after closing the terminal or disconnecting SSH. It prints the PID and log path:

```text
Pipeline started in the background.
PID: 12345
Log: runs/launcher/20260101-120000.log
```

Follow progress with the printed path:

```bash
tail -f runs/launcher/20260101-120000.log
```

Check whether the process is still running:

```bash
ps -p 12345 -o pid,etime,cmd
```

To run in the foreground instead, use:

```bash
make pipeline
```

This command performs the following steps automatically:

1. Creates deterministic tuning subsets.
2. Runs Optuna hyperparameter search.
3. Applies the best parameters to the full training configuration.
4. Trains the selected model on the full dataset.
5. Validates `best.pt` at 640, 960, and 1280 pixels.

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

For an initial smoke test, set `trials: 2`, `epochs: 3` in `configs/tune.yaml` and
temporarily reduce `epochs` in `configs/train.yaml`.

## 9. Validate an existing checkpoint

Validate any checkpoint at all deployment resolutions:

```bash
yolo-validate \
  --weights runs/train/yolov8n/weights/best.pt \
  --imgsz 640 960 1280
```

Results are saved under `runs/validation/`. Compare recall, `mAP50-95`, and
inference time at 640 and 1280 before choosing the production resolution.

## 10. Quality checks

```bash
make lint
make test
```

Important configuration files:

- `configs/data.yaml` — local train/validation paths and class names;
- `configs/train.yaml` — full training settings;
- `configs/tune.yaml` — tuning budget and tuning dataset;
- `.env` — private Azure credentials, never committed.
