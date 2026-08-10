# YOLOv8 Person Detection

A compact YOLOv8 training pipeline for person detection on an Azure VM with an
NVIDIA T4 16 GB. The default model is `yolov8n`.

The pipeline uses local model weights and runs in the background. You may close
the terminal or disconnect SSH after it starts. The process continues while the
Azure VM remains running.

## What the pipeline does

One background command performs the complete experiment:

1. Creates deterministic train and validation tuning subsets.
2. Runs Optuna hyperparameter tuning.
3. Applies the best parameters to the full training configuration.
4. Trains YOLOv8n on the complete training dataset.
5. Validates `best.pt` at 640, 960, and 1280 pixels.
6. Saves checkpoints, metrics, plots, configuration, and console logs.

## 1. Create the environment

Use Python 3.10 or 3.11:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the repository through the required pip proxy:

```bash
export PIP_PROXY='http://user:password@proxy.example:3128'
python3 -m pip install --proxy "$PIP_PROXY" --upgrade pip
python3 -m pip install --proxy "$PIP_PROXY" -e '.[dev,tune]'
```

Do not store proxy credentials in the repository.

## 2. Add local model weights

The pipeline does not download model weights. Copy the nano checkpoint to this
exact path:

```text
weights/yolov8n.pt
```

Verify it before continuing:

```bash
ls -lh weights/yolov8n.pt
```

The `.pt` files under `weights/` are ignored by Git.

## 3. Configure Azure Storage

Create the private environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
AZURE_STORAGE_CONNECTION_STRING='paste-full-connection-string-here'
AZURE_TRAIN_CONTAINER='train-container-name'
AZURE_VAL_CONTAINER='validation-container-name'
```

Keep the full connection string in single quotes because it contains
semicolons. The `.env` file is ignored by Git and must never be committed.

Load the variables:

```bash
set -a
source .env
set +a
```

## 4. Download the dataset

Both Azure containers may be flat. Every image must have a `.txt` label with
the same stem:

```text
frame_0001.jpg
frame_0001.txt
frame_0002.png
frame_0002.txt
```

Download both containers:

```bash
make download
```

The downloader automatically creates:

```text
data/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

Existing complete files are skipped when the download is restarted.

## 5. Audit the dataset

Run the audit once after downloading or replacing the dataset:

```bash
make audit
```

Progress is printed every 1,000 images. The report is saved under:

```text
runs/dataset-audit/
├── summary.json
├── train.csv
└── val.csv
```

The audit checks images, YOLO labels, normalized coordinates, missing pairs,
image dimensions, and box-size statistics. Fix fatal errors before starting the
pipeline.

## 6. Start the complete pipeline

Run this single command:

```bash
make
```

The command starts the pipeline with `nohup` and immediately prints information
similar to:

```text
Pipeline started in the background.
PID: 12345
Log: runs/launcher/20260810-120000.log
Follow: tail -f runs/launcher/20260810-120000.log
```

You may now close the terminal or disconnect SSH. Do not stop or deallocate the
Azure VM.

## 7. Follow progress

Use the exact log path printed by the launcher:

```bash
tail -f runs/launcher/20260810-120000.log
```

Press `Ctrl+C` to stop following the log. This does not stop training.

Check whether the background process is alive using the printed PID:

```bash
ps -p 12345 -o pid,etime,cmd
```

Monitor GPU memory and utilization in another SSH session:

```bash
watch -n 2 nvidia-smi
```

## 8. Find results

Every pipeline execution receives a UTC timestamp:

```text
runs/pipeline/<timestamp>/
├── resolved-train.yaml
├── train/
│   ├── results.csv
│   ├── results.png
│   ├── confusion_matrix.png
│   └── weights/
│       ├── best.pt
│       └── last.pt
└── validation/
    ├── 640/
    ├── 960/
    ├── 1280/
    └── summary.json
```

Hyperparameter trials and the selected parameters are stored separately:

```text
runs/tune/
├── trial-000/
├── trial-001/
├── ...
└── best.json
```

The production candidate is:

```text
runs/pipeline/<timestamp>/train/weights/best.pt
```

## Default speed settings

The defaults target fast iteration on a T4 16 GB:

```text
model: yolov8n
base image size: 1024
multi-scale range: approximately 768-1280
training batch: 12
training epochs: 100
early stopping patience: 20
tuning trials: 8
tuning epochs per trial: 15
tuning batch: 16
AMP: enabled
deterministic mode: disabled
```

Configuration files:

- `configs/data.yaml` — local dataset paths and class names;
- `configs/train.yaml` — full training settings;
- `configs/tune.yaml` — tuning budget and model settings;
- `.env` — private Azure credentials, never committed.
