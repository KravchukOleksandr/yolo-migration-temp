# YOLOv8 Safety Detection

A compact YOLOv8 training pipeline for safety detection on an Azure VM with an
NVIDIA T4 16 GB. The default model is `yolov8s` trained for `person` and
`no-helmet`.

The pipeline uses local model weights and runs in the background. You may close
the terminal or disconnect SSH after it starts. The process continues while the
Azure VM remains running.

## What the pipeline does

One background command performs the complete experiment:

1. Creates deterministic train and validation tuning subsets.
2. Runs Optuna hyperparameter tuning.
3. Applies the best parameters to the full training configuration.
4. Trains YOLOv8s on the complete training dataset.
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

The pipeline does not download model weights. Copy the small checkpoint to this
exact path:

```text
weights/yolov8s.pt
```

Verify it before continuing:

```bash
ls -lh weights/yolov8s.pt
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

## Validate another trained model

Use `yolo-validate` to evaluate any trained YOLO checkpoint without running
tuning or training:

```bash
yolo-validate \
  --weights /path/to/model/best.pt \
  --data configs/data.yaml \
  --imgsz 640 960 1280 \
  --batch 8 \
  --output runs/validation/other-model
```

Arguments:

- `--weights` — path to the trained `.pt` checkpoint;
- `--data` — dataset configuration containing the validation set path;
- `--imgsz` — one or more inference resolutions to test;
- `--batch` — positive validation batch size; the T4-safe default is `8`;
- `--output` — directory for metrics, plots, predictions, and the summary.

The example validates the same checkpoint three times and creates:

```text
runs/validation/other-model/
├── 640/
├── 960/
├── 1280/
└── summary.json
```

Use a single value when only one deployment resolution is required:

```bash
yolo-validate \
  --weights /path/to/model/best.pt \
  --data configs/data.yaml \
  --imgsz 1280 \
  --batch 8 \
  --output runs/validation/other-model-1280
```

Use a trained `best.pt` or `last.pt` checkpoint. A base COCO checkpoint may
have a different number of classes and is not directly comparable with the
fine-tuned safety detector.

## Class selection without rewriting labels

The source annotation IDs remain unchanged:

```text
0: person
1: helmet
2: no-helmet
```

The default training and tuning configurations contain:

```yaml
single_cls: false
classes: [0, 2]
```

Ultralytics keeps `person` and `no-helmet` and filters out `helmet` in memory.
The original `.txt` files are not edited, and `no-helmet` remains class ID `2`.
The dataset YAML must still list all three source classes.

When the dataset class configuration changes, the pipeline removes the generated
Ultralytics `labels.cache` files automatically. This forces a correct rescan but
does not modify images or `.txt` annotations.

Standalone validation of this model must use the same filter:

```bash
yolo-validate \
  --weights /path/to/model/best.pt \
  --data configs/data.yaml \
  --classes 0 2 \
  --imgsz 640 960 1280 \
  --batch 8 \
  --output runs/validation/person-no-helmet
```

The background pipeline passes the selected classes to validation
automatically.

## Run another model or class mode

Keep `configs/train.yaml` and `configs/tune.yaml` synchronized:

1. Set the same `model` (`n`, `s`, or `m`) in both files.
2. Set the same `classes` list in both files.
3. Keep `single_cls: false` when class IDs must remain distinct.
4. Add every source class ID to `configs/data.yaml`.
5. Place the matching local checkpoint under `weights/`.
6. Adjust batch sizes for the selected model.

Suggested T4 starting batches:

```text
YOLOv8n: train 12, tune 16
YOLOv8s: train 6, tune 8
YOLOv8m: train 2, tune 4
```

Then start the same detached pipeline:

```bash
make
```

For separate custom config files, pass them to the detached launcher:

```bash
bash scripts/start_pipeline.sh \
  --data configs/data.yaml \
  --train-config configs/train_custom.yaml \
  --tune-config configs/tune_custom.yaml
```

## Default speed settings

The defaults target fast iteration on a T4 16 GB:

```text
model: yolov8s
base image size: 1024
multi-scale range: approximately 768-1280
training classes: person (0), no-helmet (2)
training batch: 6
training epochs: 100
early stopping patience: 20
tuning trials: 8
tuning epochs per trial: 15
tuning batch: 8
TPE startup trials: 3
AMP: enabled
deterministic mode: disabled
```

Configuration files:

- `configs/data.yaml` — local dataset paths and class names;
- `configs/train.yaml` — full training settings;
- `configs/tune.yaml` — tuning budget and model settings;
- `.env` — private Azure credentials, never committed.
