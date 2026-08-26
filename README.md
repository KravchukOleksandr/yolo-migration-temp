# YOLO Safety Detection

A compact Ultralytics training pipeline for safety detection on an Azure VM
with an NVIDIA T4 16 GB. It supports the existing `YOLOv8s` workflow and a
separate `YOLO26s` workflow. Both train `person` and `no-helmet`.

The pipeline uses local model weights and runs in the background. You may close
the terminal or disconnect SSH after it starts. The process continues while the
Azure VM remains running.

## What the pipeline does

One background command performs the complete experiment:

1. Creates deterministic train and validation tuning subsets.
2. Runs Optuna hyperparameter tuning.
3. Applies the best parameters to the full training configuration.
4. Trains the selected model on the complete training dataset.
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

The pipeline does not download model weights. Copy the checkpoint for the mode
you want to run:

```text
weights/yolov8s.pt
weights/yolo26s.pt
```

Verify it before continuing:

```bash
ls -lh weights/yolov8s.pt
ls -lh weights/yolo26s.pt
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
runs/audit/
├── summary.json
├── train.csv
└── val.csv
```

The audit checks images, YOLO labels, normalized coordinates, missing pairs,
image dimensions, and box-size statistics. Fix fatal errors before starting the
pipeline.

## 6. Start the complete pipeline

Start the existing YOLOv8s pipeline:

```bash
make
```

Start the YOLO26s pipeline:

```bash
make yolo26
```

Each command starts its own detached tuning, training, and validation pipeline.
The YOLOv8 experiment ID starts with `yolov8-`; the YOLO26 experiment ID starts
with `yolo26-`.

The command starts the pipeline with `nohup` and immediately prints information
similar to:

```text
Pipeline started in the background.
PID: 12345
Log: runs/experiments/yolo26-20260810-120000-12345/pipeline.log
Follow: tail -f runs/experiments/yolo26-20260810-120000-12345/pipeline.log
```

You may now close the terminal or disconnect SSH. Do not stop or deallocate the
Azure VM.

## 7. Follow progress

Use the exact log path printed by the launcher:

```bash
tail -f "$(cat runs/LATEST.txt)/pipeline.log"
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

## 8. Stop all training and validation processes

Stop every YOLO Factory pipeline, tuning trial, training run, and validation
started by this repository:

```bash
make stop
```

The command first sends `SIGTERM` to allow a clean shutdown. After three
seconds, it force-stops any remaining process group with `SIGKILL`.

An interrupted training run normally keeps the most recently written
`last.pt`, but the current epoch may be incomplete. An interrupted Optuna study
does not write `best.json` until the study finishes, so its in-memory progress
is lost.

## 9. Find results

All outputs follow one fixed layout:

```text
runs/
├── LATEST.txt
├── audit/
│   ├── summary.json
│   ├── train.csv
│   └── val.csv
├── experiments/
│   └── <experiment-id>/
│       ├── pipeline.log
│       ├── pipeline.pid
│       ├── resolved-tune.yaml
│       ├── resolved-train.yaml
│       ├── tune/
│       │   ├── trial-000/
│       │   ├── trial-001/
│       │   └── best.json
│       ├── train/
│       │   ├── results.csv
│       │   ├── results.png
│       │   └── weights/
│       │       ├── best.pt
│       │       └── last.pt
│       └── validation/
│           ├── 640/
│           ├── 960/
│           ├── 1280/
│           └── summary.json
├── validation/
│   └── <standalone-name>/
└── runtime/
    ├── datasets/
    └── .manifests/
```

`runs/LATEST.txt` always contains the absolute path to the newest pipeline
directory. Print it with:

```bash
cat runs/LATEST.txt
```

Generated tuning manifests and the temporary dataset YAML are stored under
`runs/runtime/`. Runtime files are ignored by Git and never modify `configs/`.

The production candidate is:

```text
runs/experiments/<experiment-id>/train/weights/best.pt
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

Keep the matching train and tune configuration pair synchronized. Use
`train.yaml` and `tune.yaml` for YOLOv8, or `train_yolo26.yaml` and
`tune_yolo26.yaml` for YOLO26:

1. Set the same `family` and `model` size in both files.
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
YOLO26s: train 6, tune 8
```

Then start the matching detached pipeline:

```bash
make          # YOLOv8
make yolo26   # YOLO26
```

For separate custom config files, pass them to the detached launcher:

```bash
bash scripts/start_pipeline.sh \
  --data configs/data.yaml \
  --train-config configs/train_custom.yaml \
  --tune-config configs/tune_custom.yaml
```

## YOLO26 mode

YOLO26 is configured separately so its recipe does not overwrite or reuse the
YOLOv8 hyperparameters:

```text
configs/train_yolo26.yaml
configs/tune_yolo26.yaml
weights/yolo26s.pt
```

Start it with:

```bash
make yolo26
```

YOLO26 requires `ultralytics>=8.4.26`. Running the installation command from
section 1 upgrades an older environment through the configured pip proxy.

The YOLO26 tuning profile uses `MuSGD` and a declarative search space based on
the [official YOLO26s fine-tuning recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/).
It searches:

- initial and final learning-rate factors;
- momentum and weight decay;
- warmup duration;
- mosaic, mixup, scale, and translation augmentation.

YOLO26-specific loss gains and the remaining augmentation defaults stay fixed
at the official YOLO26s recipe values. The search runs 12 trials: the first 4
are exploratory and the remaining trials use TPE observations.

The tuning objective is class `2` (`no-helmet`) mAP50-95. Aggregate mAP50-95 is
also saved in each Optuna trial and `tune/best.json`. The final model is still
trained and validated for classes `[0, 2]`.

YOLO26 uses its native end-to-end one-to-one head by default, so normal
validation and inference are NMS-free. The output folders and plots are the
same as for YOLOv8.

### YOLO26 configuration files

`configs/train_yolo26.yaml` controls the final training on the complete train
dataset. It contains the image size, batch size, epoch limit, early stopping,
optimizer, augmentations, and selected classes.

`configs/tune_yolo26.yaml` controls the preliminary Optuna search on the
10,000-image tuning subset. It contains the number and length of trials, the
metric to maximize, fixed parameters, and parameter search ranges. When tuning
finishes, the pipeline copies the best searched values into the final training
configuration automatically.

The `family`, `model`, `classes`, and `single_cls` values must agree in both
files. The pipeline checks this before starting an expensive tuning run.

### Train YOLO26s on person only

In `configs/train_yolo26.yaml`, set:

```yaml
train:
  single_cls: true
  classes: [0]
```

In `configs/tune_yolo26.yaml`, set:

```yaml
objective_class: 0
single_cls: true
classes: [0]
```

Then run:

```bash
make yolo26
```

Only source class `0` (`person`) is used. Classes `1` and `2` are filtered in
memory; the original label files are not modified. To restore the default
person and no-helmet mode, set `single_cls: false`, `classes: [0, 2]`, and
`objective_class: 2` in the corresponding files.

## Default speed settings

The defaults target fast iteration on a T4 16 GB:

```text
model: yolov8s
base image size: 1024
multi-scale range: approximately 768-1280
training classes: person (0), no-helmet (2)
training batch: 6
training epochs: 100
early stopping patience: 10
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
- `configs/train_yolo26.yaml` — full YOLO26s training settings;
- `configs/tune_yolo26.yaml` — YOLO26-specific tuning profile;
- `.env` — private Azure credentials, never committed.
