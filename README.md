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

`make dataset` loads `.env` automatically. Load the variables manually only
when running an individual download command:

```bash
set -a
source .env
set +a
```

## 4. Prepare the dataset

Both Azure containers may be flat. Every image must have a `.txt` label with
the same stem:

```text
frame_0001.jpg
frame_0001.txt
frame_0002.png
frame_0002.txt
```

Download both containers, audit the YOLO dataset, and generate verifier crops
with one command:

```bash
make dataset
```

The stages run sequentially. A failed download or fatal audit error prevents
crop preparation from starting.

The command runs in the background, so the terminal or SSH session may be
closed. Follow all three stages with:

```bash
tail -f "$(cat runs/LATEST_DATASET.txt)/dataset.log"
```

Wait for `Dataset preparation completed.` before starting training.

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

The same command also creates:

```text
data/verifier/
├── train/{helmet,no_helmet}/
├── val/{helmet,no_helmet}/
├── manifest.csv
└── summary.json
```

## 5. Rerun an individual dataset stage

The complete `make dataset` command is the normal path. These targets are
available when only one stage must be repeated:

```bash
make download          # Azure download; environment must already be loaded
make audit
make verifier-prepare  # rebuild crops from the local YOLO dataset
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

Stop every YOLO Factory dataset preparation, tuning trial, training run, and
validation started by this repository:

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
│       ├── resolved-tune.yaml       # normal tuning mode
│       ├── resolved-train.yaml
│       ├── selected-trial.yaml      # selected-trial mode
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

### Train from a selected tuning trial

Use this mode when a particular trial is preferable to the Optuna winner. It
skips a new tuning search, starts full training from the local `yolo26s.pt`, and
then runs the standard validation at 640, 960, and 1280:

```bash
make yolo26-from-trial \
  TRIAL=runs/experiments/yolo26-<experiment-id>/tune/trial-004
```

The trial directory or its `args.yaml` file may be passed as `TRIAL`. The
command runs detached, updates `runs/LATEST.txt`, and writes everything to a
new `runs/experiments/yolo26-trial-<experiment-id>/` directory. Follow it with:

```bash
tail -f "$(cat runs/LATEST.txt)/pipeline.log"
```

The pipeline copies the actual trial values declared in the matching tune
configuration's `search_space` and `fixed` sections. It deliberately keeps the
full-training dataset, image size, batch size, epoch limit, and early stopping
settings from the train configuration. It does not continue from the
subset-trained trial checkpoint.

To use custom configuration files that produced the selected trial:

```bash
make yolo26-from-trial \
  TRIAL=/path/to/trial-004 \
  TRAIN_CONFIG=configs/train_custom.yaml \
  TUNE_CONFIG=configs/tune_custom.yaml
```

Use the tune configuration that created the trial. Before starting, the
pipeline verifies its model, selected classes, `single_cls` mode, and presence
of every searched parameter. The exact source and promoted values are saved in
`selected-trial.yaml`; the effective full configuration is saved in
`resolved-train.yaml`.

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

## No-helmet verifier experiment

This optional pipeline trains the small random-initialized `HelmetMicroNeXt`
classifier and evaluates it as a hard veto after an existing YOLO detector.
It uses only the factory dataset; public datasets are not mixed into this
baseline experiment.

The source IDs stay unchanged. Class `1` supplies helmet crops, class `2`
supplies no-helmet crops, and the verifier's internal binary labels are written
only to its generated manifest. The original images and YOLO labels are never
modified.

Copy the existing trained detector to the default local path and start the
complete detached experiment:

```bash
make dataset
cp /absolute/path/to/yolo/best.pt weights/detector.pt
make verifier
```

The checkpoint is not modified. To keep it elsewhere, override the default:

```bash
make verifier DETECTOR=/absolute/path/to/yolo/best.pt
```

The command requires the prepared `data/verifier/manifest.csv`, runs 200
resumable Optuna trials, trains the final classifier, validates it, and compares
baseline YOLO with the cascade at 640, 960, and 1280. It does not download or
rebuild data. You may close the terminal after the launcher returns.
Follow progress with:

```bash
tail -f "$(cat runs/LATEST_VERIFIER.txt)/pipeline.log"
```

Stop YOLO and verifier jobs started by this repository with `make stop`.

After the experiment finishes, repeat baseline/cascade validation for the
latest verifier with:

```bash
make verifier-validate
```

It uses `weights/detector.pt`, the latest `train/last.pt`, and the calibrated
threshold stored in that checkpoint. Paths can be overridden when needed:

```bash
make verifier-validate \
  DETECTOR=/path/to/yolo.pt \
  VERIFIER_MODEL=/path/to/verifier/last.pt \
  VERIFIER_OUTPUT=runs/verifier/manual-validation
```

### Dataset and split policy

Crop preparation follows the fixed experiment rules:

- expand each class `1` or `2` box by `1.25` around its center;
- reject an expanded side below 16 pixels or any out-of-image crop;
- accept width/height ratios from `0.72` through `1.39`;
- reject opposite-class overlap above 10% and same-class union overlap above
  25% of the crop;
- round outward only after all checks and store lossless PNG crops.

Generated crops, `manifest.csv`, and rejection counts are stored under
`data/verifier/`. Training reads the manifest, so unrelated stale files are
never included.

All boxes from one source image remain in the same internal split. Ten percent
of source training images are reserved for calibration before tuning. Optuna
uses a separate fit/tune split inside the remaining training data. The official
camera-separated `val` set is not used to choose hyperparameters, epochs, or
the operating threshold.

The final model is the last epoch selected by the tuning trial. Its hard-veto
threshold is chosen on the held-out calibration split for at least 95%
no-helmet recall. Classifier metrics are then reported once on the official
validation crops.

### Cascade logic

The verifier does not multiply or average unrelated confidence values. For a
YOLO class `2` proposal:

```text
verifier probability >= calibrated threshold  -> keep original YOLO confidence
verifier probability <  calibrated threshold  -> remove the proposal
```

Person detections and unclassifiable boundary crops are left unchanged. Since
surviving detections keep their original YOLO confidence, Ultralytics computes
the usual precision, recall, mAP50, mAP50-95, PR curves, confusion matrices,
and per-class results without a synthetic fused score.

### Safe proposal sweep for existing weights

Use this mode to test an existing detector and verifier without downloading
data, tuning, or training:

```bash
make verifier-sweep \
  DETECTOR=/absolute/path/to/yolo.pt \
  VERIFIER_MODEL=/absolute/path/to/verifier/last.pt
```

The command runs in the background. Follow it with:

```bash
tail -f "$(cat runs/LATEST_VERIFIER_SWEEP.txt)/sweep.log"
```

For each configured resolution, the sweep:

1. Recreates the held-out calibration image split from the original train set.
2. Runs YOLO and classifies its actual class `2` proposal crops.
3. Checks that a zero verifier threshold exactly reproduces baseline metrics.
4. Calculates candidate thresholds for several TP-retention targets.
5. Selects a threshold that retains at least 99.5% of baseline YOLO true
   positives at every IoU level from 0.50 through 0.95.
6. Evaluates only that selected threshold on the official validation set.

The official validation set is never used to select the threshold. Change the
calibration target only when required:

```bash
make verifier-sweep \
  DETECTOR=/path/to/yolo.pt \
  VERIFIER_MODEL=/path/to/verifier.pt \
  SAFE_TP_RETENTION=0.999
```

Results are stored under:

```text
runs/verifier-sweeps/<sweep-id>/
├── sweep.log
├── summary.json
└── <resolution>/
    ├── calibration-proposals.npz
    ├── threshold-sweep.csv
    ├── calibration-baseline/
    ├── calibration-threshold-zero/
    ├── baseline/
    └── cascade/
```

`threshold-sweep.csv` reports proposal rejection and TP retention on
calibration data. `summary.json` contains the selected per-resolution
thresholds, the zero-threshold sanity result, standard YOLO metrics, per-class
metrics, and baseline-to-cascade deltas.

Use `last.pt` from this repository for a clean calibration split. Because an
earlier official-validation result motivated this new experiment, treat that
validation set as development data and confirm any production decision on new
camera footage.

### Configuration and outputs

Edit `configs/verifier.yaml` to change the trial count, searched epoch range,
optimizer ranges, augmentation ranges, batches, or validation resolutions.
The default search includes epochs from 20 to 120 and uses FPR at 95% recall as
its primary objective. Ties prefer higher achieved recall, then ROC-AUC and
PR-AUC.

The newest experiment path is stored in `runs/LATEST_VERIFIER.txt`:

```text
runs/verifier/<experiment-id>/
├── pipeline.log
├── resolved-config.yaml
├── tune/
│   ├── study.db
│   ├── trials.csv
│   ├── best.json
│   └── trials/trial-XXXX/{history.csv,summary.json}
├── train/
│   ├── best.pt
│   ├── last.pt
│   ├── history.csv
│   ├── summary.json
│   └── validation.json
└── yolo-validation/
    ├── 640/{baseline,cascade}
    ├── 960/{baseline,cascade}
    ├── 1280/{baseline,cascade}
    └── summary.json
```

`last.pt` is the deployment candidate because the epoch count itself is tuned.
`best.pt` is retained only as a diagnostic checkpoint from the calibration
history. `train/validation.json` contains classifier FPR@R95, the selected
threshold, ROC-AUC, PR-AUC, accuracy, precision, recall, and F1.

To rerun only cascade validation with an explicit classifier threshold:

```bash
yolo-verifier-validate \
  --config configs/verifier.yaml \
  --detector /absolute/path/to/yolo/best.pt \
  --verifier /absolute/path/to/verifier/last.pt \
  --threshold 0.95 \
  --output runs/verifier/manual-validation
```
