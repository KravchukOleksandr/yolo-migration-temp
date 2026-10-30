# YOLO Safety Detection

Compact offline YOLO training for an Azure VM with an NVIDIA T4 16 GB.
The default is **YOLO26s**, classes **0: person** and **2: no-helmet-head**,
with Optuna tuning followed by full training and validation.

All training data, including the prepared public additions, comes from your
Azure train/val containers. The main workflow does **not** need a local
`external_dataset/` directory or download public datasets separately.

## Quick start after installation

Run from the repository root with your virtual environment active and local
`weights/yolo26s.pt` present:

```bash
set -a
source .env
set +a

make download
make check
make start
tail -f "$(cat runs/LATEST.txt)/pipeline.log"
```

`make start` launches the whole experiment in the background. You can close
the terminal or disconnect SSH after it reports a successful start. Keep the
VM running. `make` and `make yolo26` are equivalent aliases.

The rest of this guide explains installation, settings, and outputs.

## 1. Install

Use Python 3.10–3.12 and run all commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate

export PIP_PROXY='http://user:password@proxy.example:3128'
python -m pip install --proxy "$PIP_PROXY" --upgrade pip
python -m pip install --proxy "$PIP_PROXY" -e '.[dev,tune]'
```

The same installation command updates an existing environment after pulling
repository changes; recreating the environment is normally unnecessary.
Keep proxy credentials out of Git.

Check GPU availability:

```bash
nvidia-smi
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "CUDA GPU is unavailable"; print(torch.cuda.get_device_name(0))'
```

Do not start the T4 profile if the CUDA check fails. It requires an NVIDIA
driver and a CUDA-enabled PyTorch installation through your approved package
source. The repository does not install system drivers.

## 2. Copy local weights

Copy the pretrained checkpoint yourself:

```text
weights/yolo26s.pt
```

Verify its location:

```bash
ls -lh weights/yolo26s.pt
```

Other modes require their matching local file, such as `weights/yolo26n.pt`
or `weights/yolov8s.pt`. A path beginning with `/weights/` refers to the
filesystem root, not this repository. Use `weights/...` or a correct absolute
path. Checkpoints under `weights/` are ignored by Git.

Training uses local model files; missing files must not trigger downloads.
Internet access is needed for installation and Azure data retrieval, not the
training workflow itself.

Automatic Ultralytics asset downloads, package auto-installation, update checks,
and remote tracking integrations are disabled. Plots use a bundled local font.
AMP remains enabled, but its reference-model compatibility check may be skipped
with a warning when the reference weights are absent. No extra reference
checkpoint is required. If AMP causes non-finite losses, disable it in both
train and tune settings and investigate before running a long search.

## 3. Configure and download Azure data

Create `.env` once; do not overwrite an existing file containing your settings:

```bash
cp .env.example .env
```

Fill in:

```bash
AZURE_STORAGE_CONNECTION_STRING='paste-full-connection-string-here'
AZURE_TRAIN_CONTAINER='train-container-name'
AZURE_VAL_CONTAINER='val-container-name'
```

The two containers are already split. Include public train additions only in
the train container and public val additions only in the val container.
Do not upload the same additions again through the optional local-data workflow.

Load the variables and download:

```bash
set -a
source .env
set +a
make download
```

Containers may contain flat image/TXT pairs or source subfolders:

```text
factory/frame_001.jpg
factory/frame_001.txt
sfchd/frame_002.jpg
sfchd/frame_002.txt
```

Each image needs a same-stem YOLO TXT file; an empty TXT represents a genuinely
empty image. Nested source folders are preserved in the local layout:

```text
data/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

`configs/data.yaml` points to these directories. The source IDs are:

```yaml
names:
  0: person
  1: helmet-head
  2: no-helmet-head
```

Unchanged downloads are skipped using remote ETags and file sizes; changed
files are refreshed. An old download without sync state is refreshed once.
Download is not a destructive mirror: stale local samples absent from the
containers are reported, not silently deleted.
Do not change the containers while a download is running. If old local samples
are reported, archive the previous local dataset or download into a clean
directory, for example `python -m yolo_factory.download --output data_new`,
then change `configs/data.yaml: path` to `../data_new`. Do not mix old and new
splits by copying files indiscriminately.

If public examples are included in the val container, the default validation
metrics describe that combined set, not factory-only quality. Use a separate
dataset YAML with your factory-only holdout for the final deployment assessment.

## 4. Check and launch

Check dependencies, matching configs, and local weights without training:

```bash
make check
```

Start the complete detached pipeline:

```bash
make start
```

It runs these stages automatically:

1. Audit local train/val images and labels.
2. Select up to 10,000 train and 2,000 val images for tuning.
3. Run 12 Optuna trials, up to 15 epochs each.
4. Copy the winning searched and fixed hyperparameters to the final config.
5. Train on the full train split for up to 100 epochs.
6. Validate the final `best.pt` on the full val split at `imgsz=1280`.

The final training starts from the local pretrained checkpoint, not the
subset-trained trial checkpoint. Early stopping may finish a run before its
epoch limit. The tuning objective is class-2 mAP50–95, not training loss.

**Seeing 15 epochs in the log means a tuning trial is running.**
`configs/train_yolo26.yaml: train.epochs` controls final training;
`configs/tune_yolo26.yaml: epochs` controls each trial.
The initial 4 trials explore; subsequent trials use Optuna TPE observations.

To audit without starting training:

```bash
make audit
```

The audit writes `runs/audit/summary.json`, `train.csv`, and `val.csv`.
It reports progress every 1,000 images. Fix reported fatal errors before
training. `make pipeline` runs the same pipeline in the foreground, so use
`make start` when the terminal may be closed.

## 5. Follow progress and find results

```bash
tail -f "$(cat runs/LATEST.txt)/pipeline.log"
watch -n 2 nvidia-smi
```

Run these in separate terminals. `Ctrl+C` stops log following, not training.
The launcher also prints the exact log path and PID for that experiment.

```text
runs/
├── LATEST.txt
├── audit/
├── experiments/
│   └── yolo26-<run-id>/
│       ├── pipeline.log
│       ├── pipeline.pid
│       ├── resolved-tune.yaml
│       ├── resolved-train.yaml
│       ├── tune-data.yaml
│       ├── .manifests/              # experiment-specific tuning subset
│       ├── tune/
│       │   ├── trial-000/
│       │   ├── trial-001/
│       │   └── best.json
│       ├── train/
│       │   ├── args.yaml
│       │   ├── results.csv
│       │   ├── results.png
│       │   └── weights/
│       │       ├── best.pt
│       │       └── last.pt
│       └── validation/
│           ├── 1280/
│           └── summary.json
├── validation/
└── runtime/
```

`train/results.csv` records epoch metrics during training. Standard
Ultralytics plots and confusion/PR curves are enabled for final training and
validation; some plots appear only after a stage finishes. Tuning uses
`plots: false` to reduce overhead, so do not expect the same PNG files in trials.

The model to keep is `train/weights/best.pt`. `last.pt` is the recovery
checkpoint. Generated configs, subsets, caches, and outputs stay out of Git;
they are not written into `configs/`.

To stop the repository's running pipelines, tuning, training, and validation:

```bash
make stop
```

This requests termination, then force-kills remaining matching processes after
three seconds. Unsaved progress in the current epoch is lost. Optuna study
state is currently in memory: interrupted searches do not resume automatically,
and `best.json` is written only after the search completes.

## 6. Change model, classes, or tuning budget

Edit the matching pair:

| File | Purpose |
|---|---|
| `configs/data.yaml` | Downloaded train/val paths and original class names |
| `configs/train_yolo26.yaml` | Full training model, batch, epoch limit, augmentations |
| `configs/tune_yolo26.yaml` | Trial count, trial epochs, objective, fixed/search parameters |
| `configs/train.yaml`, `configs/tune.yaml` | Optional legacy YOLOv8 profile |

Default full-training settings:

```yaml
model: s
family: yolo26
train:
  imgsz: 1280
  rect: true
  multi_scale: 0.0
  batch: 6
  epochs: 100
  patience: 10
  classes: [0, 2]
  single_cls: false
```

The fixed image scale and rectangular batching preserve aspect ratio.
1280×720 camera frames typically form 1280×736 stride-aligned tensors.
This is not a forced tensor shape for every source: square images stay square.
Scale/translation augmentation remains enabled; it is different from
`multi_scale`, which is disabled. Mosaic, MixUp, and Copy-Paste are disabled.

The YOLO26 profile uses MuSGD, AMP, disk image caching, and 8 data workers.
Tuning uses batch 8; full training uses batch 6. These are starting settings,
not a guarantee against OOM for every GPU/software combination. Reduce both
configured batches if needed. Validation always requires a positive integer
batch size; `-1` is not supported there.

For another model size, set `model: n`, `s`, or `m` in **both** configs,
copy the matching weights, and adjust batches. For classes, keep original IDs:

- Default: both configs use `classes: [0, 2]`, `single_cls: false`.
- Person only: both use `classes: [0]`, `single_cls: false`;
  set `objective_class: 0` in the tune config.
- Both classes, tune for person: keep `[0, 2]`, set `objective_class: 0`.

Leave all three names in the data YAML. Class 1 is filtered in memory and
class 2 stays class 2; source TXT files are not rewritten.
Each selected objective class must be represented in the tuning validation
subset. Filtering classes does not make partially labelled data safe:
person-only public images with unlabelled no-helmet heads require a separate
partial-label strategy, which this repository does not currently implement.

For custom configs:

```bash
make start \
  DATA_CONFIG=configs/data.yaml \
  TRAIN_CONFIG=configs/train_custom.yaml \
  TUNE_CONFIG=configs/tune_custom.yaml
```

The family, model size, classes, and `single_cls` settings must agree.
Train and tune may have different batch sizes and epoch limits.

The original YOLOv8 profile is still available explicitly:

```bash
make yolov8
```

It uses `configs/train.yaml` and `configs/tune.yaml`, with their own legacy
image-size and augmentation settings. `make start` now always defaults to
YOLO26, not YOLOv8.

## 7. Train from a chosen trial instead of tuning again

Use a completed trial directory or its `args.yaml`:

```bash
make yolo26-from-trial \
  TRIAL=runs/experiments/yolo26-<run-id>/tune/trial-004
```

This skips the search and runs full training plus validation in a new detached
experiment. It copies the selected trial's searched/fixed hyperparameters,
but keeps the full-training data, resolution, batch, epoch limit, and patience.
It does not resume the trial checkpoint.

If the trial used custom configs, pass the matching `TRAIN_CONFIG` and
`TUNE_CONFIG` too. The provenance is saved as `selected-trial.yaml`;
the effective final settings are in `resolved-train.yaml`.

To resume interrupted full training from its latest saved epoch instead:

```bash
python -m yolo_factory.train \
  --config runs/experiments/yolo26-<run-id>/resolved-train.yaml \
  --resume runs/experiments/yolo26-<run-id>/train/weights/last.pt
```

This resume command runs in the foreground and does not run final standalone
validation automatically. Keep the terminal open and validate the resulting
`best.pt` afterward.

## 8. Validate an existing model without training

```bash
make validate \
  WEIGHTS=/absolute/path/to/best.pt \
  VAL_OUTPUT=runs/validation/my-model
```

The default evaluates classes `0 2` at 1280 with batch 8. The explicit command
is:

```bash
python -m yolo_factory.validate \
  --weights /absolute/path/to/best.pt \
  --data configs/data.yaml \
  --classes 0 2 \
  --imgsz 1280 \
  --batch 8 \
  --output runs/validation/my-model
```

`--weights` must be a local trained checkpoint. `--data` chooses the val
split; `--classes` filters evaluated source IDs; `--imgsz` sets inference
resolution; `--batch` must be positive; `--output` selects the result folder.
Use a new output folder for each comparison.

For person only, add `VAL_CLASSES=0` to the Make command.
For your separate factory holdout, use `DATA_CONFIG=configs/data_factory_val.yaml`
after creating that dataset YAML. To compare resolutions, use
`VAL_IMGSZ="640 1280"`; the main pipeline still validates at 1280 only.

## Optional tools and repository checks

`make subset` then `make tune` runs standalone tuning with the default
YOLO26 profile. `make train` trains the current full-training config only:
it does **not** automatically apply a previous standalone `best.json`.
Use `make start` or `make yolo26-from-trial` for automatic parameter transfer.

For optional local public-data preparation, see
[external dataset preparation](docs/external_dataset.md). Skip it when the
prepared data is already in Blob. Local external validation can still be run
explicitly, but it is not a dependency of the default pipeline.

```bash
make test
make lint
```

Make targets use `python3` from the active environment. Override it when
needed, for example `make start PYTHON=/path/to/venv/bin/python`.
