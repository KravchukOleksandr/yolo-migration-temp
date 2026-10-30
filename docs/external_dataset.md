# Optional external dataset preparation

This is an optional preparation workflow, not a prerequisite for training.
If SFCHD/SHEL5K have already been uploaded into the Azure train/val containers,
use the main README and `configs/data.yaml`; do not add them a second time via
`configs/data_external.yaml`. The default pipeline never requires this local export.

If you intentionally keep factory data in Blob and public additions local,
use `make start DATA_CONFIG=configs/data_external.yaml`. That profile uses
combined training data and factory-only validation. To evaluate the local
public validation set separately after training, run:

```bash
make validate WEIGHTS=/path/to/best.pt \
  DATA_CONFIG=configs/data_external_val.yaml \
  VAL_OUTPUT=runs/validation/external
```


The public datasets are not downloaded automatically. VIRAT requires accepting
its usage agreement, NightSurveillance is academic-use-only, and SFCHD is
distributed through its authors' download link. Use only sources whose terms
you have accepted:

- [SFCHD paper](https://arxiv.org/abs/2306.02098) and
  [official repository](https://github.com/lijfrank/SFCHD-SCALE);
- SHEL5K v4: `https://data.mendeley.com/datasets/9rcv8mm682/4`;
- VIRAT: `https://viratdata.org/`;
- NightSurveillance: `https://github.com/xiaowang1516/NightSurveillance`.

Download the SFCHD dataset ZIP from the official repository's Dataset
Acquisition section. Extract its `QY_final_dataset` directory at the exact
configured root:

```text
external_dataset/raw/
├── sfchd/
│   └── unpacked/
│       └── QY_final_dataset/
│           ├── images/
│           └── annotations/
│               ├── train.json
│               └── val.json
├── shel5k/                    # extract the complete Mendeley v4 archive here
├── virat/
└── night_surveillance/
```

The configured SFCHD root is
`external_dataset/raw/sfchd/unpacked/QY_final_dataset`. SHEL5K Pascal VOC XML
and its images are discovered recursively below `external_dataset/raw/shel5k`;
preserve the archive structure and filenames.

### Annotation trust policy

SFCHD COCO and SHEL5K Pascal VOC annotations are accepted as native source
annotations. For SFCHD, the paper reports that six professional safety
inspectors annotated the instances and three independent graduate students
reviewed every image and annotation. This is source-level evidence of
completeness, not a claim that this repository manually verified every box.
Native boxes therefore normally keep `verified: false` while their frame may
still be eligible for joint training under the declared source policy.

SFCHD `head` is mapped to class `2` (`no-helmet-head`), `helmet` is retained
only as auxiliary evidence, and `blur_head` is treated as unresolved. Frames
with unresolved head status, disputed duplicate annotations, invalid geometry,
or another explicit exclusion reason are not admitted to the joint-ready YOLO
view. Their source IDs and reasons are recorded in
`manifests/excluded.jsonl`; they are not silently converted to negative
examples.

SHEL5K receives two additional person-completeness safeguards before quota
selection. Four known mirrored/composited images are excluded explicitly:
`hard_hat_workers1130.png`, `hard_hat_workers479.png`,
`hard_hat_workers4912.png`, and `hard_hat_workers584.png`. The builder also
rejects a frame when a strong head annotation (`no_helmet` or
`helmet_present`) has less than 50% of its area inside every annotated person
box. Standalone `helmet_indicator` boxes are auxiliary objects and do not
trigger this check.

A deterministic 24-frame spot-check across both sources and generated splits
found no confident no-helmet box on a visibly helmeted head; it also exposed
the four SHEL5K composite cases above. This was a targeted check, not full
manual verification. Remaining accepted native annotations may still contain
source errors, so review `previews/`, `manifests/excluded.jsonl`, and the final
factory validation results.

`verified: true` on a head decision is reserved for explicit manual review. A
native source may provide a normalized `reviewed_manifest.jsonl` at its
configured source root; matching records override the native COCO or VOC
records. Keep the native `source_id` values, and mark every manually resolved
head decision explicitly as verified. Unset or false verification remains
unresolved.

### Video-aware splitting

The SFCHD paper documents a 4:1 official split but does not document a
video-, sequence-, or camera-aware partition. The released split contains
adjacent numbered images from the same filename sequence on opposite sides.
The builder therefore ignores that split by default, infers conservative
capture-sequence groups from filenames, and keeps every inferred group in only
one generated split.

These are inferred groups, not authenticated video IDs or independent scenes:
SFCHD removes timestamps and does not document the filename tokens. Sharing a
camera background across generated train and validation is consequently not,
by itself, evidence of a same-video leak. For a stricter cross-camera
experiment, change only the SFCHD source in `configs/external_dataset.yaml`:

```yaml
sources:
  sfchd:
    group_mode: camera
```

Camera mode keeps each inferred camera/IP prefix in one split, but measures a
different and harder cross-scene generalization problem. The default
`group_mode: sequence` is the practical video-adjacency safeguard. Neither mode
turns the public external validation into the final factory benchmark.

For VIRAT, first import the accepted DIVA/KPF release and its matching videos:

```bash
make external-import-virat VIRAT_ROOT=/path/to/accepted/virat-release
```

The importer extracts temporally separated frames, preserves track IDs, and
writes `external_dataset/raw/virat/manifest.jsonl`. A complete video is always
one split group.

NightSurveillance accepts Pascal VOC, COCO, or a normalized `manifest.jsonl`.
The normalized record format is also the boundary used for reviewed or
auto-assisted head annotations:

```json
{
  "image": "frames/camera-01/00000120.jpg",
  "source_id": "camera-01:120",
  "width": 1920,
  "height": 1080,
  "group_id": "camera-01",
  "camera_id": "camera-01",
  "frame_index": 120,
  "person_annotation_complete": true,
  "no_helmet_annotation_complete": false,
  "objects": [
    {"role": "person", "xyxy": [100, 80, 220, 500], "verified": true}
  ]
}
```

Build the 80/20 grouped export:

```bash
make external-dataset
```

The command is strict: a missing or invalid required source makes the build
fail. To intentionally build and report only the currently available sources:

```bash
yolo-external-dataset \
  --config configs/external_dataset.yaml \
  --allow-partial
```

The builder creates a staged replacement and restores the previous managed
export when a normal build or publication error is raised. Source files below
`external_dataset/raw/` are never modified. This is not a power-loss atomicity
guarantee and rebuilding is not supported while training or validation is
reading the export; stop those processes first.

Train person-box quota targets and the grouped validation fraction are defined
in `configs/external_dataset.yaml`. Selection uses whole frames, removes exact
duplicates before quota counting, compares file hashes against factory train/val
when available, and also checks decoded-pixel duplicates among public candidates.
These checks do not detect all crops, re-encodings, or correlated scenes. The
builder keeps each configured group entirely in one split. Because
whole frames and groups are indivisible, use the generated `stats.json` rather
than quota targets as the record of achieved counts.

The visible export keeps each image beside its YOLO label:

```text
external_dataset/
├── train/<source>/frame.jpg + frame.txt
├── val/<source>/frame.jpg + frame.txt
├── manifests/
│   ├── candidates.jsonl
│   ├── candidates_<source>.jsonl
│   ├── excluded.jsonl
│   ├── manifest.jsonl
│   ├── objects.jsonl
│   └── pending_review.csv
├── previews/
├── provenance_and_licenses/
├── qa_report.md
└── stats.json
```

`candidates.jsonl` records normalized source candidates before final export
selection. Its source-specific companions, such as
`candidates_sfchd.jsonl`, are valid inputs for the review tool.
`excluded.jsonl` records exclusions with machine-readable reasons.
`manifest.jsonl` and `objects.jsonl` describe the selected images and objects;
`stats.json` is the count and readiness summary.

Class IDs are never compacted: `0` is person and `2` is a head without a
safety helmet. A head wearing a helmet is only auxiliary metadata and is not
exported as a target.

Only frames whose person and no-helmet annotations are complete under either
the accepted native-source policy or explicit manual review are linked into
`external_dataset/.yolo/`. Unknown or disputed class-2 status cannot silently
become training background.

For example, apply native SFCHD corrections and write the source-root overlay
that the next build will prefer:

```bash
make external-apply-review \
  MANIFEST=external_dataset/manifests/candidates_sfchd.jsonl \
  REVIEWS=/path/to/sfchd-head-review.jsonl \
  REVIEW_OUTPUT=external_dataset/raw/sfchd/unpacked/QY_final_dataset/reviewed_manifest.jsonl
```

Each review record contains the source frame ID, whether every visible head was
resolved, and the reviewed heads:

```json
{
  "source_id": "replace-with-sfchd-source-id.jpg",
  "complete": true,
  "person_review_complete": true,
  "reviewer": "manual",
  "heads": [
    {
      "status": "helmet_absent",
      "xyxy": [120, 80, 145, 110],
      "person_index": 0,
      "verified": true
    },
    {
      "status": "head_not_visible",
      "person_index": 1,
      "verified": true
    }
  ]
}
```

Supported statuses are `helmet_absent`, `helmet_present`,
`head_status_unknown`, and `head_not_visible`. `complete: true` requires one
verified decision for every annotated person. `person_review_complete` is
optional and must be set only when the reviewer also checked that every visible
person is annotated; omit it to preserve the source person-completeness value.
Use zero-based `person_index` when person objects have no track IDs; use
`person_track_id` only when the manifest provides track IDs for every person.
An unverified positive is stored as unknown and cannot make the frame
joint-ready. The reviewed output is named `reviewed_manifest.jsonl` and is
preferred automatically by the dataset builder.

