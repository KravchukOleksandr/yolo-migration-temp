from pathlib import Path

import numpy as np

from yolo_factory.verifier_data import VerifierSample, group_holdout_split
from yolo_factory.verifier_metrics import (
    fpr_at_recall,
    proposal_gate_metrics,
    threshold_for_tp_retention,
)
from yolo_factory.verifier_model import HelmetMicroNeXt, trainable_parameters
from yolo_factory.verifier_prepare import union_area


def test_verifier_model_size() -> None:
    assert trainable_parameters(HelmetMicroNeXt()) == 17_441


def test_fpr_at_recall_uses_no_helmet_as_positive() -> None:
    scores = np.asarray([0.99, 0.90, 0.80, 0.10])
    targets = np.asarray([1, 0, 1, 0])

    fpr, threshold, recall = fpr_at_recall(scores, targets, 0.95)

    assert fpr == 0.5
    assert threshold == 0.8
    assert recall == 1.0


def test_overlap_area_is_a_union() -> None:
    rectangles = [(0.0, 0.0, 2.0, 2.0), (1.0, 0.0, 3.0, 2.0)]

    assert union_area(rectangles) == 6.0


def test_proposal_threshold_preserves_every_iou_level() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.2])
    true_positive = np.asarray(
        [
            [True, True],
            [True, False],
            [False, True],
            [False, False],
        ]
    )

    threshold = threshold_for_tp_retention(scores, true_positive, 1.0)
    metrics = proposal_gate_metrics(scores, true_positive, threshold)

    assert threshold == 0.7
    assert metrics["minimum_tp_retention"] == 1.0
    assert metrics["false_proposal_rejection_iou50"] == 0.5


def test_source_image_cannot_cross_internal_split(tmp_path: Path) -> None:
    samples = [
        VerifierSample(
            tmp_path / f"{source_index}-{label}.png",
            label,
            f"camera/frame-{source_index}.jpg",
            "train",
        )
        for source_index in range(30)
        for label in (0, 1)
    ]

    fit, holdout = group_holdout_split(samples, 0.3, 4)

    assert {sample.source_image for sample in fit}.isdisjoint(
        sample.source_image for sample in holdout
    )
