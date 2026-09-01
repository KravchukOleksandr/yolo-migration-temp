from __future__ import annotations

import math

import numpy as np


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def roc_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    positives = targets == 1
    negatives = targets == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    if not positive_count or not negative_count:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[positives].sum())
    correction = positive_count * (positive_count + 1) / 2.0
    return (rank_sum - correction) / (positive_count * negative_count)


def average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    positive_count = int((targets == 1).sum())
    if not positive_count:
        return math.nan
    ordered_targets = targets[np.argsort(-scores, kind="mergesort")]
    true_positives = np.cumsum(ordered_targets == 1)
    precision = true_positives / np.arange(1, len(targets) + 1)
    return float(precision[ordered_targets == 1].sum() / positive_count)


def fpr_at_recall(
    scores: np.ndarray,
    targets: np.ndarray,
    target_recall: float = 0.95,
) -> tuple[float, float, float]:
    positive_count = int((targets == 1).sum())
    negative_count = int((targets == 0).sum())
    if not positive_count or not negative_count:
        return math.nan, math.nan, math.nan
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_targets = targets[order]
    true_positives = np.cumsum(ordered_targets == 1)
    false_positives = np.cumsum(ordered_targets == 0)
    ends = np.flatnonzero(np.r_[ordered_scores[:-1] != ordered_scores[1:], True])
    recall = true_positives[ends] / positive_count
    fpr = false_positives[ends] / negative_count
    eligible = np.flatnonzero(recall >= target_recall)
    if not len(eligible):
        return math.nan, math.nan, math.nan
    minimum = float(fpr[eligible].min())
    candidates = eligible[np.isclose(fpr[eligible], minimum, atol=1e-12)]
    best_recall = float(recall[candidates].max())
    candidates = candidates[np.isclose(recall[candidates], best_recall)]
    indices = ends[candidates]
    selected = int(indices[np.argmax(ordered_scores[indices])])
    return (
        float(false_positives[selected] / negative_count),
        float(ordered_scores[selected]),
        float(true_positives[selected] / positive_count),
    )


def binary_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    predictions = scores >= threshold
    positive = targets == 1
    negative = targets == 0
    tp = int(np.logical_and(predictions, positive).sum())
    fn = int(np.logical_and(~predictions, positive).sum())
    fp = int(np.logical_and(predictions, negative).sum())
    tn = int(np.logical_and(~predictions, negative).sum())
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    fpr95, threshold95, achieved = fpr_at_recall(scores, targets)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": _divide(tp + tn, len(targets)),
        "precision_no_helmet": precision,
        "recall_no_helmet": recall,
        "f1_no_helmet": _divide(2 * precision * recall, precision + recall),
        "fpr_helmet": _divide(fp, fp + tn),
        "roc_auc": roc_auc(scores, targets),
        "pr_auc": average_precision(scores, targets),
        "fpr_at_recall_95": fpr95,
        "threshold_at_recall_95": threshold95,
        "achieved_recall_at_95": achieved,
    }
