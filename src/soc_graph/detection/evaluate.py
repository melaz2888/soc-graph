from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowEvaluation:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def evaluate_window_predictions(
    predicted_windows: list[bool],
    ground_truth_windows: list[bool],
) -> WindowEvaluation:
    if len(predicted_windows) != len(ground_truth_windows):
        raise ValueError("predicted and ground-truth windows must have the same length")

    tp = sum(pred and truth for pred, truth in zip(predicted_windows, ground_truth_windows, strict=True))
    fp = sum(pred and not truth for pred, truth in zip(predicted_windows, ground_truth_windows, strict=True))
    fn = sum((not pred) and truth for pred, truth in zip(predicted_windows, ground_truth_windows, strict=True))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return WindowEvaluation(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )
