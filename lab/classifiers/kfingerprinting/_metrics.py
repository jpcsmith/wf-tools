"""Metrics for evaluating the performance of the classifier."""
from typing import (
    Tuple,
    Sequence,
)

BinarySequence = Sequence[int]


def true_positive_rate(y_true: Sequence, y_pred: Sequence) -> float:
    """Calculates the true positive rate binary classification results.
    Assumes that the negative class is -1.
    """
    assert set(y_true).issubset({1, -1})
    assert set(y_pred).issubset({1, -1})

    num_positive = sum(1 for val in y_true if val == 1)
    true_positive = sum(1 for t, p in zip(y_true, y_pred) if t == p == 1)
    try:
        return true_positive / num_positive
    except ZeroDivisionError:
        return 0


def false_positive_rate(y_true: Sequence, y_pred: Sequence) -> float:
    """Calculates the false positive rate binary classification results.
    Assumes that the negative class is -1.
    """
    assert set(y_true).issubset({1, -1})
    assert set(y_pred).issubset({1, -1})

    false_positives = 0
    true_negatives = 0
    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label == -1 and predicted_label == 1:
            false_positives += 1
        elif true_label == predicted_label == -1:
            true_negatives += 1
    try:
        return false_positives / (false_positives + true_negatives)
    except ZeroDivisionError:
        return 0


def make_binary(y_true: Sequence, y_pred: Sequence, neg_label,
                strict: bool = False) -> Tuple[BinarySequence, BinarySequence]:
    """Converts the multiclass ground truth and predictions into binary
    predictions where 'neg_label' is the negative class and is mapped
    to -1, and everything else is the positive class and are mapped to 1.

    When strict is true, a prediction for a positive label is correct if and
    only if it predicts the precise label. When strict is false, a positive
    label which is predicted as another positive label is still considered
    correct.

    Returns (y_true´, y_pred´)
    """
    def _map_predicted(true_label, pred_label) -> int:
        if true_label == pred_label:
            return 1 if pred_label != neg_label else -1

        if true_label == neg_label:
            # Since true != predicted, and the true label is the negative
            # label, we predict the positive label to keep the difference
            return 1
        # Both true and pred are positive labels, but they differ
        return -1 if strict else 1

    y_true_mapped = [1 if label != neg_label else -1 for label in y_true]
    y_pred_mapped = [_map_predicted(true, pred) for true, pred in zip(
        y_true, y_pred)]

    return y_true_mapped, y_pred_mapped
