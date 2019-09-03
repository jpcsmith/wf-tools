"""Metrics for evaluating the performance of the classifier."""
from typing import (
    Tuple,
    Sequence,
)

BinarySequence = Sequence[int]


def make_binary(y_true: Sequence, y_pred: Sequence, neg_label,
                strict: bool = False) -> Tuple[BinarySequence, BinarySequence]:
    """Converts the multiclass ground truth and predictions into binary
    predictions where 'neg_label' is the negative class and is mapped
    to -1, and everything else is the positive class and are mapped to 1.

    When strict is true, a prediction for a positive label is correct if and
    only if it predicts the precise label. When strict is false, a positive
    label which is predicted as another positive label is still considered
    correct.

    Returns (y'_true, y'_pred)
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
