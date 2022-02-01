"""Methods useful for analysis of results.
"""
import itertools
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.utils import check_array
from typing_extensions import Final

from . import metrics


def median_difference(frame: pd.DataFrame, level: str = "") -> pd.DataFrame:
    """Return the differences of the medians of each column.
    """
    result_columns = {}
    medians = frame.groupby(level).median()
    level_values = medians.index.get_level_values(level)

    for lhs, rhs in itertools.combinations(level_values, 2):
        result_columns[f"({lhs} - {rhs})"] = medians.loc[lhs] - medians.loc[rhs]
    return pd.DataFrame(result_columns).T


PRCurveResult = NamedTuple("PRCurveResult", [
    ("precision", np.ndarray), ("recall", np.ndarray),
    ("thresholds", np.ndarray)
])


# pylint: disable=too-many-arguments
def rprecision_recall_curve(
    y_true,
    probas_pred,
    ratio: float = 20,
    neg_label=-1,
    pos_labels=None,
) -> PRCurveResult:
    """Calculation of the precision recall scores for various thresholds
    in multiclass classification using rprecision.

    pos_labels correspond to the columns of probas_pred, assumed to be
    [0, n_classes] if not specified.

    Predictions falling below the decision thresholds are classified
    as neg_label.
    """
    y_true = check_array(y_true, dtype=int, ensure_2d=False)
    probas_pred = check_array(probas_pred)
    pos_labels = check_array(pos_labels, dtype=int, ensure_2d=False) \
        if pos_labels else np.arange(probas_pred.shape[1])
    assert len(pos_labels) == probas_pred.shape[1]
    assert neg_label not in pos_labels

    y_pred: Final = pos_labels[np.argmax(probas_pred, axis=1)]
    max_probabilities: Final = np.max(probas_pred, axis=1)

    thresholds = np.unique(np.around(max_probabilities, decimals=3))

    precision = np.zeros(len(thresholds) + 1)
    recall = np.zeros(len(thresholds) + 1)

    for i, threshold in enumerate(thresholds):
        below_threshold = (max_probabilities < threshold)

        # Dont modify y_pred since it is reused
        y_pred_thresh = y_pred.copy()
        y_pred_thresh[below_threshold] = neg_label

        precision[i] = metrics.rprecision_score(
            y_true, y_pred_thresh, ratio=ratio, zero_division=1,
            negative_class=neg_label)
        recall[i] = metrics.recall_score(
            y_true, y_pred_thresh, negative_class=neg_label)

        # Precision and recall will be zero if there are no true positives
        # if precision[i] == 0.0 and recall[i] == 0.0:
        #     raise RuntimeError(
        #         f"Precision and recall both zero at threshold {threshold}")

    precision[len(thresholds)] = 1
    recall[len(thresholds)] = 0
    return PRCurveResult(precision, recall, thresholds)
