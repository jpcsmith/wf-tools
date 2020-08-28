"""Additional classification metrics."""
import logging


def recall_score(y_true, y_pred, negative_class=-1):
    """Calculate the recall score, accounting for confusion among
    multiple positive classes.
    """
    pos_labels = (y_true != negative_class)

    n_true_positive = sum(pos_labels & (y_true == y_pred))
    n_positive = sum(pos_labels)

    return n_true_positive / n_positive


def rprecision_score(
    y_true, y_pred, ratio: float = 1.0, negative_class=-1
):
    """Calculate r-precision score for multiclass classification.

    The variables y_true and y_pred are the true and predicted labels
    respectively.  The variable ratio defines the expected number of
    samples in the negative class relative to the foreground class.

    See the paper:

        T. Wang, "High Precision Open-World Website Fingerprinting," in
        2020 IEEE Symposium on Security and Privacy (SP), Los Alamitos,
        CA, USA, 2020, pp. 231–246, doi: 10.1109/SP.2020.00015.

    for more information.
    """
    logger = logging.getLogger(__name__)

    pos_labels = (y_true != negative_class)
    pos_predictions = (y_pred != negative_class)

    n_true_positive = sum(pos_labels & (y_true == y_pred))
    logger.debug("n_true_positive: %d", n_true_positive)

    # Positive predictions which were not correct for positive classes
    n_wrong_positive = sum(pos_labels & pos_predictions & (y_true != y_pred))
    n_false_positive = sum(~pos_labels & pos_predictions)
    logger.debug("n_wrong_positive: %d, n_false_positive: %d",
                 n_wrong_positive, n_false_positive)

    n_positive = sum(pos_labels)
    n_negative = sum(~pos_labels)
    logger.debug("n_positive: %d, n_negative: %d", n_positive, n_negative)

    true_positive_rate = n_true_positive / n_positive
    wrong_positive_rate = n_wrong_positive / n_positive
    false_positive_rate = n_false_positive / n_negative

    if n_true_positive == n_wrong_positive == n_false_positive == 0:
        return 0

    logger.debug("r_%d-precision = %.3g / (%.3g + %.3g + %d * %.3g)",
                 ratio, true_positive_rate, true_positive_rate,
                 wrong_positive_rate, ratio, false_positive_rate)
    return true_positive_rate / (
        true_positive_rate + wrong_positive_rate + ratio * false_positive_rate)
