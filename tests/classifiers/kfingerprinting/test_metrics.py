"""Tests related to the metrics in kfingerprinting."""
import pytest

from lab.classifiers.kfingerprinting import make_binary, false_positive_rate


@pytest.mark.parametrize('strict', [True, False])
def test_make_binary_simple(strict: bool):
    """It should convert unmonitored labels to -1 and monitored to 1."""
    binary_true, binary_pred = make_binary(
       y_true=['google.com', 'facebook.com', 'background'],
       y_pred=['google.com', 'facebook.com', 'background'],
       neg_label='background', strict=strict)
    assert binary_true == binary_pred == [1, 1, -1]


def test_make_binary_no_strict():
    """It should treat each monitored page as equivalent."""
    binary_true, binary_pred = make_binary(
        y_true=['facebook.com', 'background', 'mail.google.com'],
        y_pred=['facebook.com', 'background', 'pintrest.com'],
        neg_label='background', strict=False)
    assert binary_true == binary_pred == [1, -1, 1]


def test_make_binary_strict():
    """It should require predictions for monitored pages to be precise."""
    binary_true, binary_pred = make_binary(
        y_true=['facebook.com', 'background', 'mail.google.com', 'background'],
        y_pred=['facebook.com', 'background', 'pintrest.com', 'google.com'],
        neg_label='background', strict=True)
    assert binary_true == [1, -1, 1, -1]
    assert binary_pred == [1, -1, -1, 1]


def test_false_positive_rate():
    """It should correctly report the false positive rate."""
    rate = false_positive_rate(y_true=[1, 1, -1, -1, -1, -1, -1],
                               y_pred=[1, -1, -1, 1, 1, -1, 1])
    assert rate == 0.6
