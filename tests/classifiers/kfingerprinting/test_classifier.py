"""Tests for the KFingerprintingClassifier."""
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks
from lab.classifiers.kfingerprinting import KFingerprintingClassifier


@parametrize_with_checks([KFingerprintingClassifier])
def test_sklearn_compatiblity(estimator, check):
    """Test that the k-fingerprinting classifier is compatible with sklearn."""
    if check.func.__name__ == 'check_supervised_y_2d':
        pytest.skip("KFingerprintingClassifier does not support 2d labels.")
    check(estimator)
