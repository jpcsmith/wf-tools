"""Tests for the KFingerprintingClassifier."""
from sklearn.utils.estimator_checks import parametrize_with_checks
from lab.classifiers.kfingerprinting import KFingerprintingClassifier


@parametrize_with_checks([KFingerprintingClassifier])
def test_sklearn_compatiblity(estimator, check):
    """Test that the k-fingerprinting classifier is compatible with sklearn."""
    check(estimator)
