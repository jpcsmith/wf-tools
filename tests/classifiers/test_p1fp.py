"""Tests for the classifiers of p1-FP."""
import pytest
import numpy as np

from lab.classifiers.p1fp import KerasP1FPClassifierC
try:
    from lab.classifiers.p1fp import P1FPClassifierC
except ImportError:
    USE_TFLEARN = False
else:
    USE_TFLEARN = True


if USE_TFLEARN:
    @pytest.mark.slow
    def test_p1fpclassifierc(train_test_sizes):
        """Test that it performs classification."""
        x_train, x_test, y_train, y_test = train_test_sizes
        classifier = P1FPClassifierC(n_epoch=1)
        classifier.fit(x_train, y_train)

        prediction = classifier.predict(x_test)
        assert prediction.shape == (y_test.size, )
        assert np.all(np.isin(np.unique(prediction), np.unique(y_train)))

        prob_prediction = classifier.predict_proba(x_test)
        assert prob_prediction.shape == (y_test.size, np.unique(y_test).size)


@pytest.mark.slow
def test_keras_p1fpclassifierc(train_test_sizes):
    """Test that it performs classification."""
    x_train, x_test, y_train, y_test = train_test_sizes

    classifier = KerasP1FPClassifierC(
        n_features=5000, n_classes=3, epochs=10)

    classifier.fit(x_train, y_train)
    assert classifier.score(x_test, y_test) > 0.8
