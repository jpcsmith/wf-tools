"""Tests for the third-party li2018measuring module."""
import pytest
import numpy as np

from lab.third_party.li2018measuring import extract_features


@pytest.fixture(name="random_traces")
def fixture_random_traces():
    """Return a sequence of 10 traces of random lengths and values,
    where each is the tuple (timestamps, sizes).
    """
    rand = np.random.RandomState(1231)
    lengths = rand.randint(5000, 10_000, size=100)

    size_list = []
    time_list = []
    for length in lengths:
        # Generate the sizes and make sure they are non-zero
        sizes = rand.randint(-1500, 1501, size=length)
        sizes[sizes == 0] = 1500
        size_list.append(sizes)

        # Generate interarrival times and shift to the interval
        # [0.01, 0.1) i.e, [10ms, 100ms).
        interarrival = rand.random_sample(length)
        interarrival = (0.1 - 0.01) * interarrival + 0.01
        # Convert to to timestamps
        timestamps = np.cumsum(interarrival)
        # Always start from zero
        timestamps[0] = 0.0
        time_list.append(timestamps)

    return list(zip(time_list, size_list))


def test_extract_features_time(random_traces):
    """Timing analysis of extract_features."""
    for (times, sizes) in random_traces:
        _ = extract_features(times, sizes)
