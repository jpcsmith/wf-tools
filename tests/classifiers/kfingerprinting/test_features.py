"""Tests for the kfingerprinting.features module."""
# pylint: disable=redefined-outer-name
import random
import itertools
from typing import (
    Iterator
)

import pytest

from lab.classifiers.kfingerprinting.features import (
    Packet,
    Trace,
    extract_features,
    OUT,
    IN,
)


def _random_packets(seed: int) -> Iterator[Packet]:
    """Infinite sequence of packets generated with the seed."""
    rand = random.Random(seed)
    next_timestamp = 0.0
    while True:
        yield Packet(next_timestamp,
                     rand.choice([IN, OUT]),
                     rand.choice(range(1, 1501)))
        next_timestamp += rand.uniform(0.01, 0.1)


@pytest.fixture
def mixed_trace() -> Trace:
    """Returns a trace of both input and output packets."""
    return list(itertools.islice(_random_packets(0), 0, 2000))


def test_extract_features_sanity(mixed_trace: Trace):
    """Test the extract features function that it runs and returns a list of
    175 features.
    """
    features = extract_features(mixed_trace)
    assert len(features) == 175
