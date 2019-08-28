"""Tests for lab.classifiers.kfingerprinting."""
# pylint: disable=redefined-outer-name
from typing import (
    Iterable,
    Tuple,
)

import pytest
from pytest import approx
import pandas as pd
from pandas.api.types import CategoricalDtype

from lab.classifiers.kfingerprinting import (
    interarrival_stats,
    timestamp_percentiles,
    packet_counts,
)


@pytest.fixture
def empty_trace():
    """An empty packet trace."""
    return make_trace([])


@pytest.fixture
def mixed_trace():
    """Returns a trace of both input and output packets."""
    return make_trace([(0.0, 1000, False), (0.1, 1500, True), (0.3, 1000, True),
                       (0.6, 900, False), (1.0, 1500, False), (1.5, 100, True)])


@pytest.fixture
def outgoing_trace():
    """Returns a trace of both input and output packets."""
    return make_trace(
        [(0.0, 1000, False), (0.6, 900, False), (1.0, 1500, False)])


def make_trace(packets: Iterable[Tuple[float, int, bool]]) -> pd.DataFrame:
    """Create a dataframe containing the provided packets."""
    direction_cat = CategoricalDtype(['in', 'out', 'both'])
    trace = pd.DataFrame.from_records(
        packets, columns=['timestamp', 'size', 'direction'])
    trace['direction'] = trace['direction'].map(
        {True: 'in', False: 'out'}).astype(direction_cat)
    return trace


@pytest.fixture
def long_trace():
    """Returns a long trace of both input and output packets."""
    return make_trace([
        (0.0, 1000, False), (0.1, 900, True), (0.2, 1500, False),
        (0.3, 1000, False), (0.4, 900, True), (0.10, 1000, False),
        (0.11, 900, True), (0.12, 1500, True), (0.13, 1000, True),
        (0.14, 900, True), (0.20, 1500, True), (0.21, 1000, False),
        (0.22, 900, False), (0.23, 1500, True), (0.24, 1500, True)])


class TestInterarrivalStats:
    """Test cases for the interarrival_stats feature set."""
    @staticmethod
    def test_mixed_trace(mixed_trace):
        """It should return the max, mean, std and upper quartiles of the
        interarrival times of the incoming, outgoing and entire trace.
        """
        stats = interarrival_stats(mixed_trace)

        assert stats.loc['in', 'max'] == approx(1.2)
        assert stats.loc['in', 'mean'] == approx(0.7)
        assert stats.loc['in', 'std'] == approx(0.707, 1e-3)
        assert stats.loc['in', '75%'] == approx(0.95)

        assert stats.loc['out', 'max'] == approx(0.6)
        assert stats.loc['out', 'mean'] == approx(0.5)
        assert stats.loc['out', 'std'] == approx(0.141, 1e-2)
        assert stats.loc['out', '75%'] == approx(0.55)

        assert stats.loc['both', 'max'] == approx(0.5)
        assert stats.loc['both', 'mean'] == approx(0.3)
        assert stats.loc['both', 'std'] == approx(0.158, 1e-2)
        assert stats.loc['both', '75%'] == approx(0.4)

    @staticmethod
    def test_outgoing_trace(outgoing_trace):
        """It should return the max, mean, std and upper quartile for the
        outgoing trace and total trace (the same) while leaving the incoming
        stats as 0.
        """
        stats = interarrival_stats(outgoing_trace)

        assert stats.loc['in', 'max'] == 0
        assert stats.loc['in', 'mean'] == 0
        assert stats.loc['in', 'std'] == 0
        assert stats.loc['in', '75%'] == 0

        assert stats.loc['out', 'max'] == approx(0.6)
        assert stats.loc['out', 'mean'] == approx(0.5)
        assert stats.loc['out', 'std'] == approx(0.141, 1e-2)
        assert stats.loc['out', '75%'] == approx(0.55)

        assert stats.loc['both', 'max'] == approx(0.6)
        assert stats.loc['both', 'mean'] == approx(0.5)
        assert stats.loc['both', 'std'] == approx(0.141, 1e-2)
        assert stats.loc['both', '75%'] == approx(0.55)


class TestTimestampPercentiles:
    """Test cases for the timestamp_percentile feature set."""
    @staticmethod
    def test_mixed_trace(mixed_trace):
        """It should return the max, mean, std and upper quartiles of the
        interarrival times of the incoming, outgoing and entire trace.
        """
        percentiles = timestamp_percentiles(mixed_trace)

        assert list(percentiles.loc['in']) == approx([0.2, 0.3, 0.9, 1.5])
        assert list(percentiles.loc['out']) == approx([0.3, 0.6, 0.8, 1])
        assert list(percentiles.loc['both']) == approx(
            [0.15, 0.45, 0.9, 1.5], 1e-3)

    @staticmethod
    def test_outgoing_trace(outgoing_trace):
        """It should return the max, mean, std and upper quartile for the
        outgoing trace and total trace (the same) while leaving the incoming
        stats as 0.
        """
        percentiles = timestamp_percentiles(outgoing_trace)

        print(list(percentiles.loc['in']))
        assert list(percentiles.loc['in']) == [0.0] * 4
        assert list(percentiles.loc['out']) == approx([0.3, 0.6, 0.8, 1])
        assert list(percentiles.loc['both']) == approx([0.3, 0.6, 0.8, 1])


class TestPacketCounts:
    """Test cases for the timestamp_percentile feature set."""
    @staticmethod
    def test_mixed_trace(mixed_trace: pd.DataFrame):
        """It should return the number of packets in the
        (incoming, outgoing, entire) subtraces.
        """
        result = packet_counts(mixed_trace)

        assert result.loc['in'] == 3
        assert result.loc['out'] == 3
        assert result.loc['both'] == 6

    @staticmethod
    def test_outgoing_trace(outgoing_trace: pd.DataFrame):
        """It should return the number of packets in the
        (incoming, outgoing, entire) subtraces, with the form
        (0, outgoing, outgoing) since there are no incoming packets.
        """
        result = packet_counts(outgoing_trace)

        assert result.loc['in'] == 0
        assert result.loc['out'] == 3
        assert result.loc['both'] == 3
#
#
# class TestHeadTailConcentration:
#     """Test cases for the timestamp_percentile feature set."""
#     @staticmethod
#     def test_empty_trace(empty_trace: Trace):
#         """It should return a feature set of 4 zeroes."""
#         assert head_tail_concentration(empty_trace) == (0, 0, 0, 0)
#
#     @staticmethod
#     def test_mixed_trace(mixed_trace: Trace):
#         """It should return the # inc, # out. for the head and tail
#         respectively.
#         """
#         assert head_tail_concentration(mixed_trace, length=3) == (2, 1, 1, 2)
#
#     @staticmethod
#     def test_outgoing_trace(outgoing_trace: Trace):
#         """It should return the number of packets in the
#         (incoming, outgoing, entire) subtraces, with the form
#         (0, outgoing, outgoing) since there are no incoming packets.
#         """
#         assert head_tail_concentration(outgoing_trace, length=2) == (0, 2, 0, 2)
#
#
# class TestOutgoingConcentrationStats:
#     """Test cases for the outgoing_concentration_stats."""
#     @staticmethod
#     def test_empty_trace(empty_trace: Trace):
#         pass
#
#     @staticmethod
#     def test_mixed_trace(long_trace: Trace):
#         """It should compute the summary stats over bins of size 5 from the
#         long trace.
#         """
#         result = outgoing_concentration_stats(long_trace, bin_size=5)
#         print(result)
#         assert result['min'] == 1
#         assert result['max'] == 3
#         assert result['50%'] == 2
#         assert result['mean'] == 2
#         assert result['std'] == 1
