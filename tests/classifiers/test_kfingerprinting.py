"""Tests for lab.classifiers.kfingerprinting."""
# pylint: disable=redefined-outer-name
import pytest
from pytest import approx

from lab.classifiers.kfingerprinting import (
    interarrival_stats,
    timestamp_percentiles,
    packet_counts,
    EncPacket,
    Trace,
    head_tail_concentration,
)


@pytest.fixture
def empty_trace():
    """An empty packet trace."""
    return []


@pytest.fixture
def mixed_trace():
    """Returns a trace of both input and output packets."""
    return [
        EncPacket(0.0, 1000, incoming=False),
        EncPacket(0.1, 1500, incoming=True),
        EncPacket(0.3, 1000, incoming=True),
        EncPacket(0.6, 900, incoming=False),
        EncPacket(1.0, 1500, incoming=False),
        EncPacket(1.5, 100, incoming=True)
    ]


@pytest.fixture
def outgoing_trace():
    """Returns a trace of both input and output packets."""
    return [
        EncPacket(0.0, 1000, incoming=False),
        EncPacket(0.6, 900, incoming=False),
        EncPacket(1.0, 1500, incoming=False),
    ]


class TestInterarrivalStats:
    """Test cases for the interarrival_stats feature set."""
    @staticmethod
    def test_empty_trace(empty_trace):
        """It should return a feature set of 12 zeroes."""
        assert list(interarrival_stats(empty_trace)) == ([0] * 12)

    @staticmethod
    def test_mixed_trace(mixed_trace):
        """It should return the max, mean, std and upper quartiles of the
        interarrival times of the incoming, outgoing and entire trace.
        """
        stats = list(interarrival_stats(mixed_trace))
        assert stats[:4] == approx([1.2, 0.7, 0.5, 0.95])
        assert stats[4:8] == approx([0.6, 0.5, 0.1, 0.55])
        assert stats[8:] == approx([0.5, 0.3, 0.1414, 0.4], 1e-3)

    @staticmethod
    def test_outgoing_trace(outgoing_trace):
        """It should return the max, mean, std and upper quartile for the
        outgoing trace and total trace (the same) while leaving the incoming
        stats as 0.
        """
        stats = list(interarrival_stats(outgoing_trace))
        assert stats[:4] == [0.0] * 4
        assert stats[4:8] == approx([0.6, 0.5, 0.1, 0.55])
        assert stats[8:] == approx([0.6, 0.5, 0.1, 0.55])


class TestTimestampPercentiles:
    """Test cases for the timestamp_percentile feature set."""
    @staticmethod
    def test_empty_trace(empty_trace):
        """It should return a feature set of 12 zeroes."""
        assert list(timestamp_percentiles(empty_trace)) == ([0] * 12)

    @staticmethod
    def test_mixed_trace(mixed_trace):
        """It should return the max, mean, std and upper quartiles of the
        interarrival times of the incoming, outgoing and entire trace.
        """
        percentiles = list(timestamp_percentiles(mixed_trace))
        assert percentiles[:4] == approx([0.2, 0.3, 0.9, 1.5])
        assert percentiles[4:8] == approx([0.3, 0.6, 0.8, 1])
        assert percentiles[8:] == approx([0.15, 0.45, 0.9, 1.5], 1e-3)

    @staticmethod
    def test_outgoing_trace(outgoing_trace):
        """It should return the max, mean, std and upper quartile for the
        outgoing trace and total trace (the same) while leaving the incoming
        stats as 0.
        """
        percentiles = list(timestamp_percentiles(outgoing_trace))
        assert percentiles[:4] == [0.0] * 4
        assert percentiles[4:8] == approx([0.3, 0.6, 0.8, 1])
        assert percentiles[8:] == approx([0.3, 0.6, 0.8, 1])


class TestPacketCounts:
    """Test cases for the timestamp_percentile feature set."""
    @staticmethod
    def test_empty_trace(empty_trace: Trace):
        """It should return a feature set of 3 zeroes."""
        assert packet_counts(empty_trace) == (0, 0, 0)

    @staticmethod
    def test_mixed_trace(mixed_trace: Trace):
        """It should return the number of packets in the
        (incoming, outgoing, entire) subtraces.
        """
        assert packet_counts(mixed_trace) == (3, 3, 6)

    @staticmethod
    def test_outgoing_trace(outgoing_trace: Trace):
        """It should return the number of packets in the
        (incoming, outgoing, entire) subtraces, with the form
        (0, outgoing, outgoing) since there are no incoming packets.
        """
        assert packet_counts(outgoing_trace) == (0, 3, 3)


class TestHeadTailConcentration:
    """Test cases for the timestamp_percentile feature set."""
    @staticmethod
    def test_empty_trace(empty_trace: Trace):
        """It should return a feature set of 4 zeroes."""
        assert head_tail_concentration(empty_trace) == (0, 0, 0, 0)

    @staticmethod
    def test_mixed_trace(mixed_trace: Trace):
        """It should return the # inc, # out. for the head and tail
        respectively.
        """
        assert head_tail_concentration(mixed_trace, length=3) == (2, 1, 1, 2)

    @staticmethod
    def test_outgoing_trace(outgoing_trace: Trace):
        """It should return the number of packets in the
        (incoming, outgoing, entire) subtraces, with the form
        (0, outgoing, outgoing) since there are no incoming packets.
        """
        assert head_tail_concentration(outgoing_trace, length=2) == (0, 2, 0, 2)
