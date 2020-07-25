"""Tests for feature extraction."""
# pylint: disable=invalid-name
import pytest
import numpy as np

from lab.feature_extraction.trace import (
    extract_interarrival_times, extract_metadata, Metadata, ensure_non_ragged
)


@pytest.fixture(name="sample_data")
def fixture_sample_data() -> tuple:
    """Returns a tuple of ragged sizes and interarrival times."""
    sizes = [[1350, 1350, -600, 70], [1300, 1350, 1200], [1200, 1350]]
    times = [[0, 0.010, 0.020, 0.03], [0, 0.015, 0.025], [0, 0.020]]
    return (sizes, times)


def test_extract_interarrival_times():
    """It should extract the interarrival times and pad as necessary.
    """
    data = [[0, 0.010, 0.020, 0.03], [0, 0.015, 0.025], [0, 0.020]]
    expected = [
        [0, 0.010, 0.01, 0.01], [0, 0.015, 0.01, 0.00], [0, 0.020, 0.00, 0.00]
    ]
    np.testing.assert_allclose(extract_interarrival_times(data), expected)


def test_metadata_dimension():
    """It should return the number of columns required for the metadata.
    """
    assert Metadata.COUNT_METADATA.n_features == 5
    assert Metadata.SIZE_METADATA.n_features == 5
    assert Metadata.TIME_METADATA.n_features == 2
    assert Metadata.UNSPECIFIED.n_features == 12  # pylint: disable=no-member
    assert (Metadata.COUNT_METADATA | Metadata.SIZE_METADATA).n_features == 10
    assert (Metadata.COUNT_METADATA | Metadata.TIME_METADATA).n_features == 7
    assert (Metadata.UNSPECIFIED | Metadata.TIME_METADATA).n_features == 2


@pytest.fixture(name="ragged_data")
def fixture_ragged_data() -> tuple:
    """Returns a tuple of ragged and padded data of with a max length
    of 4 in axis 1.
    """
    ragged = [[1350, 1350, -600, 70], [1300, 1350, 1200], [1200, 1350]]
    non_ragged = np.array([[1350, 1350, -600, 70], [1300, 1350, 1200, 0],
                           [1200, 1350, 0, 0]])
    return ragged, non_ragged


def test_ensure_non_ragged(ragged_data):
    """Ensures that a ragged array is made not ragged.
    """
    data, expected = ragged_data

    result = ensure_non_ragged(data, copy=False)
    np.testing.assert_array_equal(result, expected)
    assert not np.shares_memory(result, expected)


def test_ensure_non_ragged_noop(ragged_data):
    """If already not ragged, it should not be changed"""
    _, expected = ragged_data
    result = ensure_non_ragged(expected, copy=False)
    np.testing.assert_array_equal(result, expected)
    assert np.shares_memory(result, expected)


def test_ensure_non_ragged_copy(ragged_data):
    """Should copy when copy=True"""
    _, expected = ragged_data
    result = ensure_non_ragged(expected, copy=True)
    np.testing.assert_array_equal(result, expected)
    assert not np.shares_memory(result, expected)


def test_ensure_non_ragged_crop(ragged_data):
    """It should reduce the data to the specified dimension."""
    data, expected = ragged_data
    result = ensure_non_ragged(data, dimension=3)
    np.testing.assert_array_equal(result, expected[:, :3])


def test_ensure_non_ragged_pad(ragged_data):
    """It should reduce the data to the specified dimension."""
    data, expected = ragged_data
    result = ensure_non_ragged(data, dimension=6)
    np.testing.assert_array_equal(result, np.pad(expected, [(0, 0), (0, 2)]))


def test_ensure_non_ragged_crop_no_copy(ragged_data):
    """It should reduce the data to the specified dimension if no copy is
    necessary.
    """
    _, expected = ragged_data
    result = ensure_non_ragged(expected, dimension=3)
    np.testing.assert_array_equal(result, expected[:, :3])


def test_extract_metadata_time_metadata(sample_data):
    """It should extract duration metadata from the traces."""
    _, times = sample_data
    np.testing.assert_allclose(
        extract_metadata(timestamps=times, metadata=Metadata.TIME_METADATA),
        np.transpose([
            # Duration
            [0.03, 0.025, 0.02],
            # Duration per packet
            [0.03/4, 0.025/3, 0.02/2]
        ]))


def test_extract_metadata_count_metadata(sample_data):
    """It should extract all count metadata from the traces."""
    sizes, _ = sample_data
    np.testing.assert_allclose(
        extract_metadata(sizes=sizes, metadata=Metadata.COUNT_METADATA),
        np.transpose([
            # Packet counts
            [4, 3, 2],
            # Outgoing and incoming counts
            [3, 3, 2], [1, 0, 0],
            # Outgoing and incoming count ratios
            [3/4, 3/3, 2/2], [1/4, 0, 0]
        ]))


def test_extract_metadata_size_metadata(sample_data):
    """It should extract all size metadata from the traces."""
    sizes, _ = sample_data
    np.testing.assert_allclose(
        extract_metadata(sizes=sizes, metadata=Metadata.SIZE_METADATA),
        np.transpose([
            # Sizes
            [3370, 3850, 2550],
            # Outgoing and incoming sizes
            [2770, 3850, 2550], [600, 0, 0],
            # Outgoing and incoming size ratio
            [2770/3370, 1, 1], [600/3370, 0, 0],
        ]))


def test_extract_metadata_unspecified(sample_data):
    """It should return all of the metadata if unspecified."""
    n_features = 12
    assert extract_metadata(*sample_data).shape == (3, n_features)


def test_extract_metadata_subsets(sample_data):
    """It should return some of the metadata only."""
    sizes, times = sample_data
    assert extract_metadata(
        sizes, times, metadata=(Metadata.SIZE_METADATA | Metadata.TIME_METADATA)
    ).shape == (3, 7)
    assert extract_metadata(
        sizes, times,
        metadata=(Metadata.SIZE_METADATA | Metadata.COUNT_METADATA)
    ).shape == (3, 10)


def test_extract_metadata_require_sizes_or_timestamps(sample_data):
    """Test that it raises value error if incorrect features are provided."""
    sizes, times = sample_data

    with pytest.raises(ValueError):
        extract_metadata(sizes, metadata=Metadata.TIME_METADATA)
    with pytest.raises(ValueError):
        extract_metadata(timestamps=times, metadata=Metadata.COUNT_METADATA)
    with pytest.raises(ValueError):
        extract_metadata(timestamps=times, metadata=Metadata.SIZE_METADATA)
