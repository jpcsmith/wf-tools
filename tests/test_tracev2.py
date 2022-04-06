"""Tests for the tracev2 module."""
import io
import numpy as np
import lab.tracev2 as trace


def test_as_trace():
    """Test trace creation."""
    sequence = [(0.3, 100), (1.2, -400), (1.3, 500), (0.44, -700)]

    result = trace.as_trace(sequence)
    assert result.dtype == trace.PACKET_DTYPE
    np.testing.assert_array_equal(result["time"], [.3, 1.2, 1.3, .44])
    np.testing.assert_array_equal(result["size"], [100, -400, 500, -700])


def test_from_csv():
    csv_text = io.StringIO("0.0,1000\n0.1,-500\n.3,700")
    result = trace.from_csv(csv_text)

    assert result.dtype == trace.PACKET_DTYPE
    np.testing.assert_array_equal(result["time"], [0, 0.1, 0.3])
    np.testing.assert_array_equal(result["size"], [1000, -500, 700])
