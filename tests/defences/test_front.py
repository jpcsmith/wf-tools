"""Tests from the lab.defences.front module"""
import numpy.testing as npt

import lab.tracev2 as trace
from lab.defences import front


def test_simulate():
    """Test that it joins the two traces."""
    traffic = trace.as_trace([(0.3, 100), (.5, 200), (.7, 700)])
    padding = trace.as_trace([(0.1, 300), (.7, -200), (.9, -100)])
    result = front.simulate(traffic, padding)

    npt.assert_array_equal(result["time"], [0.1, 0.3, 0.5, 0.7, 0.7, 0.9])
    npt.assert_array_equal(result["size"], [300, 100, 200, 700, -200, -100])


def test_simulate_repeated_time():
    """Test that simulate correctly handles repeated timestamps."""
    traffic = trace.as_trace([(0.3, 100), (.5, 200), (.7, 700), (1.1, 400)])
    padding = trace.as_trace([(0.1, 300), (.7, -200), (.9, -100), (1.1, 300)])
    result = front.simulate(traffic, padding)

    npt.assert_array_equal(result["time"], [0.1, 0.3, 0.5, 0.7, 0.7, 0.9, 1.1])
    npt.assert_array_equal(
        result["size"], [300, 100, 200, 700, -200, -100, 700])
