"""Tests for the sniffer.py module."""
# pylint: disable=invalid-name
import signal
import subprocess
from unittest.mock import Mock, call
import pytest
from lab.sniffer import stop_process


@pytest.mark.timeout(5)
def test_stop_subprocess():
    """It should escalate the stopping procedure on a failed stop."""
    mock_process = Mock(spec=subprocess.Popen, strict=True, name="Popen")

    def mock_communicate(timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired([], timeout=timeout)
        return None, None
    mock_process.communicate.side_effect = mock_communicate

    stop_process(mock_process, timeout=3)
    assert mock_process.method_calls == [
        call.send_signal(signal.SIGINT),
        call.communicate(timeout=3),
        call.send_signal(signal.SIGTERM),
        call.communicate(timeout=3),
        call.send_signal(signal.SIGKILL),
        call.communicate(timeout=None),
    ]


@pytest.mark.timeout(5)
def test_stop_subprocess_suppress_term():
    """It should suppress termination signal errors."""
    mock_process = Mock(spec=subprocess.Popen, strict=True, name="Popen")
    mock_process.communicate.side_effect = [
        subprocess.TimeoutExpired([], timeout=3),
        subprocess.CalledProcessError(15, ["test"], stderr="stderr of test")
    ]

    stdout, stderr = stop_process(mock_process, timeout=3)
    assert stdout is None
    assert stderr == "stderr of test"
    assert mock_process.method_calls == [
        call.send_signal(signal.SIGINT),
        call.communicate(timeout=3),
        call.send_signal(signal.SIGTERM),
        call.communicate(timeout=3),
    ]


@pytest.mark.timeout(5)
def test_stop_subprocess_suppress_kill():
    """It should suppress kill signal errors."""
    mock_process = Mock(spec=subprocess.Popen, strict=True, name="Popen")
    mock_process.communicate.side_effect = [
        subprocess.TimeoutExpired([], timeout=3),
        subprocess.TimeoutExpired([], timeout=3),
        subprocess.CalledProcessError(9, ["test"], stderr="stderr of test")
    ]

    stdout, stderr = stop_process(mock_process, timeout=3)
    assert stdout is None
    assert stderr == "stderr of test"
    assert mock_process.method_calls == [
        call.send_signal(signal.SIGINT),
        call.communicate(timeout=3),
        call.send_signal(signal.SIGTERM),
        call.communicate(timeout=3),
        call.send_signal(signal.SIGKILL),
        call.communicate(timeout=None),
    ]
