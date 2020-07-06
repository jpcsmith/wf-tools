"""Tests for the sniffer.py module."""
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
