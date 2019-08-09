"""A wrapper around tshark to perform traffic sniffing."""
import time
import logging
from typing import (
    List,
    Optional
)
from subprocess import (
    PIPE,
    Popen,
    CompletedProcess,
    CalledProcessError,
)


class TShark:
    """A wrapper around tshark to perform traffic sniffing."""
    def __init__(self):
        self._logger = logging.getLogger(__name__)

        self._subprocess: Popen = None
        self._args: List[str] = None

    def sniff(self, capture_filter: str = None, fmt='pcap'):
        """Start sniffing for traffic."""
        assert fmt
        assert self._subprocess is None

        self._args = ['tshark', '-F', fmt] + (
            ['-f', capture_filter] if capture_filter else [])
        self._subprocess = Popen(self._args, stdout=PIPE, stderr=PIPE)
        self._logger.info("Started TShark with args '%s'", ' '.join(self._args))

    def _terminate(self) -> CompletedProcess:
        if self._subprocess.poll() is None:
            self._logger.info("Manually stopping TShark.")
            self._subprocess.terminate()
        else:
            self._logger.debug("TShark already terminated")

        stdout, stderr = self._subprocess.communicate()
        return CompletedProcess(
            self._args, self._subprocess.poll(), stdout, stderr)

    def _reset(self):
        assert self._subprocess is None or self._subprocess.poll() is not None
        self._subprocess = None
        self._args = None

    def stop(self) -> bytes:
        """Stops sniffing."""
        assert self._subprocess is not None
        result = self._terminate()

        try:
            result.check_returncode()
        except CalledProcessError as err:
            self._logger.fatal("TShark failed with error:\n %s", err.stderr)
            raise
        else:
            self._logger.debug("TShark stderr output:\n %s", result.stderr)
            return result.stdout
        finally:
            self._reset()


class PacketSniffer:
    """Class for capturing network traffic."""
    start_delay = 0.01

    def __init__(self, capture_filter: str = 'tcp or udp port 443'):
        self._logger = logging.getLogger(__name__)
        self._filter = capture_filter
        self._results: bytes = b''
        self._sniffer: Optional[TShark] = None

    @property
    def results(self) -> bytes:
        """Returns the packet list of captured packets."""
        return self._results

    def pcap(self) -> bytes:
        """Returns the results in pcap format serialised to bytes."""
        return self.results

    def start(self) -> None:
        """Start capturing packets."""
        assert self._sniffer is None
        self._sniffer = TShark()
        self._sniffer.sniff(self._filter)
        self._logger.info('Waiting %fs for sniffer to initialise',
                          self.start_delay)
        time.sleep(self.start_delay)
        self._logger.info('Began sniffing for traffic with filter "%s"',
                          self._filter)

    def stop(self) -> None:
        """Stop capturing packets."""
        assert self._sniffer is not None
        self._results = self._sniffer.stop()
        if self.results:
            self._logger.info('Sniffing complete. Capture length: %d',
                              len(self.results))
        else:
            self._logger.warning('Sniffing complete but failed to capture')
        self._sniffer = None
