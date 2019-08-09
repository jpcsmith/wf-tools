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


class TCPDump:
    """A wrapper around tshark to perform traffic sniffing."""
    stop_delay = 0.5

    def __init__(self):
        self._logger = logging.getLogger(__name__)

        self._subprocess: Popen = None
        self._args: List[str] = None

    def sniff(self, capture_filter: str = ''):
        """Start sniffing for traffic."""
        assert self._subprocess is None

        self._args = ['tcpdump', '--no-promiscuous-mode', '--immediate-mode',
                      '-U', '-w', '/dev/stdout', capture_filter]
        self._subprocess = Popen(self._args, stdout=PIPE, stderr=PIPE)
        self._logger.info("Started TCPDump with args '%s'",
                          ' '.join(self._args))

    def _terminate(self) -> CompletedProcess:
        if self._subprocess.poll() is None:
            # Wait for tcpdump to flush, this may only work because it's in
            # packet-buffered & immediate modes
            self._logger.info('Waiting %.2fs for tcpdump to flush',
                              self.stop_delay)
            time.sleep(self.stop_delay)
            self._logger.info("Manually stopping TCPDump.")
            self._subprocess.terminate()
        else:
            self._logger.debug("TCPDump already terminated")

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
            self._logger.fatal("TCPDump failed with error:\n%s",
                               err.stderr.decode('utf-8'))
            raise
        else:
            self._logger.debug("TCPDump stderr output:\n%s",
                               result.stderr.decode('utf-8'))
            return result.stdout
        finally:
            self._reset()


class PacketSniffer:
    """Class for capturing network traffic."""
    start_delay = 1

    def __init__(self, capture_filter: str = 'tcp or udp port 443'):
        self._logger = logging.getLogger(__name__)
        self._filter = capture_filter
        self._results: bytes = b''
        self._sniffer: Optional[TCPDump] = None

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
        self._sniffer = TCPDump()
        self._sniffer.sniff(self._filter)
        self._logger.info('Waiting %.2fs for sniffer to initialise',
                          self.start_delay)
        time.sleep(self.start_delay)
        self._logger.info('Began sniffing for traffic with filter "%s"',
                          self._filter)

    def stop(self) -> None:
        """Stop capturing packets."""
        assert self._sniffer is not None
        self._results = self._sniffer.stop()
        if self.results:
            self._logger.info('Sniffing complete. PCAP length: %d',
                              len(self.results))
        else:
            self._logger.warning('Sniffing complete but failed to capture')
        self._sniffer = None
