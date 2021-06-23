"""Perform packet sniffing.

Currently uses the scapy library and is limited to TCP/UDP packets.
"""
import io
import abc
from abc import abstractmethod
import time
import contextlib
import logging
import threading
from typing import Optional, IO, List
import signal
import tempfile
import subprocess
from subprocess import CompletedProcess, CalledProcessError

try:
    import scapy
    import scapy.compat
    import scapy.plist
    import scapy.sendrecv
    import scapy.utils
    from scapy.layers import inet, l2

    # Enable parsing only to the level of UDP and TCP packets
    l2.Ether.payload_guess = [({"type": 0x800}, inet.IP)]
    inet.IP.payload_guess = [
        ({"frag": 0, "proto": 0x11}, inet.UDP), ({"proto": 0x06}, inet.TCP)
    ]
    inet.UDP.payload_guess = []
    inet.TCP.payload_guess = []
except ImportError:
    USE_SCAPY = False
else:
    USE_SCAPY = True


class SnifferStartTimeout(Exception):
    """Raised when the sniffer fails to start due to a timeout."""


class PacketSniffer(abc.ABC):
    """Base class for packet sniffers."""
    @property
    def results(self) -> bytes:
        """Alias for pcap"""
        return self.pcap()

    @abstractmethod
    def pcap(self) -> bytes:
        """Return the pcap as bytes."""

    @abstractmethod
    def start(self) -> None:
        """Begin capturing packets."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing packets."""


if USE_SCAPY:
    class ScapyPacketSniffer(PacketSniffer):
        """Class for capturing network traffic."""
        stop_delay = 1

        def __init__(self, capture_filter: str = 'tcp or udp',
                     snaplen: Optional[int] = None, **kwargs):
            def _started_callback():
                with self._start_condition:
                    self._started = True
                    self._start_condition.notify_all()

            self._logger = logging.getLogger(__name__)
            self._filter = capture_filter
            self._start_condition = threading.Condition()
            self._started = False
            self._sniffer = scapy.sendrecv.AsyncSniffer(
                filter=capture_filter,
                started_callback=_started_callback,
                promisc=False,
                **kwargs,
            )

            self.snaplen = snaplen

        def _truncate_pcap(self, pcap: bytes) -> bytes:
            assert self.snaplen is not None and self.snaplen > 0
            command = ['editcap', '-F', 'pcap',
                       '-s', str(self.snaplen), '-', '-']
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(pcap)

            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, ' '.join(command), stdout, stderr)
            return stdout

        def pcap(self) -> bytes:
            """Returns the results in pcap format serialised to bytes."""
            pcap = ScapyPacketSniffer.to_pcap(self._sniffer.results)
            if self.snaplen:
                self._logger.info("Truncating packets to %d bytes",
                                  self.snaplen)
                pcap = self._truncate_pcap(pcap)
            return pcap

        @staticmethod
        def to_pcap(packet_list: scapy.plist.PacketList) -> bytes:
            """Encodes the provided packet list in PCAP format."""
            byte_buffer = io.BytesIO()
            with scapy.utils.PcapWriter(byte_buffer) as writer:
                writer.write(packet_list)
                writer.flush()
                # PcapWriter will close the bytebuffer so must return in 'with'
                return byte_buffer.getvalue()

        def start(self) -> None:
            """Start capturing packets."""
            with self._start_condition:
                self._sniffer.start()
                notified = self._start_condition.wait_for(lambda: self._started,
                                                          timeout=5)

            if not notified:
                raise SnifferStartTimeout()
            self._logger.info('Began sniffing for traffic with filter "%s"',
                              self._filter)

        def stop(self) -> None:
            """Stop capturing packets."""
            self._logger.info('Waiting %.2fs for sniffer to flush',
                              self.stop_delay)
            time.sleep(self.stop_delay)

            try:
                self._sniffer.stop()
            except OSError as err:
                if err.errno != 9:
                    raise
                if self._sniffer.running:
                    self._logger.fatal('%s has been raised by the sniffer but '
                                       'the sniffer is still running.', err)
                    raise
                self._logger.info('%s has been suppressed as the sniffer is not'
                                  ' running.', err)

            if not self._sniffer.results:
                self._logger.warning('Sniffing complete but failed to capture '
                                     'packets [result: %s]', self.results)
                return

            self._sniffer.results = scapy.plist.PacketList(
                name='Sniffed',
                res=self._sniffer.results,
                stats=[inet.TCP, inet.UDP])
            self._logger.info('Sniffing complete. %r', self._sniffer.results)


@contextlib.contextmanager
def tcpdump(*args, **kwargs):
    """Sniff packets within a context manager using tcpdump.
    """
    sniffer = TCPDumpPacketSniffer(*args, **kwargs)
    sniffer.start()
    try:
        yield sniffer
    finally:
        sniffer.stop()


class TCPDumpPacketSniffer(PacketSniffer):
    """A wrapper around TCPDump to perform traffic sniffing."""
    start_delay = 2
    # How long to wait before terminating the sniffer
    stop_delay = 2
    buffer_size = 4096

    def __init__(
        self, capture_filter: str = 'udp or tcp', iface: Optional[str] = None,
        snaplen: Optional[int] = None
    ):
        self._log = logging.getLogger(__name__)
        self._subprocess: Optional[subprocess.Popen] = None
        self._pcap: Optional[IO[bytes]] = None
        self.interface = iface or 'any'
        self.snaplen = snaplen or 0
        self.capture_filter = capture_filter
        self._args: List[str] = []

    def pcap(self) -> bytes:
        assert self._pcap is not None
        pcap_bytes = self._pcap.read()
        self._pcap.seek(0)
        return pcap_bytes

    def is_running(self) -> bool:
        """Returns true if the sniffer is running."""
        return self._subprocess is not None

    def start(self) -> None:
        assert not self.is_running()

        self._pcap = tempfile.NamedTemporaryFile(mode='rb', suffix='.pcap')
        self._args = [
            'tcpdump', '-n', '--buffer-size', str(self.buffer_size),
            '--interface', self.interface, '--dont-verify-checksums',
            '--no-promiscuous-mode', '--snapshot-length', str(self.snaplen),
            '-w', self._pcap.name, self.capture_filter]
        self._subprocess = subprocess.Popen(
            self._args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(self.start_delay)
        self._log.info("Started tcpdump: '%s'", ' '.join(self._args))

    def _terminate(self) -> CompletedProcess:
        assert self.is_running()
        assert self._subprocess is not None

        if self._subprocess.poll() is None:
            # Wait for tcpdump to flush, this may only work because it's in
            # packet-buffered & immediate modes
            self._log.info('Waiting %.2fs for tcpdump to flush',
                           self.stop_delay)
            time.sleep(self.stop_delay)

            stdout, stderr = stop_process(
                self._subprocess, timeout=3, name="tcpdump")
            return_code = 0
        else:
            self._log.debug("tcpdump already terminated")
            stdout, stderr = self._subprocess.communicate()
            return_code = self._subprocess.poll()

        return CompletedProcess(self._args, return_code, stdout, stderr)

    def stop(self) -> None:
        """Stops sniffing."""
        assert self.is_running()
        result = self._terminate()

        try:
            result.check_returncode()
        except CalledProcessError as err:
            self._log.fatal(
                "TCPDump failed with error:\n%s", err.stderr.decode('utf-8'))
            raise
        else:
            n_collected = ', '.join(result.stderr.decode('utf-8').strip()
                                    .split('\n')[-3:])
            self._log.info("tcpdump complete: %s", n_collected)
        finally:
            self._subprocess = None


def stop_process(
    process: subprocess.Popen, timeout: int = 5, name: str = ''
) -> tuple:
    """Stop the process by sending SIGINT -> SIGTERM -> SIGKILL, waiting 5
    seconds between each pair of signals.
    """
    log = logging.getLogger(__name__)
    name = name or 'process'

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        log.info("Stopping %s with %s.", name, sig)
        next_timeout = None if sig == signal.SIGKILL else timeout

        try:
            process.send_signal(sig)
            return process.communicate(timeout=next_timeout)
        except subprocess.TimeoutExpired:
            log.info("%s did not stop after %.2fs. Trying next signal",
                     name, next_timeout)
        except subprocess.CalledProcessError as err:
            if err.returncode in (signal.SIGTERM, signal.SIGKILL):
                return err.stdout, err.stderr
            raise

    assert False
    return None, None
