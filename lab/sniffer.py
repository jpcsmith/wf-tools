"""A wrapper around tshark to perform traffic sniffing."""
import io
import time
import logging
import threading

import scapy
import scapy.plist
import scapy.sendrecv
import scapy.utils


class SnifferStartTimeout(Exception):
    """Raised when the sniffer fails to start due to a timeout."""


class PacketSniffer:
    """Class for capturing network traffic."""
    stop_delay = 1

    def __init__(self, capture_filter: str = 'tcp or udp port 443'):
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
            started_callback=_started_callback
        )

    @property
    def results(self) -> bytes:
        """Alias for pcap"""
        return self.pcap()

    def pcap(self) -> bytes:
        """Returns the results in pcap format serialised to bytes."""
        return PacketSniffer.to_pcap(self._sniffer.results)

    @staticmethod
    def to_pcap(packet_list: scapy.plist.PacketList) -> bytes:
        """Encodes the provided packet list in PCAP format."""
        byte_buffer = io.BytesIO()
        with scapy.utils.PcapWriter(byte_buffer) as writer:
            writer.write(packet_list)
            writer.flush()
            return byte_buffer.getvalue()

    def start(self) -> None:
        """Start capturing packets."""
        with self._start_condition:
            self._sniffer.start()
            notified = self._start_condition.wait_for(
                lambda: self._started, timeout=5)

        if not notified:
            raise SnifferStartTimeout()
        self._logger.info('Began sniffing for traffic with filter "%s"',
                          self._filter)

    def stop(self) -> None:
        """Stop capturing packets."""
        self._logger.info('Waiting %.2fs for sniffer to flush', self.stop_delay)
        time.sleep(self.stop_delay)

        self._sniffer.stop()
        if self._sniffer.results:
            self._logger.info('Sniffing complete. Captured %d packets',
                              len(self._sniffer.results))
        else:
            self._logger.warning('Sniffing complete but failed to capture '
                                 'packets [result: %s]', self.results)
