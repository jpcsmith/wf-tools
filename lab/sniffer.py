"""A wrapper around tshark to perform traffic sniffing."""
import io
import time
import logging
import threading
from typing import Optional

import scapy
import scapy.compat
import scapy.plist
import scapy.sendrecv
import scapy.utils
import scapy.layers.inet

# Disable all layers & protocols besides ethernet, IP, TCP and UDP
# scapy.layers.l2.Ether.payload_guess = [({"type": 0x800}, scapy.layers.inet.IP)]
# scapy.layers.inet.IP.payload_guess = [
#     ({"frag": 0, "proto": 0x11}, scapy.layers.inet.UDP),
#     ({"frag": 0, "proto": 0x06}, scapy.layers.inet.TCP),
# ]
# scapy.layers.inet.UDP.payload_guess = []
# scapy.layers.inet.TCP.payload_guess = []


class SnifferStartTimeout(Exception):
    """Raised when the sniffer fails to start due to a timeout."""


class TruncatingPcapWriter(scapy.utils.PcapWriter):
    """A pcap writer which additionally truncated packets to a predefined
    length before  writing.
    """
    def __init__(self, *args, snaplen: Optional[int] = None, **kwargs):
        assert snaplen is None or snaplen > 0
        super().__init__(*args, **kwargs)
        self.snaplen = snaplen

    def write(self, pkt):
        # We patch out the MTU variable here to ensure the correct value is
        # written to the PCAP header
        old_mtu = scapy.utils.MTU
        try:
            scapy.utils.MTU = self.snaplen or old_mtu
            super().write(pkt)
        finally:
            scapy.utils.MTU = old_mtu

    # pylint: disable=too-many-arguments
    def _write_packet(self, packet, sec=None, usec=None, caplen=None,
                      wirelen=None):
        """Writes a single packet to the pcap file.

        Operates the same as in PcapWriter, but truncated the packet to
        snaplen.
        """
        if hasattr(packet, "time"):
            if sec is None:
                sec = int(packet.time)
                usec = int(round((packet.time - sec)
                                 * (1000000000 if self.nano else 1000000)))
        if usec is None:
            usec = 0

        assert caplen is None and wirelen is None
        rawpkt = scapy.compat.raw(packet)

        caplen = len(rawpkt) if self.snaplen is None else self.snaplen
        wirelen = getattr(packet, "wirelen", None) or len(rawpkt)

        if self.snaplen is not None:
            rawpkt = rawpkt[:self.snaplen]

        # pylint: disable=protected-access
        assert caplen and wirelen
        scapy.utils.RawPcapWriter._write_packet(
            self, rawpkt, sec=sec, usec=usec, caplen=caplen, wirelen=wirelen)


class PacketSniffer:
    """Class for capturing network traffic."""
    stop_delay = 1

    def __init__(self, capture_filter: str = 'tcp or udp port 443',
                 snaplen: Optional[int] = None):
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
        )

        self.snaplen = snaplen

    @property
    def results(self) -> bytes:
        """Alias for pcap"""
        return self.pcap()

    def pcap(self) -> bytes:
        """Returns the results in pcap format serialised to bytes."""
        if self.snaplen:
            self._logger.info("Truncating packets to %d bytes", self.snaplen)
        return PacketSniffer.to_pcap(self._sniffer.results, self.snaplen)

    @staticmethod
    def to_pcap(packet_list: scapy.plist.PacketList,
                snaplen: Optional[int] = None) -> bytes:
        """Encodes the provided packet list in PCAP format."""
        byte_buffer = io.BytesIO()
        # with TruncatingPcapWriter(byte_buffer, snaplen=snaplen) as writer:
        with TruncatingPcapWriter("/tmp/some-pcap.pcap", snaplen=snaplen) as writer:
            writer.write(packet_list)
            writer.flush()
            return b'Dumy'
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
        self._logger.info('Waiting %.2fs for sniffer to flush', self.stop_delay)
        time.sleep(self.stop_delay)

        self._sniffer.stop()

        if not self._sniffer.results:
            self._logger.warning('Sniffing complete but failed to capture '
                                 'packets [result: %s]', self.results)
            return

        self._sniffer.results = scapy.plist.PacketList(
            name='Sniffed',
            res=self._sniffer.results,
            stats=[scapy.layers.inet.TCP, scapy.layers.inet.UDP])
        self._logger.info('Sniffing complete. %r', self._sniffer.results)
