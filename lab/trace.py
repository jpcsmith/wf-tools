"""Utilities and definitions relating to packet traces."""
import io
import logging
from enum import IntEnum
from typing import (
    Iterator,
    Iterable,
    List,
    NamedTuple,
    Set,
    Optional,
)

import scapy.utils
# Bug fix for rdpcap, see scapy.ml.secdev.narkive.com/h0rkmsiG/bug-in-rdpcap
from scapy.all import Raw  # pylint: disable=unused-import


_LOGGER = logging.getLogger(__name__)


class Direction(IntEnum):
    """The direction of a packet, incoming or outgoing."""
    IN = -1  # pylint: disable=invalid-name
    OUT = 1


class Packet(NamedTuple):
    """A packet in a trace.

    An outgoing packet has a direction of 1 and an incoming packet has a
    direction of -1. The size of the packet is in bytes.
    """
    timestamp: float
    direction: Direction
    size: int


Trace = List[Packet]


class ClientIndeterminable(Exception):
    """Raised if it is not possible to determine the client from the sequence of
    packets.
    """


def _ip_layers(packets: Iterable) -> Iterator:
    return (pkt.getlayer('IP') for pkt in packets if pkt.haslayer('IP'))


def _common_ip(packets: Iterable) -> Optional[str]:
    """Attempts to identify an IP common to all packets. Assumes that the
    capture was not made in promiscuous mode, and thus all packets were either
    from or directed to the endpoint. Returns None if it does not find a
    single such IP.
    """
    common_ip: Optional[Set[str]] = None
    for ip_layer in _ip_layers(packets):
        ips = {ip_layer.src, ip_layer.dst}
        if common_ip is None:
            common_ip = ips
        else:
            common_ip &= ips

        # Stop early if no common IPs among some subset of packets
        if not common_ip:
            break

    if common_ip and len(common_ip) == 1:
        return common_ip.pop()

    if common_ip is None:
        _LOGGER.debug("Common IP set never initialised, were there packets?")
    elif len(common_ip) > 1:
        _LOGGER.debug("Could not narrow down common packets %s", common_ip)
    elif not common_ip:
        _LOGGER.debug("No common packets present.")
    return None


def _syn_originator(packets: Iterable) -> Optional[str]:
    """Identifies the client as the IP address with originating TCP-SYN
    packets. If there are multiple such IPs, then the function returns None.
    """
    with_syn: Set[str] = set()
    for ip_layer in _ip_layers(packets):
        if not ip_layer.haslayer('TCP'):
            continue
        tcp = ip_layer.getlayer('TCP')

        if tcp.flags == 0b10:  # TCP SYN flag
            with_syn.add(ip_layer.src)
            if len(with_syn) > 1:
                break

    if len(with_syn) == 1:  # pylint: disable=no-else-return
        return with_syn.pop()
    elif len(with_syn) > 1:
        _LOGGER.debug("Multiple clients with originating SYNs: %s", with_syn)
    else:
        _LOGGER.debug("No SYN packets found.")
    return None


def _determine_client_ip(packets: Iterable) -> str:
    """Determines the IP address of the client from the sequence of packets.

    Raises ClientIndeterminable on failure.
    """
    packet_list = list(packets)
    client = _common_ip(packet_list) or _syn_originator(packet_list)
    if not client:
        raise ClientIndeterminable("Unable to determine client from trace.")
    return client


def _packets_with_endpoint(packets: Iterable, client: str) -> Iterator:
    def _has_endpoint(ip_layer, client: str) -> bool:
        return client in (ip_layer.src, ip_layer.dst)

    for packet in packets:
        if packet.haslayer('IP') and _has_endpoint(packet['IP'], client):
            yield packet


def pcap_to_trace(pcap: bytes) -> Trace:
    """Converts a pcap to a packet trace."""
    packets = scapy.utils.rdpcap(io.BytesIO(pcap))
    client = _determine_client_ip(packets)

    trace: Trace = []
    for packet in _packets_with_endpoint(packets, client):
        ip_layer = packet['IP']
        direction = Direction.OUT if ip_layer.src == client else Direction.IN
        trace.append(Packet(packet.time, direction, ip_layer.len))
    _LOGGER.info("pcap conversion resulted in %d trace packets, %d in pcap.",
                 len(trace), len(packets))
    return trace
