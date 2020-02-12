"""Utilities and definitions relating to packet traces."""
import io
import json
import logging
import itertools
import subprocess
from enum import IntEnum
from ipaddress import IPv4Network, IPv6Network, ip_address
from typing import (
    Iterable, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Union,
)
import dataclasses
from dataclasses import dataclass

from mypy_extensions import TypedDict
import scapy.utils
# Bug fix for rdpcap, see scapy.ml.secdev.narkive.com/h0rkmsiG/bug-in-rdpcap
from scapy.all import Raw  # pylint: disable=unused-import
import pandas as pd

# Disable all layers & protocols besides ethernet, IP, TCP and UDP
scapy.layers.l2.Ether.payload_guess = [({"type": 0x800}, scapy.layers.inet.IP)]
scapy.layers.inet.IP.payload_guess = [
    ({"frag": 0, "proto": 0x11}, scapy.layers.inet.UDP),
    ({"frag": 0, "proto": 0x06}, scapy.layers.inet.TCP),
]
scapy.layers.inet.UDP.payload_guess = []
scapy.layers.inet.TCP.payload_guess = []

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
IPNetworkType = Union[IPv4Network, IPv6Network]


class ClientIndeterminable(Exception):
    """Raised if it is not possible to determine the client from the sequence of
    packets.
    """


def _ip_layers(packets: Iterable) -> Iterator:
    return (pkt.getlayer('IP') for pkt in packets if pkt.haslayer('IP'))


def _common_ip(packets: Iterable, client_subnet: Optional[IPNetworkType]) \
        -> Optional[str]:
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

    if common_ip and len(common_ip) == 2 and client_subnet:
        if not all(ip_address(ip) in client_subnet for ip in common_ip):
            # Exactly zero or one is in the subnet
            for ip_addr in common_ip:
                if ip_address(ip_addr) in client_subnet:
                    return ip_addr

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


def _has_endpoint(ip_layer, client: str) -> bool:
    return client in (ip_layer.src, ip_layer.dst)


def _packets_with_endpoint(packets: Iterable, client: str) -> Iterator:
    for packet in packets:
        if packet.haslayer('IP') and _has_endpoint(packet['IP'], client):
            yield packet


def _determine_client_ip(packets: pd.DataFrame, client_subnet) -> str:
    """Determines the IP address of the client from the sequence of packets.

    Raises ClientIndeterminable on failure.
    """
    # The client must of course be one of the senders
    unique_ips = packets['ip.src'].unique()
    candidates = [ip for ip in unique_ips if ip_address(ip) in client_subnet]

    if not candidates:
        raise ClientIndeterminable("No source IPs were in the subnet.")
    if len(candidates) > 1:
        raise ClientIndeterminable(f"Too many client candidates {candidates}.")
    return candidates[0]


def pcap_to_trace(pcap: bytes, client_subnet: IPNetworkType) \
        -> Tuple[Trace, Sequence[scapy.all.Packet]]:
    """Converts a pcap to a packet trace."""
    packets = load_pcap(pcap, str(client_subnet))
    client = _determine_client_ip(packets, client_subnet)

    packets['direction'] = Direction.OUT
    packets['direction'] = packets['direction'].where(
        packets['ip.src'] == client, Direction.IN)

    zero_time = packets['frame.time_epoch'].iloc[0]
    packets['frame.time_epoch'] = packets['frame.time_epoch'] - zero_time

    trace = [Packet(*fields) for fields in zip(
        packets['frame.time_epoch'], packets['direction'], packets['ip.len'])]
    return trace, packets


def load_pcap(pcap: bytes, client_subnet: str) -> pd.DataFrame:
    """Load the pcap into a dataframe.  Packets are filtered to those
    with an endpoint in client_subnet.
    """
    fields = ['frame.time_epoch', 'ip.src', 'ip.dst', 'ip.len', 'udp.stream',
              'tcp.stream']
    command = ['tshark', '-r', '-',
               '-Y', f'ip.src == {client_subnet} or ip.dst == {client_subnet}',
               '-Tfields', '-E', 'header=y', '-E', 'separator=,'] + list(
        itertools.chain.from_iterable(('-e', field) for field in fields))

    result = subprocess.run(
        command, input=pcap, check=True, capture_output=True)

    return pd.read_csv(io.BytesIO(result.stdout))


TraceStats = TypedDict('TraceStats', {
    'udp-flows': int, 'tcp-flows': int, 'udp-bytes': int, 'tcp-bytes': int
})


@dataclass
class TraceData:
    """Serialisable information pertaining to a traffic trace.

    Attributes
    ----------
    domain :
        An internet domain name.
    protocol : 'tcp' or 'quic'
        The protocol associated with the trace.
    connections :
        Counts of the number of 'udp' and 'tcp' flows in the trace,
        where each flow is identified by the IP-port 4-tuple. As well as the
        total bytes sent via UDP & TCP
    trace :
        The encoded traffic trace
    """
    domain: str
    protocol: str
    connections: TraceStats
    trace: Trace

    def serialise(self) -> str:
        """Serialise the trace info to a string for writing to a file."""
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def deserialise(cls, value: str) -> 'TraceData':
        """Deserialise a TraceData object from a string."""
        data = json.loads(value)
        data['trace'] = [Packet(*pkt) for pkt in data['trace']]
        return cls(**data)
