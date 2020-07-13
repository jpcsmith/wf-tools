"""Utilities and definitions relating to packet traces."""
import io
import json
import logging
import itertools
import subprocess
from enum import IntEnum
from ipaddress import IPv4Network, IPv6Network, ip_address
from typing import List, NamedTuple, Tuple, Union, Optional
import dataclasses
from dataclasses import dataclass

from mypy_extensions import TypedDict
import pandas as pd

Trace = List["Packet"]
IPNetworkType = Union[IPv4Network, IPv6Network]
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


class ClientIndeterminable(Exception):
    """Raised if it is not possible to determine the client from the sequence of
    packets.
    """


class PcapParsingError(Exception):
    """Raised when we fail to parse the pcap."""


def _determine_client_ip(packets: pd.DataFrame, client_subnet) -> str:
    """Determines the IP address of the client from the sequence of packets.

    Raises ClientIndeterminable on failure.
    """
    # The client must of course be one of the senders
    unique_ips = packets['ip.src'].unique()
    candidates = [ip for ip in unique_ips if ip_address(ip) in client_subnet]

    if not candidates:
        # There were no IPs from the source in the subnet. This can happen
        # due to tcpdump being overloaded and dropping packets. See if we can
        # find the client IP in the destination IPs
        unique_ips = set(unique_ips)
        unique_ips.update(packets['ip.dst'].unique())
        candidates = [ip for ip in unique_ips if ip_address(ip) in
                      client_subnet]

    if not candidates:
        raise ClientIndeterminable(
            f"No source nor destination IPs were in the subnet: {unique_ips}.")
    if len(candidates) > 1:
        raise ClientIndeterminable(f"Too many client candidates {candidates}.")
    return candidates[0]


def pcap_to_trace(
    pcap: bytes, client_subnet: IPNetworkType,
    display_filter: Optional[str] = None
) -> Tuple[Trace, pd.DataFrame]:
    """Converts a pcap to a packet trace."""
    packets = load_pcap(pcap, str(client_subnet), display_filter)
    if len(packets) == 0:
        return [], packets

    client = _determine_client_ip(packets, client_subnet)

    packets['direction'] = Direction.OUT
    packets['direction'] = packets['direction'].where(
        packets['ip.src'] == client, Direction.IN)

    zero_time = packets['frame.time_epoch'].iloc[0]
    packets['frame.time_epoch'] = packets['frame.time_epoch'] - zero_time

    trace = [Packet(*fields) for fields in zip(
        packets['frame.time_epoch'], packets['direction'], packets['ip.len'])]
    return trace, packets


def load_pcap(
    pcap: bytes, client_subnet: str, display_filter: Optional[str] = None
) -> pd.DataFrame:
    """Load the pcap into a dataframe.  Packets are filtered to those
    with an endpoint in client_subnet.
    """
    fields = ['frame.time_epoch', 'ip.src', 'ip.dst', 'ip.len', 'udp.stream',
              'tcp.stream']

    filter_ip = f'ip.src == {client_subnet} or ip.dst == {client_subnet}'
    display_filter = (f'({filter_ip}) and ({display_filter})'
                      if display_filter else filter_ip)
    command = ['tshark', '-r', '-', '-Y', display_filter,
               '-Tfields', '-E', 'header=y', '-E', 'separator=,'] + list(
        itertools.chain.from_iterable(('-e', field) for field in fields))

    try:
        result = subprocess.run(
            command, input=pcap, check=True, capture_output=True)
    except subprocess.CalledProcessError as err:
        raise PcapParsingError(err.stderr.decode("utf-8").strip()) from err

    return (pd.read_csv(io.BytesIO(result.stdout))
            .sort_values(by='frame.time_epoch'))


TraceStats = TypedDict('TraceStats', {
    'udp-flows': int, 'tcp-flows': int, 'udp-bytes': int, 'tcp-bytes': int
})


@dataclass
class TraceData:
    """Serialisable information pertaining to a traffic trace.

    Attributes
    ----------
    url :
        The url fetched in the trace.
    protocol :
        The protocol associated with the trace.
    connections :
        Counts of the number of 'udp' and 'tcp' flows in the trace,
        where each flow is identified by the IP-port 4-tuple. As well as the
        total bytes sent via UDP & TCP
    trace :
        The encoded traffic trace
    """
    url: str
    protocol: str
    connections: Optional[TraceStats]
    trace: Trace
    region: Optional[str] = None

    def serialise(self) -> str:
        """Serialise the trace info to a string for writing to a file."""
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def deserialise(cls, value: str) -> 'TraceData':
        """Deserialise a TraceData object from a string."""
        data = json.loads(value)
        data['trace'] = [Packet(*pkt) for pkt in data['trace']]
        return cls(**data)
