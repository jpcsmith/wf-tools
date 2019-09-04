"""Utilities and definitions relating to packet traces."""
import io
import logging
from decimal import Decimal
from enum import IntEnum
from ipaddress import (
    IPv4Network,
    IPv6Network,
    ip_address,
)
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Union,
)

import scapy.utils
# Bug fix for rdpcap, see scapy.ml.secdev.narkive.com/h0rkmsiG/bug-in-rdpcap
from scapy.all import Raw  # pylint: disable=unused-import
import pandas as pd


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


class PcapToTraceConverter:
    """Converts pcaps to `Trace`s. Allows caching the client IP address between
    traces to recover from failed client IP deductions.
    """
    def __init__(self, cache_client_ip: bool = False,
                 client_subnet: Optional[IPNetworkType] = None):
        self.cache_client_ip = cache_client_ip
        self.client_subnet = client_subnet
        self._cache: Set[str] = set()
        self._ip_stats: Dict['str', int] = {}
        self._packet_stats: Dict['str', List[int]] = {'pcap': [], 'trace': []}

    @property
    def cache(self) -> Set[str]:
        """The read-only cache of previously seen client IP addresses."""
        return self._cache

    def add_to_cache(self, ip_addr: str) -> None:
        """Adds the provided ip_addr to the cache."""
        self._cache.add(ip_addr)

    def _client_from_cache(self, packets: Iterable) -> Optional[str]:
        is_present = {client: False for client in self.cache}

        for client in self.cache:
            is_present[client] = any(_has_endpoint(ip_layer, client)
                                     for ip_layer in _ip_layers(packets))

        result = [client for client in self.cache if is_present[client]]
        if len(result) == 1:
            return result[0]

        if not result:
            _LOGGER.debug("No IPs from cache %s in trace.", self.cache)
        else:
            _LOGGER.debug("Multiple IPs from cache %s in trace.", self.cache)
        return None

    def _determine_client_ip(self, packets: Iterable) -> str:
        """Determines the IP address of the client from the sequence of packets.

        Raises ClientIndeterminable on failure.
        """
        packet_list = list(packets)
        client = (_common_ip(packet_list, self.client_subnet)
                  or _syn_originator(packet_list))

        if client and self.cache_client_ip:
            self.add_to_cache(client)
        elif not client and self.cache_client_ip:
            client = self._client_from_cache(packets)

        if not client:
            raise ClientIndeterminable("Unable to determine client from trace.")

        self._ip_stats[client] = self._ip_stats.get(client, 0) + 1
        return client

    def to_trace(self, pcap: bytes) -> Trace:
        """Converts a pcap to a packet trace."""
        packets = scapy.utils.rdpcap(io.BytesIO(pcap))
        client = self._determine_client_ip(packets)

        trace: Trace = []
        zero_time: Optional[Decimal] = None

        for packet in _packets_with_endpoint(packets, client):
            ip_layer = packet['IP']
            direction = (Direction.OUT if ip_layer.src == client
                         else Direction.IN)
            if zero_time is None:
                zero_time = packet.time
            timestamp = float(packet.time - zero_time)
            trace.append(Packet(timestamp, direction, ip_layer.len))
        self._packet_stats['pcap'].append(len(packets))
        self._packet_stats['trace'].append(len(trace))
        return trace

    def ip_stats(self) -> Dict['str', int]:
        """Returns the counts of deduced client IPs."""
        return self._ip_stats

    def packet_stats(self) -> pd.DataFrame:
        """Returns summary statistics of the packet sizes."""
        return pd.DataFrame(self._packet_stats).describe().transpose()
