"""Read and manipulate numpy arrays that represent traffic traces."""
import io
import subprocess
from typing import Union, List, Tuple, IO
from pathlib import PurePath

import numpy as np
import pandas as pd

#: Data type used for traces
PACKET_DTYPE = np.dtype([("time", "<f8"), ("size", "<i8")])


def as_trace(seq: List[Tuple[float, int]]) -> np.ndarray:
    """Return the sequence as a trace."""
    return np.array(seq, dtype=PACKET_DTYPE)


def sort(trace: np.ndarray) -> np.ndarray:
    """Sort a trace by time with with ties broken by outgoing packets
    first.
    """
    assert trace.dtype == PACKET_DTYPE, "invalid trace dtype"
    key_trace = np.array(
        list(map(lambda x: (x["time"], -x["size"]), trace)), dtype=PACKET_DTYPE
    )
    return trace[key_trace.argsort()]


def from_pcap(
    filename: Union[str, PurePath], /,
    relative_timestamps: bool = True
) -> np.ndarray:
    """Read and return a trace from the pcap at the specified filename.

    Assume that incoming packets have a srcport of 443 and all others
    are outgoing packets.
    """
    command = [
        "tshark", "-r", str(filename),
        "-T", "fields", "-E", "separator=,",
        "-e", "frame.time_epoch", "-e", "udp.length", "-e", "udp.srcport"
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    data = pd.read_csv(
        io.BytesIO(result.stdout), names=["time", "length", "is_outgoing"]
    )
    data["is_outgoing"] = data["is_outgoing"] != 443
    data.loc[~data["is_outgoing"], "length"] *= -1

    if relative_timestamps:
        data["time"] -= data["time"].min()

    return np.rec.fromarrays([data["time"], data["length"]], dtype=PACKET_DTYPE)


def from_csv(filename: Union[str, IO]) -> np.ndarray:
    """Create a trace from the CSV at filename.

    The file must be a headerless CSV file with the first column
    being timestamps in seconds, and the second being the signed packet
    sizes.
    """
    return np.loadtxt(filename, delimiter=",", dtype=PACKET_DTYPE)
