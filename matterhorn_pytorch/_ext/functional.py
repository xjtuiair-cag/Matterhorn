import re
import matterhorn_pytorch.snn.firing as _firing
from typing import Tuple as _Tuple


def multi_spiking(spiking_function: _firing.Firing) -> bool:
    if isinstance(spiking_function, (_firing.Floor, _firing.Ceil, _firing.Round)):
        return True
    elif isinstance(spiking_function, (_firing.Rectangular, _firing.Polynomial, _firing.Sigmoid, _firing.Gaussian)):
        return False
    else:
        raise ValueError("Unknown spiking function: %s" % (spiking_function.__class__.__name__,))


def purify_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", name.lower().replace(".", "_").replace("-", "_"))