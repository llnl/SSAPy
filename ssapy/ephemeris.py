"""
Planetary ephemeris selection and long-span kernels.

SSAPy evaluates Sun, Moon and planet positions from JPL development
ephemerides (DE). The split SSATK data packages ship the supported kernels:

========== ======================== =========================
Ephemeris  Shipped kernel           Coverage
========== ======================== =========================
``de440``  ``de440s.bsp``           1849-12-26 to 2150-01-22
``de430``  ``de430_1900_2150.bsp``  1900-01-01 to 2150-01-01
========== ======================== =========================

DE440 is the default. Select DE430, for example to reproduce results made
with SSAPy <= 1.1.10, with :func:`set_planetary_ephemeris` or the
``SSAPY_EPHEMERIS`` environment variable.

For epochs outside the shipped kernel, SSAPy uses a longer kernel of the same
ephemeris (``de440.bsp`` or ``de430.bsp``, 1550-2650) and, for DE440 beyond
that, DE441 (-13200 to +17191). These are 120 MB to 1.65 GB, so they are not
shipped: SSAPy looks for them in the working directory and split data packages
and the cache directory, and otherwise downloads them from NAIF, checks their
SHA-256, and caches them, warning with :class:`EphemerisDownloadWarning`
before each download. Set ``SSAPY_EPHEMERIS_DOWNLOAD=0`` to forbid downloads
(e.g. on compute nodes without network access; prefetch on a login node with
:func:`fetch`) and ``SSAPY_DATA_CACHE`` to choose the cache directory.

The shipped and long kernels of one ephemeris hold the same solution, so
positions are continuous where SSAPy switches between them. DE441 is a
different solution from DE440 (no lunar core-mantle damping; less accurate
in the current century), so using it raises :class:`EphemerisRangeWarning`.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "EphemerisDownloadWarning",
    "EphemerisRangeWarning",
    "EphemerisUnavailableError",
    "set_planetary_ephemeris",
    "get_planetary_ephemeris",
    "fetch",
    "cache_dir",
]

NAIF_PLANETS_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"


class EphemerisDownloadWarning(UserWarning):
    """A long-span ephemeris kernel is being downloaded."""


class EphemerisRangeWarning(UserWarning):
    """An epoch is evaluated with a different solution than the one selected."""


class EphemerisUnavailableError(FileNotFoundError):
    """No kernel of the selected ephemeris covers the requested epoch."""


@dataclass(frozen=True)
class Kernel:
    name: str
    start_jd: float  # TDB Julian dates covered by every segment SSAPy uses
    end_jd: float
    sha256: Optional[str] = None  # None for kernels shipped in split data packages
    nbytes: Optional[int] = None
    solution: str = ""

    @property
    def url(self):
        return NAIF_PLANETS_URL + self.name

    def covers(self, jd):
        return (jd >= self.start_jd) & (jd <= self.end_jd)


EPHEMERIDES = {
    "de440": (
        Kernel("de440s.bsp", 2396752.5, 2506352.5, solution="DE440"),
        Kernel("de440.bsp", 2287184.5, 2688976.5,
               "a4ce9bf9b3282becc9f4b2ac3cebe03a2ae7599981aabd7265fd8482fff7c4b5", 119799808, "DE440"),
        Kernel("de441_part-2.bsp", 2440400.5, 8000016.5,
               "3abb17dae2d78dd34880377544aacb54892104a0d4462b322cb9f4454d4887f6", 1656830976, "DE441"),
        Kernel("de441_part-1.bsp", -3100015.5, 2440432.5,
               "13757827f5db41b835a24bbd637488636ce79a8ca754062fed17844f7d5b618e", 1651119104, "DE441"),
    ),
    "de430": (
        Kernel("de430_1900_2150.bsp", 2415020.5, 2506331.5, solution="DE430"),
        Kernel("de430.bsp", 2287184.5, 2688976.5,
               "6e1b277c5f07135a84950604b83e56b736be696a7f3560bcddb1d4aeb944fca1", 119741440, "DE430"),
    ),
}

_selected = None
_open_kernels = {}


def set_planetary_ephemeris(name):
    """Select the planetary ephemeris, ``"de440"`` (default) or ``"de430"``.

    Affects Sun, Moon and planet positions created afterwards
    (``get_body``, ``SunPosition``, ``MoonPosition``, ``PlanetPosition``).
    """
    global _selected
    key = str(name).lower()
    if key not in EPHEMERIDES:
        raise ValueError(f"Unknown planetary ephemeris {name!r}; choose from {sorted(EPHEMERIDES)}")
    _selected = key


def get_planetary_ephemeris():
    """Name of the selected planetary ephemeris."""
    if _selected is not None:
        return _selected
    key = os.environ.get("SSAPY_EPHEMERIS", "de440").lower()
    if key not in EPHEMERIDES:
        raise ValueError(f"SSAPY_EPHEMERIS={key!r} is not one of {sorted(EPHEMERIDES)}")
    return key


def cache_dir():
    """Directory where downloaded kernels are kept (``SSAPY_DATA_CACHE``)."""
    return os.path.expanduser(os.environ.get("SSAPY_DATA_CACHE", os.path.join("~", ".cache", "ssapy")))


def _downloads_allowed():
    return os.environ.get("SSAPY_EPHEMERIS_DOWNLOAD", "1").strip().lower() not in ("0", "false", "no", "off")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_url(url):  # separated so tests can replace the network
    from urllib.request import urlopen
    return urlopen(url, timeout=60)


def _download(kernel, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    handle, partial = tempfile.mkstemp(dir=os.path.dirname(destination), suffix=".part")
    try:
        with os.fdopen(handle, "wb") as out, _open_url(kernel.url) as response:
            shutil.copyfileobj(response, out, 1 << 20)
        if os.path.getsize(partial) != kernel.nbytes or _sha256(partial) != kernel.sha256:
            raise EphemerisUnavailableError(
                f"{kernel.name} downloaded from {kernel.url} does not match its recorded SHA-256"
            )
        os.replace(partial, destination)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def _local_path(kernel):
    from .utils import find_file
    try:
        return find_file(kernel.name)
    except FileNotFoundError:
        pass
    cached = os.path.join(cache_dir(), kernel.name)
    return cached if os.path.isfile(cached) else None


def fetch(name=None, include_de441=True):
    """Download (if needed) the long-span kernels of an ephemeris and return
    their paths. Use on a login node to fill ``SSAPY_DATA_CACHE`` before
    running jobs without network access."""
    family = EPHEMERIDES[(name or get_planetary_ephemeris()).lower()]
    return [
        _path(kernel, "requested with ssapy.ephemeris.fetch")
        for kernel in family[1:]
        if include_de441 or kernel.solution == family[0].solution
    ]


def _path(kernel, reason):
    path = _local_path(kernel)
    if path is not None:
        return path
    if kernel.sha256 is None:
        raise EphemerisUnavailableError(
            f"{kernel.name} is missing; install the appropriate ssatk-data-* package."
        )
    destination = os.path.join(cache_dir(), kernel.name)
    manual = (f"Download {kernel.url} ({kernel.nbytes / 1e6:.0f} MB) into {cache_dir()} "
              f"(or set SSAPY_DATA_CACHE), or place it in the working directory.")
    if not _downloads_allowed():
        raise EphemerisUnavailableError(f"{kernel.name} is needed ({reason}) and downloads are disabled "
                                        f"(SSAPY_EPHEMERIS_DOWNLOAD=0). {manual}")
    warnings.warn(
        f"Downloading {kernel.name} ({kernel.nbytes / 1e6:.0f} MB) from NAIF to {destination}: {reason}.",
        EphemerisDownloadWarning,
        stacklevel=6,
    )
    try:
        _download(kernel, destination)
    except EphemerisUnavailableError:
        raise
    except Exception as exc:  # no network, proxy, HTTP error, disk full
        raise EphemerisUnavailableError(f"Could not download {kernel.name}: {exc}. {manual}") from exc
    return destination


def _open(kernel, path):
    from jplephem.spk import SPK
    key = os.path.abspath(path)
    if key not in _open_kernels:
        _open_kernels[key] = SPK.open(path)
    return _open_kernels[key]


def _jd_text(jd):
    from astropy.time import Time
    return Time(float(jd), format="jd", scale="tdb").iso[:10]


class PlanetaryEphemeris:
    """Evaluate SPK segment chains, choosing a kernel per epoch.

    The shipped kernel of ``family`` (passed to :meth:`compute` as
    ``primary`` once opened) is used for every epoch it covers; other epochs
    go to the first longer kernel of the family that covers them.
    """

    def __init__(self, family=None):
        self.family = (family or get_planetary_ephemeris()).lower()
        self.kernels = EPHEMERIDES[self.family]

    def _kernel_for(self, kernel, reason, primary):
        if kernel is self.kernels[0] and primary is not None:
            return primary
        return _open(kernel, _path(kernel, reason))

    def shipped_path(self):
        return _path(self.kernels[0], "shipped kernel")

    def compute(self, chain, jd1, jd2, primary=None):
        """Sum ``sign * kernel[center, target].compute(jd1, jd2)`` over the
        ``(center, target, sign)`` terms in ``chain``; returns km."""
        jd2 = np.asarray(jd2, dtype=float)
        shipped = self.kernels[0]
        if primary is not None and np.all(shipped.covers(jd1 + jd2)):
            # Fast path (every call in normal use): one kernel, no masking.
            total = 0.0
            for center, target, sign in chain:
                total = total + sign * np.asarray(primary[center, target].compute(jd1, jd2))
            return total
        scalar = jd2.ndim == 0
        jd2 = np.atleast_1d(jd2)
        jd = jd1 + jd2
        out = np.zeros((3, jd2.size))
        pending = np.ones(jd2.size, dtype=bool)
        for kernel in self.kernels:
            mask = pending & kernel.covers(jd)
            if not np.any(mask):
                continue
            reason = None
            if kernel is not shipped:
                reason = (f"epochs {_jd_text(jd[mask].min())} to {_jd_text(jd[mask].max())} are outside "
                          f"{shipped.name} ({_jd_text(shipped.start_jd)} to {_jd_text(shipped.end_jd)})")
            spk = self._kernel_for(kernel, reason, primary)
            if kernel.solution != self.kernels[0].solution:
                warnings.warn(
                    f"Using {kernel.solution} ({kernel.name}) for {reason}; {kernel.solution} differs from "
                    f"{self.kernels[0].solution}, and the switch makes positions jump slightly at its boundary.",
                    EphemerisRangeWarning,
                    stacklevel=4,
                )
            for center, target, sign in chain:
                out[:, mask] += sign * np.asarray(spk[center, target].compute(jd1, jd2[mask])).reshape(3, -1)
            pending &= ~mask
        if np.any(pending):
            raise EphemerisUnavailableError(
                f"No {self.family.upper()} kernel covers {_jd_text(jd[pending].min())} to "
                f"{_jd_text(jd[pending].max())}."
            )
        return out[:, 0] if scalar else out
