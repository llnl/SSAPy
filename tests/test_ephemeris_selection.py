"""Planetary ephemeris selection, long-span downloads, and their failure modes.

The download tests stand in a local file for NAIF, so they run offline: a
2000-2050 excerpt of the shipped de440s.bsp plays the shipped kernel and the
full de440s.bsp plays the long-span kernel "downloaded" for other epochs.
"""
import io
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from astropy.time import Time
from jplephem.spk import SPK

import ssapy
from ssapy import ephemeris
from ssapy.body import MoonPosition, SunPosition
from ssapy.utils import find_file


@pytest.fixture(autouse=True)
def _reset_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(ephemeris, "_selected", None)
    monkeypatch.delenv("SSAPY_EPHEMERIS", raising=False)
    monkeypatch.delenv("SSAPY_EPHEMERIS_DOWNLOAD", raising=False)
    monkeypatch.setenv("SSAPY_DATA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(ephemeris, "_open_kernels", {})


def _gps(iso):
    # TT avoids ERFA's "dubious year" warnings for UTC far from the present.
    return Time(iso, scale="tt").gps


def test_de440_and_de430_agree_to_their_published_difference():
    # DE440 is the default; DE430 is selectable. In 2026 the two solutions'
    # geocentric Moon differs by <= 10 m and Sun by <= 0.5 km over 1975-2050
    # (measured from the NAIF kernels); hold them to 20 m and 1 km.
    t = _gps("2026-10-08T00:00:00")
    assert ephemeris.get_planetary_ephemeris() == "de440"
    moon440, sun440 = MoonPosition()(t), SunPosition()(t)
    ephemeris.set_planetary_ephemeris("de430")
    moon430, sun430 = MoonPosition()(t), SunPosition()(t)
    assert 0 < np.linalg.norm(moon440 - moon430) < 20.0
    assert np.linalg.norm(sun440 - sun430) < 1e3
    with pytest.raises(ValueError):
        ephemeris.set_planetary_ephemeris("de405")


def test_environment_variable_selects_the_ephemeris(monkeypatch):
    monkeypatch.setenv("SSAPY_EPHEMERIS", "DE430")
    assert ephemeris.get_planetary_ephemeris() == "de430"
    assert os.path.basename(ssapy.body._planetary_ephemeris_path()) == "de430_1900_2150.bsp"


def test_no_download_inside_the_shipped_span(monkeypatch):
    def no_network(url):
        raise AssertionError("no download expected")

    monkeypatch.setattr(ephemeris, "_open_url", no_network)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MoonPosition()(_gps(["1900-01-01T00:00:00", "2026-10-08T00:00:00", "2149-12-31T00:00:00"]))


def _harness(monkeypatch, tmp_path, *, long_solution="DE440", served=None):
    """Shipped kernel = 2000-2050 excerpt of de440s; long kernel = de440s."""
    shipped_full = find_file("de440s.bsp")
    excerpt = tmp_path / "short_test.bsp"
    subprocess.run([sys.executable, "-m", "jplephem", "excerpt", "2000/1/1", "2050/1/1", shipped_full, str(excerpt)],
                   check=True, stdout=subprocess.DEVNULL)
    payload = open(shipped_full, "rb").read() if served is None else served
    monkeypatch.chdir(tmp_path)
    full = SPK.open(shipped_full)
    start = max(seg.start_jd for seg in full.segments)
    end = min(seg.end_jd for seg in full.segments)
    family = (
        ephemeris.Kernel("short_test.bsp", 2451544.5, 2469807.5, solution="DE440"),
        ephemeris.Kernel("long_test.bsp", start, end, ephemeris._sha256(shipped_full),
                         os.path.getsize(shipped_full), long_solution),
    )
    monkeypatch.setitem(ephemeris.EPHEMERIDES, "de440", family)
    calls = []

    def fake_open_url(url):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(ephemeris, "_open_url", fake_open_url)
    return full, calls


def _direct_moon(kernel, t):
    mjd = ssapy.utils._gpsToTT(t)
    return (kernel[3, 301].compute(2400000.5, mjd) - kernel[3, 399].compute(2400000.5, mjd)) * 1e3


def test_epochs_outside_the_shipped_span_download_with_a_warning(monkeypatch, tmp_path):
    # An epoch beyond the shipped span downloads the long kernel once, warns,
    # caches it, and evaluates it exactly; epochs inside stay on the shipped
    # kernel, and the two agree at the boundary (same solution).
    full, calls = _harness(monkeypatch, tmp_path)
    moon = MoonPosition()
    times = _gps(["2026-10-08T00:00:00", "2100-06-01T00:00:00"])
    with pytest.warns(ephemeris.EphemerisDownloadWarning, match="long_test.bsp"):
        got = moon(times)
    assert len(calls) == 1 and calls[0].endswith("long_test.bsp")
    assert os.path.isfile(tmp_path / "cache" / "long_test.bsp")
    np.testing.assert_allclose(got[:, 1], _direct_moon(full, times[1]), rtol=0, atol=1e-6)
    np.testing.assert_allclose(got[:, 0], _direct_moon(full, times[0]), rtol=0, atol=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        moon(times)  # cached: no second download or warning
    assert len(calls) == 1


def test_a_different_long_span_solution_warns(monkeypatch, tmp_path):
    _harness(monkeypatch, tmp_path, long_solution="DE441")
    with pytest.warns(ephemeris.EphemerisDownloadWarning):
        with pytest.warns(ephemeris.EphemerisRangeWarning, match="DE441"):
            MoonPosition()(_gps("2100-06-01T00:00:00"))


@pytest.mark.parametrize("iso, kernel", [
    ("1700-01-01T00:00:00", "de440.bsp"),
    ("2500-01-01T00:00:00", "de440.bsp"),
    ("1000-01-01T00:00:00", "de441_part-1.bsp"),
    ("3000-01-01T00:00:00", "de441_part-2.bsp"),
])
def test_long_span_epochs_route_to_full_de440_then_de441(monkeypatch, iso, kernel):
    # With downloads disabled the error names the kernel that would be fetched:
    # full DE440 for 1550-2650 outside de440s, DE441 beyond (NAIF coverage).
    monkeypatch.setenv("SSAPY_EPHEMERIS_DOWNLOAD", "0")
    with pytest.raises(ephemeris.EphemerisUnavailableError, match=kernel.replace(".", r"\.")):
        MoonPosition()(_gps(iso))


def test_offline_or_disabled_downloads_fail_clearly(monkeypatch, tmp_path):
    _harness(monkeypatch, tmp_path)
    t = _gps("2100-06-01T00:00:00")

    def offline(url):
        raise OSError("network unreachable")

    monkeypatch.setattr(ephemeris, "_open_url", offline)
    with pytest.warns(ephemeris.EphemerisDownloadWarning):
        with pytest.raises(ephemeris.EphemerisUnavailableError, match="Could not download long_test.bsp"):
            MoonPosition()(t)
    assert not os.listdir(tmp_path / "cache")  # no partial file left

    monkeypatch.setenv("SSAPY_EPHEMERIS_DOWNLOAD", "0")
    with pytest.raises(ephemeris.EphemerisUnavailableError, match="downloads are disabled"):
        MoonPosition()(t)


def test_a_corrupt_download_is_rejected(monkeypatch, tmp_path):
    _harness(monkeypatch, tmp_path, served=b"not a kernel")
    with pytest.warns(ephemeris.EphemerisDownloadWarning):
        with pytest.raises(ephemeris.EphemerisUnavailableError, match="SHA-256"):
            MoonPosition()(_gps("2100-06-01T00:00:00"))
    assert not os.path.exists(tmp_path / "cache" / "long_test.bsp")


@pytest.mark.skipif(os.environ.get("SSAPY_TEST_NETWORK") != "1", reason="set SSAPY_TEST_NETWORK=1 to download from NAIF")
@pytest.mark.timeout(600)
def test_real_download_of_full_de440():
    # Downloads de440.bsp (119.8 MB) from NAIF, checks its SHA-256, and
    # evaluates the Moon in 1700, outside de440s.bsp.
    with pytest.warns(ephemeris.EphemerisDownloadWarning, match="de440.bsp"):
        position = MoonPosition()(_gps("1700-01-01T00:00:00"))
    assert 3.5e8 < np.linalg.norm(position) < 4.1e8
