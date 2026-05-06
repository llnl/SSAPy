import numpy as np
from astropy.time import Time
import pytest
import importlib
import sys
import types

from ssapy.compute import groundTrack, radecRateObsToRV


def test_groundTrack_invalid_format():
    orbit = np.zeros((1, 3, 3))
    time = Time([0, 100, 200], format="gps")

    with pytest.raises(ValueError, match="Format must be either 'cartesian' or 'geodetic'"):
        groundTrack(orbit, time, format="invalid")


def mock_lb_to_unit(ra, dec):
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=-1)


@pytest.fixture(autouse=True)
def mock_compute_lb_to_unit(monkeypatch):
    monkeypatch.setattr("ssapy.compute.lb_to_unit", mock_lb_to_unit)


def test_radecRateObsToRV_without_obsVel():
    ra = np.array([0.1, 0.2])
    dec = np.array([0.3, 0.4])
    slantRange = np.array([1e7, 1e7])
    obsPos = np.array([[6371e3, 0, 0], [6371e3, 0, 0]])

    r, v = radecRateObsToRV(ra, dec, slantRange, obsPos=obsPos)

    assert r.shape == (2, 3)
    assert v is None
    assert np.allclose(r[0], obsPos[0] + mock_lb_to_unit(ra[0], dec[0]) * slantRange[0])


def test_radecRateObsToRV_requires_obsPos():
    ra = np.array([0.1])
    dec = np.array([0.2])
    slantRange = np.array([1e7])

    with pytest.raises(ValueError, match="obsPos must be set!"):
        radecRateObsToRV(ra, dec, slantRange)


MODULE_NAME = "ssapy.compute"


@pytest.fixture(autouse=True)
def cleanup_module_cache():
    if MODULE_NAME in sys.modules:
        del sys.modules[MODULE_NAME]
    yield
    if MODULE_NAME in sys.modules:
        del sys.modules[MODULE_NAME]


def test_import_erfa_present():
    fake_erfa = types.ModuleType("erfa")
    sys.modules["erfa"] = fake_erfa
    sys.modules.pop("astropy._erfa", None)

    import ssapy.compute
    importlib.reload(ssapy.compute)

    assert ssapy.compute.erfa is fake_erfa