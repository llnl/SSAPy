import warnings

import numpy as np
import pytest
from sgp4.api import Satrec
from astropy.time import Time
from astropy import units as u

from ssapy import Orbit
from ssapy.propagator import SGP4Propagator
from ssapy.compute import rv
from ssapy.utils import teme_to_gcrf
from ssapy.tle_drag import (bstar_to_cd_a_over_m, numerical_from_tle,
                            _sgp4_reference_arc, fit_drag)

ISS = ("1 25544U 98067A   24015.54791435  .00016717  00000-0  30074-3 0  9993",
       "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49514637123456")
GEO = ("1 41866U 16071A   24015.50000000 -.00000267  00000-0  00000+0 0  9990",
       "2 41866   0.0177  86.6394 0001679 191.2716 168.7788  1.00272058 26978")


def _direct_sgp4_gcrf(tle, dt_minutes):
    sat = Satrec.twoline2rv(*tle)
    t0 = Time(sat.jdsatepoch, format="jd") + sat.jdsatepochF * u.d
    gps = t0.gps + np.asarray(dt_minutes, dtype=float) * 60.0
    rs = np.array([np.array(sat.sgp4_tsince(dt)[1]) * 1e3 for dt in dt_minutes])
    rot = teme_to_gcrf(gps)                      # per-output-time (vectorized)
    return np.einsum("nij,nj->ni", rot, rs)


def test_fromtletuple_retains_satrec():
    """Path A: the native SGP4 record and B* survive construction."""
    orb = Orbit.fromTLETuple(ISS)
    assert hasattr(orb, "_sat")
    assert hasattr(orb, "_tle")
    assert orb._sat.bstar == pytest.approx(3.0074e-4, rel=1e-6)


@pytest.mark.parametrize("tle", [ISS, GEO])
def test_path_a_matches_direct_sgp4(tle):
    """Path A reproduces direct sgp4 to sub-mm, including drag-sensitive LEO."""
    dts = [0.0, 60.0, 6 * 60.0, 12 * 60.0, 24 * 60.0]
    ref = _direct_sgp4_gcrf(tle, dts)
    orb = Orbit.fromTLETuple(tle)
    prop = SGP4Propagator()
    got = np.array([rv(orb, orb.t + dt * 60.0, propagator=prop)[0]
                    for dt in dts])
    err = np.linalg.norm(ref - got, axis=1)
    assert err.max() < 1e-3  # < 1 mm over a full day


def test_path_a_survives_scalar_vector_promotion():
    """_sat must survive scalar->vector->scalar round-trips, since compute.rv
    promotes a scalar Orbit internally before propagating."""
    from ssapy.compute import _countOrbit
    orb = Orbit.fromTLETuple(ISS)
    _, _, promoted = _countOrbit(orb)      # scalar -> vector
    assert hasattr(promoted, "_sat")
    assert next(iter(promoted)).__dict__.get("_sat") is not None  # vector -> scalar


def test_bstar_seed_is_physical_order():
    """B* -> Cd*A/m seed lands in a physically plausible range."""
    orb = Orbit.fromTLETuple(ISS)
    cd = bstar_to_cd_a_over_m(orb._sat.bstar)
    assert 1e-4 < cd < 1e-1


def test_path_b_fit_improves_residual():
    """Path B: fitting Cd*A/m against an arc reduces the residual and yields a
    sane physical coefficient (short arc for test speed)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orb0, _ = _sgp4_reference_arc(ISS, [0.0])
        t0 = orb0.t
        times = t0 + np.arange(0, 3 * 60 + 1, 30) * 60.0   # 3 h, 30-min steps
        _, r_ref = _sgp4_reference_arc(ISS, times)
        res = fit_drag(orb0, times, r_ref)
    assert res["rms_after"] <= res["rms_before"]
    assert 1e-4 < res["cd_a_over_m"] < 1e-1
    assert set(res["propkw"]) == {"area", "mass", "CD", "CR"}
