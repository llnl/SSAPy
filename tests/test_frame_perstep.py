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

ISS = ("1 25544U 98067A   24015.54791435  .00016717  00000-0  30074-3 0  9993",
       "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49514637123456")
GEO = ("1 41866U 16071A   24015.50000000 -.00000267  00000-0  00000+0 0  9990",
       "2 41866   0.0177  86.6394 0001679 191.2716 168.7788  1.00272058 26978")


def test_teme_to_gcrf_accepts_time_array():
    """teme_to_gcrf must vectorize over time and match a scalar loop exactly."""
    t = 1.1e9 + np.array([0.0, 3600.0, 7200.0, 43200.0])
    stacked = teme_to_gcrf(t)
    assert stacked.shape == (4, 3, 3)
    loop = np.array([teme_to_gcrf(ti) for ti in t])
    assert np.array_equal(stacked, loop)
    assert teme_to_gcrf(t[0]).shape == (3, 3)   # scalar unchanged


def _astropy_teme_to_gcrf(tle, gps):
    """Independent TEME->GCRF via astropy, per output time (ground truth)."""
    coords = pytest.importorskip("astropy.coordinates")
    TEME = coords.TEME
    from astropy.coordinates import GCRS, CartesianRepresentation
    sat = Satrec.twoline2rv(*tle)
    epoch = Time(sat.jdsatepoch, format="jd") + sat.jdsatepochF * u.d
    r_teme = np.array([np.array(sat.sgp4_tsince((g - epoch.gps) / 60.0)[1])
                       for g in gps])  # km, TEME
    obst = Time(gps, format="gps")
    teme = TEME(CartesianRepresentation(r_teme.T * u.km), obstime=obst)
    gcrf = teme.transform_to(GCRS(obstime=obst)).cartesian.xyz.T.to(u.km).value
    return gcrf * 1e3  # m


@pytest.mark.parametrize("tle,single_epoch_floor", [(ISS, 1.0), (GEO, 10.0)])
def test_per_timestep_beats_single_epoch_vs_astropy(tle, single_epoch_floor):
    """The default (per-timestep) frame handling agrees with astropy far better
    than the single-epoch approximation over a 12 h arc."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orb = Orbit.fromTLETuple(tle)
        gps = orb.t + np.array([0.0, 1.0, 3.0, 6.0, 12.0]) * 3600.0
        ref = _astropy_teme_to_gcrf(tle, gps)

        per_step = np.array([rv(orb, g, propagator=SGP4Propagator())[0]
                             for g in gps])
        single = np.array([rv(orb, g, propagator=SGP4Propagator(t=orb.t))[0]
                           for g in gps])

    e_step = np.linalg.norm(per_step - ref, axis=1)
    e_single = np.linalg.norm(single - ref, axis=1)
    # per-timestep tracks astropy to well under a metre (IAU-model floor)
    assert e_step.max() < 0.5
    # and is dramatically better than single-epoch by the end of the arc
    assert e_single.max() > single_epoch_floor
    assert e_step.max() < 0.1 * e_single.max()
