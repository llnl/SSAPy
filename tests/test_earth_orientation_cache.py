import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time

from ssapy.body import EarthOrientation


def _angle_arcsec(a, b):
    # Small-angle rotation between two frames; the Frobenius norm of
    # (A B^T - I) / sqrt(2) avoids arccos's loss of precision near 1.
    return np.degrees(np.linalg.norm(a @ b.T - np.eye(3)) / np.sqrt(2.0)) * 3600.0


@pytest.mark.parametrize("days", [1.0, 7.0, 29.0])
def test_cached_earth_orientation_matches_a_fresh_evaluation(days):
    # A long-lived EarthOrientation (as held by Body / AccelHarmonic over a
    # propagation) must agree with a freshly built one to 0.005 arcsec at any
    # later time, and with astropy's GCRS -> ITRS rotation to 0.5 arcsec (the
    # IAU 1980 / 2000A model difference; measured 0.27 arcsec). The old 30-day
    # cache drifted 0.06, 1.0 and 5.1 arcsec after 1, 7 and 29 days.
    t0 = Time("2026-10-08T00:00:00", scale="utc").gps
    held = EarthOrientation()
    held(t0)
    t = t0 + days * 86400.0
    assert _angle_arcsec(held(t), EarthOrientation()(t)) < 0.005
    r = np.array([7000e3, 1000e3, 500e3])
    when = Time(t, format="gps")
    expected = GCRS(CartesianRepresentation(r * u.m), obstime=when).transform_to(ITRS(obstime=when)).cartesian.xyz.to_value(u.m)
    got = held(t) @ r
    assert np.degrees(np.linalg.norm(np.cross(got, expected)) / (np.linalg.norm(got) * np.linalg.norm(expected))) * 3600.0 < 0.5
