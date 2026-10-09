import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import get_body_barycentric
from astropy.time import Time

from ssapy import utils


def test_fast_sun_position_stays_within_an_arcminute_from_1980_to_2060():
    # Reference: astropy's geometric geocentric Sun (built-in ephemeris).
    # Montenbruck & Gill's series with the Earth-Moon barycentre's perihelion
    # advance (0.32327364 deg/century, JPL approximate elements) stays within
    # 1 arcmin in direction (measured 0.58 arcmin) and 3e-4 in distance over
    # 1980-2060; without the advance it reached 11.6 arcmin.
    worst = 0.0
    for jd in np.linspace(Time("1980-01-01", scale="tt").jd, Time("2060-01-01", scale="tt").jd, 200):
        t = Time(jd, format="jd", scale="tt")
        expected = (get_body_barycentric("sun", t) - get_body_barycentric("earth", t)).xyz.to_value(u.m)
        position = np.asarray(utils.sunPos(t.gps, fast=True), dtype=float).reshape(3)
        cosine = np.dot(position, expected) / (np.linalg.norm(position) * np.linalg.norm(expected))
        worst = max(worst, np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) * 60.0)
        assert np.linalg.norm(position) == pytest.approx(np.linalg.norm(expected), rel=3e-4)
    assert worst < 1.0
