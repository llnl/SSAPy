"""Sexagesimal, hour-angle, horizon, and ecliptic conversions against references."""

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import Angle

from ssapy import utils


def test_negative_sub_degree_dms_keeps_its_sign():
    # The sign belongs to the whole angle: -00:30:00 is -0.5 deg (exact).
    for text, degrees in {"-00:30:00": -0.5, "-0:00:36": -0.01, "-10:30:36": -10.51, "+00:30:00": 0.5}.items():
        assert utils.dms_to_dd(text) == pytest.approx(degrees, abs=1e-12), text


@pytest.mark.parametrize("degrees", [-15.0, -236.375, 0.0, 15.0, 187.5, 359.99999999, 757.5])
def test_dd_to_hms_wraps_into_one_day_like_astropy(degrees):
    # astropy Angle.wrap_at(360 deg).hour gives the same instant within
    # 1e-4 s of time, and hours lie in [0, 24).
    hours, minutes, seconds = (float(part) for part in utils.dd_to_hms(degrees).split(":"))
    assert 0 <= hours < 24
    expected = Angle(degrees * u.deg).wrap_at(360 * u.deg).hour * 3600.0
    actual = hours * 3600 + minutes * 60 + seconds
    assert min(abs(actual - expected), 86400 - abs(actual - expected)) < 1e-4


def test_hour_angle_is_sidereal_time_minus_right_ascension_in_hours():
    # HA = LST - RA, wrapped to [0, 24) h (exact for these inputs).
    assert utils.rightascension_to_hourangle("10:30:00", "12:45:00") == "2:15:0"
    assert utils.rightascension_to_hourangle(157.5, 191.25) == "2:15:0"
    assert utils.rightascension_to_hourangle("23:00:00", "01:00:00") == "2:0:0"


@pytest.mark.parametrize("latitude", [37.68, -33.87])
@pytest.mark.parametrize("hour_angle", [-60.0, 0.0, 45.0, 120.0])
@pytest.mark.parametrize("declination", [-20.0, 20.0])
def test_horizon_conversions_match_astropy_altaz_geometry(latitude, hour_angle, declination):
    # The topocentric direction of (HA, dec) in the local east/north/up frame,
    # built from rotation matrices, gives the reference azimuth (from north
    # through east) and altitude; the forward and inverse transforms agree
    # with it to 1e-9 deg in both hemispheres.
    phi, h, d = np.radians([latitude, hour_angle, declination])
    east = -np.cos(d) * np.sin(h)
    north = np.sin(d) * np.cos(phi) - np.cos(d) * np.sin(phi) * np.cos(h)
    up = np.sin(d) * np.sin(phi) + np.cos(d) * np.cos(phi) * np.cos(h)
    az_ref = np.degrees(np.arctan2(east, north)) % 360.0
    alt_ref = np.degrees(np.arcsin(up))
    az, alt = utils.equatorial_to_horizontal(latitude, declination, hour_angle=hour_angle)
    assert alt == pytest.approx(alt_ref, abs=1e-9)
    if abs(alt_ref) < 89.9:
        assert ((az - az_ref + 180.0) % 360.0) - 180.0 == pytest.approx(0.0, abs=1e-9)
    ha_back, dec_back = utils.horizontal_to_equatorial(latitude, az, alt)
    assert dec_back == pytest.approx(declination, abs=1e-9)
    assert ((ha_back - hour_angle + 180.0) % 360.0) - 180.0 == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("ra, dec", [(30.0, 10.0), (120.0, 10.0), (200.0, -20.0), (300.0, 40.0)])
def test_ecliptic_conversions_match_the_obliquity_rotation(ra, dec):
    # Rotating the unit vector about +x by the J2000 obliquity (SSAPy's
    # cos_ec/sin_ec) gives the reference ecliptic longitude and latitude in
    # every quadrant; both directions agree with it to 1e-9 deg.
    ce, se = utils.cos_ec, utils.sin_ec
    ra_r, dec_r = np.radians([ra, dec])
    x = np.cos(dec_r) * np.cos(ra_r)
    y = ce * np.cos(dec_r) * np.sin(ra_r) + se * np.sin(dec_r)
    z = -se * np.cos(dec_r) * np.sin(ra_r) + ce * np.sin(dec_r)
    lon_ref, lat_ref = np.degrees(np.arctan2(y, x)) % 360.0, np.degrees(np.arcsin(z))
    lon, lat = utils.equatorial_to_ecliptic(ra, dec, degrees=True)
    assert lon == pytest.approx(lon_ref, abs=1e-9) and lat == pytest.approx(lat_ref, abs=1e-9)
    ra_back, dec_back = utils.ecliptic_to_equatorial(lon_ref, lat_ref, degrees=True)
    assert ra_back == pytest.approx(ra, abs=1e-9) and dec_back == pytest.approx(dec, abs=1e-9)
