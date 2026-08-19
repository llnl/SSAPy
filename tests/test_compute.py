import numpy as np
from astropy.time import Time
import astropy.units as u
import pytest
import sys
import subprocess

import ssapy
from ssapy import compute
from ssapy.compute import groundTrack, radecRateObsToRV, rvObsToRaDecRate


def test_groundTrack_invalid_format():
    orbit = np.zeros((1, 3, 3))
    time = Time([0, 100, 200], format="gps")

    with pytest.raises(ValueError, match="Format must be either 'cartesian' or 'geodetic'"):
        groundTrack(orbit, time, format="invalid")


def _patch_identity_ground_track_frame(monkeypatch):
    def identity_pn(jd0, mjd_tt):
        return np.broadcast_to(np.eye(3), (len(np.atleast_1d(mjd_tt)), 3, 3))

    def zero_like_time(time):
        return np.zeros_like(np.asarray(time, dtype=float))

    monkeypatch.setattr("ssapy.compute._gpsToTT", lambda time: np.asarray(time, dtype=float))
    monkeypatch.setattr("ssapy.compute._iers_interp", lambda time: (zero_like_time(time), zero_like_time(time), zero_like_time(time)))
    monkeypatch.setattr("ssapy.compute.erfa.pnm80", identity_pn)
    monkeypatch.setattr("ssapy.compute.erfa.gst94", lambda jd0, mjd_ut1: zero_like_time(mjd_ut1))


def test_groundTrack_cartesian_matches_identity_frame(monkeypatch):
    _patch_identity_ground_track_frame(monkeypatch)
    r = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    time = np.array([0.0, 1.0])

    x, y, z = groundTrack(r, time, format="cartesian")

    np.testing.assert_allclose(x, r[:, 0])
    np.testing.assert_allclose(y, r[:, 1])
    np.testing.assert_allclose(z, r[:, 2])


def test_groundTrack_geodetic_uses_ellipsoid_conversion(monkeypatch):
    _patch_identity_ground_track_frame(monkeypatch)

    class FakeEllipsoid:
        def cartToSphere(self, x, y, z):
            return x + 10.0, y + 20.0, z + 30.0

    monkeypatch.setattr("ssapy.compute._Ellipsoid", FakeEllipsoid)
    r = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    time = np.array([0.0, 1.0])

    lon, lat, height = groundTrack(r, time, format="geodetic")

    np.testing.assert_allclose(lon, [[11.0, 14.0]])
    np.testing.assert_allclose(lat, [[22.0, 25.0]])
    np.testing.assert_allclose(height, [[33.0, 36.0]])


def test_groundTrack_accepts_orbit_inputs(monkeypatch):
    _patch_identity_ground_track_frame(monkeypatch)
    orbit = ssapy.Orbit(
        np.array([7.0e6, 0.0, 0.0]),
        np.array([0.0, 7.5e3, 0.0]),
        0.0,
    )
    time = np.array([0.0, 1.0])

    x, y, z = groundTrack(orbit, time, format="cartesian")
    r_expected, _ = ssapy.rv(orbit, time)

    np.testing.assert_allclose(x, r_expected[:, 0])
    np.testing.assert_allclose(y, r_expected[:, 1])
    np.testing.assert_allclose(z, r_expected[:, 2])


def test_groundTrack_rejects_invalid_position_shape(monkeypatch):
    _patch_identity_ground_track_frame(monkeypatch)
    with pytest.raises(ValueError, match="Incorrect r dimensions"):
        groundTrack(np.zeros((2, 2)), np.array([0.0, 1.0]), format="cartesian")


def test_process_observer_validation_paths():
    time = np.array([0.0, 1.0])
    obs_pos = np.zeros((2, 3))
    obs_vel = np.zeros((2, 3))

    with pytest.raises(ValueError, match="Exactly one of obsPos and observer"):
        compute._processObserver(None, None, None, 2, False, time)
    with pytest.raises(ValueError, match="Exactly one of obsVel and observer"):
        compute._processObserver(None, obs_pos, None, 2, False, time, doObsVel=True)

    observer = ssapy.EarthObserver(lon=0.0, lat=0.0, elevation=0.0, fast=True)
    with pytest.raises(ValueError, match="Exactly one of obsPos and observer"):
        compute._processObserver(observer, obs_pos, None, 2, False, time)
    with pytest.raises(ValueError, match="Exactly one of obsVel and observer"):
        compute._processObserver(observer, None, obs_vel, 2, False, time, doObsVel=True)

    observers = [observer, observer]
    with pytest.raises(ValueError, match="observer and time must be broadcastable"):
        compute._processObserver(observers, None, None, 3, False, np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="obsPos and time must be broadcastable"):
        compute._processObserver(None, np.zeros((2, 3)), None, 3, False, np.array([0.0, 1.0, 2.0]))


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


def test_angle_rate_rv_round_trip_with_observer_velocity():
    r = np.array([[7.0e6, 1.0e6, 2.0e6]])
    v = np.array([[500.0, 7.5e3, -100.0]])
    obs_pos = np.array([[6.0e6, -2.0e6, 0.5e6]])
    obs_vel = np.array([[10.0, 20.0, 30.0]])

    ra, dec, slant_range, ra_rate, dec_rate, slant_range_rate = rvObsToRaDecRate(
        r,
        v,
        obsPos=obs_pos,
        obsVel=obs_vel,
    )
    r_round, v_round = radecRateObsToRV(
        ra,
        dec,
        slant_range,
        raRate=ra_rate,
        decRate=dec_rate,
        slantRangeRate=slant_range_rate,
        obsPos=obs_pos,
        obsVel=obs_vel,
    )

    np.testing.assert_allclose(r_round, r, atol=1e-9)
    np.testing.assert_allclose(v_round, v, atol=1e-12)


def test_radecRate_wrapper_matches_radec_rate_output():
    orbit = ssapy.Orbit(
        np.array([7.0e6, 0.0, 0.0]),
        np.array([0.0, 7.5e3, 0.0]),
        0.0,
    )
    time = np.array([0.0, 10.0])
    obs_pos = np.zeros((2, 3))
    obs_vel = np.zeros((2, 3))

    with pytest.warns(UserWarning, match="deprecated"):
        ra_rate, dec_rate, slant_rate = compute.radecRate(
            orbit,
            time,
            obsPos=obs_pos,
            obsVel=obs_vel,
        )
    expected = compute.radec(orbit, time, obsPos=obs_pos, obsVel=obs_vel, rate=True)[3:]

    np.testing.assert_allclose(ra_rate, expected[0])
    np.testing.assert_allclose(dec_rate, expected[1])
    np.testing.assert_allclose(slant_rate, expected[2])


def test_light_time_correction_error_paths():
    r = np.zeros((1, 1, 3))
    v = np.zeros((1, 1, 3))
    obs_pos = np.zeros((1, 3))
    obs_vel = np.zeros((1, 3))

    with pytest.raises(ValueError, match="Invalid value"):
        compute._obsAngleCorrection(r, v, obs_pos, obs_vel, [], np.array([0.0]), None, "bad")
    with pytest.raises(RuntimeError, match="did not converge"):
        compute._obsAngleCorrection(r, v, obs_pos, obs_vel, [], np.array([0.0]), None, "exact", max_iter=-1)


def test_radec_rate_requires_observer_velocity():
    orbit = ssapy.Orbit(
        np.array([7.0e6, 0.0, 0.0]),
        np.array([0.0, 7.5e3, 0.0]),
        0.0,
    )
    with pytest.raises(ValueError, match="obsVel"):
        compute.radec(orbit, np.array([0.0]), obsPos=np.zeros((1, 3)), rate=True)


def test_quickAltAz_invokes_requested_angle_correction(monkeypatch):
    orbit = ssapy.Orbit(
        np.array([7.0e6, 0.0, 0.0]),
        np.array([0.0, 7.5e3, 0.0]),
        0.0,
    )
    observer = ssapy.EarthObserver(lon=0.0, lat=0.0, elevation=0.0, fast=True)
    called = {}

    def fake_correction(r, v, obs_pos, obs_vel, orbit_arg, time_arg, propagator, correction_type):
        called["correction_type"] = correction_type
        return r, v

    monkeypatch.setattr(compute, "_obsAngleCorrection", fake_correction)
    alt, az = compute.quickAltAz(orbit, np.array([0.0]), observer, obsAngleCorrection="linear")

    assert called == {"correction_type": "linear"}
    assert np.isfinite(alt)
    assert np.isfinite(az)


def test_rvObsToRaDecRate_defaults_observer_position_to_origin():
    r = np.array([[1.0, 1.0, np.sqrt(2.0)]])
    v = np.array([[0.0, 1.0, 0.0]])

    ra, dec, slant_range, ra_rate, dec_rate, slant_rate = rvObsToRaDecRate(r, v)

    assert np.isclose(ra[0], np.pi / 4)
    assert np.isclose(dec[0], np.pi / 4)
    assert np.isclose(slant_range[0], 2.0)
    assert np.all(np.isfinite([ra_rate[0], dec_rate[0], slant_rate[0]]))


def test_earthShadowCoords_accepts_time_and_vector_positions():
    time = Time([0.0, 10.0], format="gps")
    r = np.array([[7.0e6, 0.0, 0.0], [0.0, 7.0e6, 0.0]])

    r_par, r_perp = compute.earthShadowCoords(r, time)

    assert r_par.shape == (2,)
    assert r_perp.shape == (2,)
    assert np.all(np.isfinite(r_par))
    assert np.all(r_perp >= 0.0)


def test_find_and_refine_pass_quantity_inputs(monkeypatch):
    orbit = object()
    observer = object()

    def fake_quick_alt_az(orbit_arg, times, observer_arg, propagator=None):
        times = np.asarray(times, dtype=float)
        alt = np.where(times < 2.0, np.deg2rad(5.0), np.deg2rad(30.0))
        return alt, np.zeros_like(alt)

    monkeypatch.setattr(compute, "quickAltAz", fake_quick_alt_az)
    passes = compute.find_passes(
        orbit,
        [observer],
        Time(0.0, format="gps"),
        4.0 * u.s,
        1.0 * u.s,
        horizon=20.0 * u.deg,
    )

    assert len(passes[observer]) == 1
    assert np.isclose(passes[observer][0].gps, 2.0)

    class FakeObserver:
        def sunAlt(self, time):
            return 0.0

    def smooth_quick_alt_az(orbit_arg, times, observer_arg, propagator=None):
        times = np.asarray(times, dtype=float)
        return 1.0 - (times / 100.0) ** 2, np.zeros_like(times)

    monkeypatch.setattr(compute, "quickAltAz", smooth_quick_alt_az)
    monkeypatch.setattr(compute, "rv", lambda orbit_arg, time_arg, propagator=None: (np.array([1.0, 0.0, 0.0]), np.zeros(3)))
    monkeypatch.setattr(compute, "earthShadowCoords", lambda r_arg, time_arg: (1.0, ssapy.constants.EARTH_RADIUS + 1.0))

    refined = compute.refine_pass(
        orbit,
        FakeObserver(),
        Time(0.0, format="gps"),
        horizon=0.0 * u.deg,
        maxSpan=1000.0 * u.s,
    )

    assert np.isclose(refined["horizon"], 0.0)
    assert refined["duration"].to_value(u.s) > 0.0
    assert refined["tTerminator"] is None


def test_import_erfa_present():
    code = """
import sys
import types
import astropy.coordinates
import astropy.time
import astropy.units

fake_erfa = types.ModuleType("erfa")
sys.modules["erfa"] = fake_erfa
sys.modules.pop("astropy._erfa", None)

import ssapy.compute

assert ssapy.compute.erfa is fake_erfa
"""
    subprocess.run([sys.executable, "-c", code], check=True)
