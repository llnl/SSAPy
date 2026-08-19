import numpy as np
import pytest
from astropy.time import Time
import astropy.units as u
from pathlib import Path

import ssapy
from ssapy.constants import EARTH_RADIUS
from ssapy import utils
from ssapy.utils import normed
from .ssapy_test_helpers import checkSphere, timer


def test_find_file_without_extension_finds_existing_file():
    path = utils.find_file("pyproject.toml")
    assert Path(path).name == "pyproject.toml"


def test_wrap_and_num_wraps():
    angles = np.array([4, -4, np.pi * 3])
    wrapped = utils._wrapToPi(angles)
    assert np.all((-np.pi <= wrapped) & (wrapped <= np.pi))

    assert utils.num_wraps(np.pi * 5) == 2
    assert utils.num_wraps(720 * u.deg) == 2


def test_emcee_version_parser_handles_unknown_versions():
    assert utils._emcee_version_before_3("2.2.1") is True
    assert utils._emcee_version_before_3("3.0.0") is False
    assert utils._emcee_version_before_3("not-a-version") is False


def test_norm_functions():
    v = np.array([[1.0, 2.0, 2.0]])
    assert np.isclose(utils.normSq(v), 9.0)
    assert np.isclose(utils.norm(v), 3.0)
    np.testing.assert_allclose(utils.normed(v), v / 3.0)

    a = np.random.randn(10, 3)
    np.testing.assert_allclose(utils.einsum_norm(a, 'ij,ij->i'), utils.norm(a))


def test_unit_angle():
    a = np.random.randn(10, 3)
    a = utils.normed(a)
    b = a.copy()
    np.testing.assert_allclose(utils.unitAngle3(a, b), 0.0)
    assert np.isclose(utils.unitAngle3([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]), np.pi)


def test_newton_raphson():
    f = lambda x: x**2 - 2
    fprime = lambda x: 2 * x
    root = utils.newton_raphson(1.0, f, fprime)
    assert np.isclose(root, np.sqrt(2), atol=1e-10)


def test_find_extrema_brackets():
    y = np.array([1, 2, 1, 0, -1, -2, -1])
    brackets = utils.find_extrema_brackets(y)
    assert len(brackets) > 0

    plateau = np.array([0.0, 1.0, 1.0, 0.0])
    assert utils.find_extrema_brackets(plateau) == [(0, 1, 3)]


def test_sample_points():
    x = np.array([0.0, 0.0])
    C = np.eye(2)
    samples = utils.sample_points(x, C, 100)
    assert samples.shape == (100, 2)


def test_sigma_points():
    x = np.array([1.0, 2.0])
    C = np.eye(2)
    f = lambda pts: pts @ np.array([1.0, 1.0])
    out = utils.sigma_points(f, x, C)
    assert out.shape[0] == 2 * len(x) + 1


def test_lru_cache():
    hits = []
    def f(x): hits.append(x); return x * 2
    cached = utils.LRU_Cache(f, maxsize=2)
    assert cached(2) == 4
    assert cached(2) == 4
    assert len(hits) == 1  # Cached

    cached(3)
    cached.resize(3)
    cached(4)
    assert len(cached.cache) == 3
    cached.resize(1)
    assert len(cached.cache) == 1
    cached.resize(1)
    assert len(cached.cache) == 1
    with pytest.raises(ValueError):
        cached.resize(-1)

    uncached = utils.LRU_Cache(f, maxsize=0)
    assert uncached(5) == 10
    assert uncached(5) == 10
    assert hits[-2:] == [5, 5]
    with pytest.raises(ValueError):
        utils.LRU_Cache(f, maxsize=-1)


def test_lazy_property():
    class Foo:
        @utils.lazy_property
        def val(self):
            return 42
    f = Foo()
    assert isinstance(Foo.__dict__['val'].__get__(None, Foo), utils.lazy_property)
    assert f.val == 42
    f.__dict__['val'] = 100
    assert f.val == 100


def test_sigma_and_unscented_branches():
    np.random.seed(0)
    x = np.array([1.0, 2.0])
    C = np.diag([4.0, 9.0])
    samples = utils.sample_points(x, C, 5, sqrt=True)
    assert samples.shape == (5, 2)

    sigma = utils.sigma_points(None, np.array([1.0, 2.0, 3.0]), np.eye(2), fixed_dimensions=[False, True, False])
    assert sigma.shape == (5, 3)
    np.testing.assert_allclose(sigma[:, 1], 2.0)

    mean, covar = utils.unscented_transform_mean_covar(lambda pts: pts, x, np.eye(2))
    np.testing.assert_allclose(mean, x)
    assert covar.shape == (2, 2)


def test_ntw_cartesian_round_trip():
    r = np.array([[7000.0, 0.0, 0.0]])
    v = np.array([[0.0, 7.5, 1.0]])
    offset = np.array([[5.0, 10.0, 15.0]])

    ntw = utils.rv_to_ntw(r, v, r + offset)
    np.testing.assert_allclose(utils.ntw_to_r(r, v, ntw), r + offset)
    np.testing.assert_allclose(utils.ntw_to_r(r, v, ntw, relative=True), offset)

    r_circular = np.array([[7000.0, 0.0, 0.0]])
    v_circular = np.array([[0.0, 7.5, 0.0]])
    components = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(
        utils.ntw_to_r(r_circular, v_circular, components, relative=True),
        [[1.0, 2.0, 3.0]],
    )
    np.testing.assert_allclose(
        utils.rv_to_ntw(r_circular, v_circular, r_circular + [[1.0, 2.0, 3.0]]),
        components,
    )


def test_small_coordinate_and_angle_helpers(capsys):
    np.testing.assert_allclose(utils.unit_vector(np.array([3.0, 4.0, 0.0])), [0.6, 0.8, 0.0])
    np.testing.assert_allclose(utils.get_angle([1, 0, 0], [0, 0, 0], [0, 1, 0]), [np.pi / 2])
    assert np.isclose(utils.dms_to_rad("30d00m00s"), np.pi / 6)
    np.testing.assert_allclose(utils.dms_to_deg(["30d00m00s", "-30d00m00s"]), [30.0, -30.0])
    assert utils.rad0to2pi(-np.pi / 2) == 3 * np.pi / 2
    assert utils.deg0to360([-10, 0, 370]) == [350, 0, 10]
    assert utils.deg0to360(-10) == 350
    assert utils.deg0to360array([-10, 370]) == [350, 10]
    assert utils.deg90to90([-100, 100, 45]) == [-10, 10, 45]
    assert utils.deg90to90(-100) == -10
    assert utils.deg90to90array([100, -100]) == [10, 80]

    az, el, radius = utils.cart2sph_deg(1.0, 1.0, 1.0)
    assert np.isclose(az, 45.0)
    assert np.isclose(el, 35.264389682754654)
    assert np.isclose(radius, np.sqrt(3.0))
    cyl_radius, theta, z = utils.cart_to_cyl(1.0, 1.0, 2.0)
    assert np.isclose(cyl_radius, np.sqrt(2.0))
    assert np.isclose(theta, np.pi / 4)
    assert z == 2.0

    assert np.isclose(utils.lonlat_distance(0.0, 0.0, 0.0, np.pi / 2), EARTH_RADIUS * np.pi / 2)
    assert utils.altitude_to_zenithangle(30.0) == 60.0
    assert utils.zenithangle_to_altitude(60.0) == 30.0
    assert np.isclose(utils.altitude_to_zenithangle(np.pi / 6, deg=False), np.pi / 3)
    assert np.isclose(utils.zenithangle_to_altitude(np.pi / 3, deg=False), np.pi / 6)
    assert utils.rightascension_to_hourangle(30.0, 2.0) == "0:0:0"
    assert utils.rightascension_to_hourangle("02:00:00", "01:00:00").count(":") == 2

    assert utils.dms_to_dd("12:30:00") == 12.5
    assert utils.dms_to_dd(["12:30:00", "-12:30:00"]) == [12.5, -12.5]
    assert utils.dd_to_dms(-12.5) == "-12:30:0"
    assert utils.hms_to_dd("12:00:00") == 180.0
    assert utils.hms_to_dd(["12:00:00", "01:30:00"]) == [180.0, 22.5]
    assert utils.dd_to_hms(-180.0) == "12:0:0"
    assert "cannot be negative" in capsys.readouterr().out


def test_sun_ra_dec_matches_astropy_solar_position():
    from astropy.coordinates import get_sun

    time = Time("J2000", scale="utc")
    ra, dec = utils.sun_ra_dec(time.mjd)
    sun = get_sun(time)

    dra = ((ra - sun.ra.rad + np.pi) % (2 * np.pi)) - np.pi
    separation = np.hypot(dra * np.cos(dec), dec - sun.dec.rad)
    assert separation < np.deg2rad(60.0 / 3600.0)


def test_sun_ra_dec_handles_vector_sunpos_shapes(monkeypatch):
    monkeypatch.setattr(utils, "sunPos", lambda time: np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    ra, dec = utils.sun_ra_dec(np.array([0.0, 1.0]))
    np.testing.assert_allclose(ra, [0.0, np.pi / 2])
    np.testing.assert_allclose(dec, [0.0, 0.0])

    monkeypatch.setattr(utils, "sunPos", lambda time: np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    ra, dec = utils.sun_ra_dec(np.array([0.0, 1.0]))
    np.testing.assert_allclose(ra, [0.0, np.pi / 2])
    np.testing.assert_allclose(dec, [0.0, 0.0])


def test_catalog_to_apparent_accepts_gps_float_time():
    ra, dec = utils.catalog_to_apparent(
        np.array([0.0]),
        np.array([0.0]),
        Time("J2000", scale="utc").gps,
        skipAberration=True,
    )
    np.testing.assert_allclose(ra, [0.0], atol=1e-12)
    np.testing.assert_allclose(dec, [0.0], atol=1e-12)


def test_ra_dec_accepts_documented_component_inputs():
    ra, dec = utils.ra_dec(
        x=1.0,
        y=2.0,
        z=3.0,
        vx=0.1,
        vy=0.2,
        vz=0.3,
    )
    np.testing.assert_allclose(ra, [np.arctan2(2.0, 1.0)])
    np.testing.assert_allclose(dec, [np.arcsin(3.0 / np.sqrt(14.0))])


def test_horizontal_to_equatorial_inverts_simple_transit_case():
    azimuth, altitude = utils.equatorial_to_horizontal(
        observer_latitude=45.0,
        declination=0.0,
        hour_angle=0.0,
    )
    hour_angle, declination = utils.horizontal_to_equatorial(
        observer_latitude=45.0,
        azimuth=azimuth,
        altitude=altitude,
    )

    wrapped_hour_angle = hour_angle % 360.0
    assert (
        np.isclose(wrapped_hour_angle, 0.0, atol=1e-12)
        or np.isclose(wrapped_hour_angle, 360.0, atol=1e-12)
    )
    assert np.isclose(declination, 0.0, atol=1e-12)


def test_find_all_zeros_recovers_polynomial_roots():
    roots = utils.find_all_zeros(lambda x: (x - 1.0) * (x - 3.0), 0.0, 4.0, n=9)
    np.testing.assert_allclose(roots, [1.0, 3.0], atol=1e-12)

    roots = utils.find_all_zeros(lambda x: (x - 0.2) ** 2 - 0.01, 0.0, 0.5, n=6)
    np.testing.assert_allclose(roots, [0.1, 0.3], atol=1e-12)

    roots = utils.find_all_zeros(lambda x: 0.01 - (x - 0.2) ** 2, 0.0, 0.5, n=6)
    np.testing.assert_allclose(roots, [0.1, 0.3], atol=1e-12)


def test_mcmc_chain_selection_helpers_are_deterministic_with_seed():
    chain = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    lnprob = np.array([[0.0, 0.0, 0.0, 0.0, -100.0]] * 4)
    lnprior = lnprob - 1.0

    clustered_chain, clustered_lnprob, clustered_lnprior = utils.cluster_emcee_walkers(
        chain,
        lnprob,
        lnprior,
    )
    assert clustered_chain.shape == (4, 3, 6)
    np.testing.assert_allclose(clustered_lnprob, 0.0)
    np.testing.assert_allclose(clustered_lnprior, -1.0)

    equal_lnprob = np.zeros((2, 3))
    equal_chain = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    equal_lnprior = equal_lnprob - 2.0
    out = utils.cluster_emcee_walkers(
        equal_chain,
        equal_lnprob,
        equal_lnprior,
        verbose=True,
    )
    assert out[0].shape == equal_chain.shape

    original_version_check = utils._emcee_version_before_3
    try:
        utils._emcee_version_before_3 = lambda version: True
        with pytest.raises(ValueError, match="emcee version"):
            utils.cluster_emcee_walkers(equal_chain, equal_lnprob, equal_lnprior)
    finally:
        utils._emcee_version_before_3 = original_version_check

    np.random.seed(0)
    samples, sample_lnprob, sample_lnprior = utils.subsample_high_lnprob(
        chain[:2, :4],
        np.array([[0.0, -1.0, -20.0, -30.0], [2.0, 1.0, 0.0, -50.0]]),
        np.array([[-100.0, -101.0, -120.0, -130.0], [-98.0, -99.0, -100.0, -150.0]]),
        nSample=3,
        thresh=-2.0,
    )
    assert samples.shape == (3, 6)
    assert np.all(sample_lnprob >= 0.0)
    np.testing.assert_allclose(sample_lnprior, sample_lnprob - 100.0)


def test_resample_non_pod_uses_full_particle_regularization(monkeypatch):
    particles = np.arange(12.0).reshape(3, 4)
    ln_weights = np.log([0.2, 0.3, 0.5])

    def fake_regularize_default(particles_arg, weights_arg):
        np.testing.assert_allclose(particles_arg, particles)
        np.testing.assert_allclose(weights_arg, [0.2, 0.3, 0.5])
        return np.ones_like(particles_arg), np.full(particles_arg.shape[0], 1.0 / particles_arg.shape[0])

    monkeypatch.setattr(utils, "regularize_default", fake_regularize_default)
    monkeypatch.setattr(utils.np.random, "uniform", lambda high: 0.0)

    resampled, weights = utils.resample(particles, ln_weights, pod=False)
    np.testing.assert_allclose(resampled, particles + 1.0)
    np.testing.assert_allclose(weights, np.full(3, 1.0 / 3.0))


def test_tangent_plane_and_simulation_frame_helpers():
    lb = utils.xyz_to_lb(1.0, 0.0, 0.0)
    np.testing.assert_allclose(lb, (0.0, 0.0))

    theta, phi = utils.xyz_to_tp(0.0, 0.0, 2.0)
    assert np.isclose(theta, 0.0)
    assert np.isclose(phi, 0.0)

    x, y, vx, vy = utils.lb_to_tan(
        np.array([0.0]),
        np.array([0.0]),
        mul=np.array([0.01]),
        mub=np.array([0.02]),
        lcen=np.array([0.0]),
        bcen=np.array([0.0]),
    )
    np.testing.assert_allclose((x, y, vx, vy), ([0.0], [0.0], [0.01], [0.02]))

    x_auto, y_auto = utils.lb_to_tan(np.array([0.0, 0.1]), np.array([0.0, 0.0]))
    assert x_auto.shape == (2,)
    assert y_auto.shape == (2,)

    xrot, yrot = utils.inert2rot(-1.0, 0.0, -1.0, 0.0)
    assert np.isclose(xrot, -1.0)
    assert np.isclose(yrot, 0.0, atol=1e-15)

    longitude, latitude, radius = utils.sim_lonlatrad(
        1.0, 2.0, 3.0,
        -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    )
    assert np.isclose(longitude, 315.0)
    assert np.isclose(latitude, np.degrees(np.arcsin(3.0 / np.sqrt(17.0))))
    assert np.isclose(radius, np.sqrt(17.0))


def test_angle_format_converters_carry_rounded_seconds():
    almost_13_deg = 12.0 + 59.0 / 60.0 + 59.99999 / 3600.0
    almost_2_hours = 15.0 * (1.0 + 59.0 / 60.0 + 59.99999 / 3600.0)

    assert utils.dd_to_dms(almost_13_deg) == "13:0:00"
    assert utils.dd_to_dms(-almost_13_deg) == "-13:0:00"
    assert utils.dd_to_hms(almost_2_hours) == "2:0:00"
    assert utils.dd_to_hms("15:00:00") == "1:0:0"


def test_ecliptic_equatorial_helpers_and_class_extension():
    xyz = (1.0, 2.0, 3.0)
    ecliptic = utils.equatorial_xyz_to_ecliptic_xyz(*xyz)
    np.testing.assert_allclose(utils.ecliptic_xyz_to_equatorial_xyz(*ecliptic), xyz)

    lon, lat = utils.xyz_to_ecliptic(0.0, 1.0, 0.0, degrees=True)
    assert np.isclose(lon, 90.0)
    assert np.isclose(lat, 0.0)
    ra, dec = utils.xyz_to_equatorial(0.0, 1.0, 0.0, degrees=True)
    assert np.isclose(ra, 90.0)
    assert np.isclose(dec, 0.0)
    ra2, dec2 = utils.ecliptic_xyz_to_equatorial(1.0, 0.0, 0.0, degrees=True)
    assert np.isclose(ra2, 0.0)
    assert np.isclose(dec2, 0.0)
    lon2, lat2 = utils.equatorial_to_ecliptic(0.0, 0.0, degrees=True)
    assert np.isclose(lon2, 0.0)
    assert np.isclose(lat2, 0.0)
    ra3, dec3 = utils.ecliptic_to_equatorial(0.0, 0.0, degrees=True)
    assert np.isclose(ra3, 0.0)
    assert np.isclose(dec3, 0.0)

    lon_rad, lat_rad = utils.xyz_to_ecliptic(0.0, 1.0, 0.0)
    np.testing.assert_allclose((lon_rad, lat_rad), (np.pi / 2, 0.0))
    ra_rad, dec_rad = utils.xyz_to_equatorial(0.0, 1.0, 0.0)
    np.testing.assert_allclose((ra_rad, dec_rad), (np.pi / 2, 0.0))
    ecliptic_ra, ecliptic_dec = utils.ecliptic_xyz_to_equatorial(1.0, 0.0, 0.0)
    np.testing.assert_allclose((ecliptic_ra, ecliptic_dec), (0.0, 0.0))

    lon4, lat4 = utils.equatorial_to_ecliptic(np.pi / 2, 0.0)
    ra4, dec4 = utils.ecliptic_to_equatorial(lon4, lat4)
    np.testing.assert_allclose((ra4, dec4), (np.pi / 2, 0.0), atol=1e-12)

    assert utils.isAttributeSafeToTransfer("__doc__", None) is False
    assert utils.isAttributeSafeToTransfer("new_method", object()) is True

    class _ContinueClassTarget:
        existing = "kept"

    setattr(__import__(__name__, fromlist=["_ContinueClassTarget"]), "_ContinueClassTarget", _ContinueClassTarget)

    @utils.continueClass
    class _ContinueClassTarget:
        @classmethod
        def added(cls):
            return cls.existing

    assert _ContinueClassTarget.added() == "kept"


def test_gps_to_tt_and_interpolate_points_between():
    t = Time(0.0, format="gps")
    assert np.isclose(utils._gpsToTT(t), utils._gpsToTT(0.0))
    assert utils.moonPos(t).shape == (3,)
    assert len(utils.iers_interp(t)) == 3

    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    interpolated = utils.interpolate_points_between(points, 2)
    np.testing.assert_allclose(
        interpolated,
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
    )


def test_coordinate_angle_edge_cases_and_errors(capsys):
    np.testing.assert_allclose(utils.dms_to_rad(["30d00m00s", "60d00m00s"]), [np.pi / 6, np.pi / 3])
    assert np.isclose(utils.dms_to_deg("30d00m00s"), 30.0)
    assert utils.deg90to90(190.0) == 10.0
    with pytest.raises(ValueError, match="hms cannot be negative"):
        utils.hms_to_dd("-01:00:00")

    with pytest.raises(ValueError, match="Either provide r and v"):
        utils.ra_dec(x=1.0, y=2.0, z=3.0)

    with pytest.raises(ValueError, match="Either right_ascension or hour_angle"):
        utils.equatorial_to_horizontal(45.0, 0.0)

    az, alt = utils.equatorial_to_horizontal(
        observer_latitude=-45.0,
        declination=0.0,
        hour_angle="00:00:00",
        hms=True,
    )
    assert np.isfinite(az)
    assert np.isfinite(alt)

    az2, alt2 = utils.equatorial_to_horizontal(
        observer_latitude=45.0,
        declination=0.0,
        right_ascension="01:00:00",
        hour_angle="00:00:00",
        local_time="02:00:00",
        hms=True,
    )
    assert np.isfinite(az2)
    assert np.isfinite(alt2)
    assert "Using hour_angle" in capsys.readouterr().out

    az3, alt3 = utils.equatorial_to_horizontal(
        observer_latitude=45.0,
        declination=0.0,
        right_ascension="01:00:00",
        local_time="02:00:00",
        hms=True,
    )
    assert np.isfinite(az3)
    assert np.isfinite(alt3)

    hour_angle, declination = utils.horizontal_to_equatorial(-45.0, az, alt)
    assert np.isfinite(hour_angle)
    assert np.isfinite(declination)

    hour_angle, declination = utils.horizontal_to_equatorial(45.0, 180.0, 30.0)
    assert np.isclose(hour_angle % 360.0, 0.0)
    assert declination < 0.0


@timer
def test_catalog_to_apparent():
    """No real test here, just want to make sure it runs vectorized"""
    size = 1_000_000
    ra = np.random.uniform(0.0, 2*np.pi, size=size)
    dec = np.arccos(np.random.uniform(-1.0, 1.0, size=size))
    pmra = np.random.uniform(-100.0, 100.0, size=size)
    pmdec = np.random.uniform(-100.0, 100.0, size=size)
    parallax = np.random.uniform(0.0, 0.1, size=size)
    t = Time("J2020")
    observer = ssapy.EarthObserver(lon=100., lat=10., elevation=100.)
    ra1, dec1 = ssapy.utils.catalog_to_apparent(ra, dec, t, skipAberration=True)
    ra2, dec2 = ssapy.utils.catalog_to_apparent(ra, dec, t, pmra=pmra, pmdec=pmdec, skipAberration=True)
    ra3, dec3 = ssapy.utils.catalog_to_apparent(ra, dec, t, parallax=parallax, skipAberration=True)
    ra4, dec4 = ssapy.utils.catalog_to_apparent(ra, dec, t)
    ra5, dec5 = ssapy.utils.catalog_to_apparent(ra, dec, t, observer=observer)
    ra6, dec6 = ssapy.utils.catalog_to_apparent(ra, dec, t, pmra=pmra, pmdec=pmdec, parallax=parallax, observer=observer)


@timer
def test_catalog_to_apparent_SOFA():
    """Checking against test case using SOFA library,
    where SOFA is the Standards of Fundamental Astronomy.
    """
    t = Time("2013-04-02T23:15:43.55", scale='utc')
    ra = np.array([np.deg2rad(15*(14+34/60+16.81183/3600))])
    dec = np.array([-np.deg2rad(12+31/60+10.3965/3600)])
    # Verify null transformation first
    ra1, dec1 = ssapy.utils.catalog_to_apparent(
        ra, dec, t, skipAberration=True
    )
    checkSphere(ra, dec, ra1, dec1, atol=1e-15)

    # Try proper motion
    pmra = -354.45
    pmdec = 595.35
    ra2, dec2 = ssapy.utils.catalog_to_apparent(
        ra, dec, t, pmra=pmra, pmdec=pmdec, skipAberration=True
    )
    ra2_SOFA = np.array([np.deg2rad(15*(14+34/60+16.4910486/3600))])
    dec2_SOFA = np.array([-np.deg2rad(12+31/60+2.506613/3600)])
    # milliarcsec precision
    checkSphere(ra2, dec2, ra2_SOFA, dec2_SOFA, atol=np.deg2rad(1e-5/3600))

    # Try parallax
    ra3, dec3 = ssapy.utils.catalog_to_apparent(
        ra, dec, t, parallax=0.16499, skipAberration=True
    )
    ra3_SOFA = np.array([np.deg2rad(15*(14+34/60+16.8168100/3600))])
    dec3_SOFA = np.array([-np.deg2rad(12+31/60+10.413678/3600)])
    checkSphere(ra3, dec3, ra3_SOFA, dec3_SOFA, atol=np.deg2rad(1e-5/3600))

    # Try aberration
    ra4, dec4 = ssapy.utils.catalog_to_apparent(
        ra, dec, t,
    )
    ra4_SOFA = np.array([np.deg2rad(15*(14+34/60+17.9779815/3600))])
    dec4_SOFA = np.array([-np.deg2rad(12+31/60+16.427072/3600)])
    checkSphere(ra4, dec4, ra4_SOFA, dec4_SOFA, atol=np.deg2rad(1e-3/3600))

    # Try all together
    ra5, dec5 = ssapy.utils.catalog_to_apparent(
        ra, dec, t, pmra=pmra, pmdec=pmdec, parallax=0.16499
    )
    ra5_SOFA = np.array([np.deg2rad(15*(14+34/60+17.6621826/3600))])
    dec5_SOFA = np.array([-np.deg2rad(12+31/60+08.554809/3600)])
    checkSphere(ra5, dec5, ra5_SOFA, dec5_SOFA, atol=np.deg2rad(1e-3/3600))


@timer
def test_angular_conversions():
    seed = 42
    np.random.seed(seed)
    npts = 10000
    uv = normed(np.random.randn(npts, 3))
    lb = utils.unit_to_lb(uv)
    tp = utils.unit_to_tp(uv)
    # round trips
    # 3 systems, back and forth to all other systems -> 6 tests.
    np.testing.assert_allclose(uv,
                               utils.lb_to_unit(*utils.unit_to_lb(uv)),
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(uv,
                               utils.tp_to_unit(*utils.unit_to_tp(uv)),
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.concatenate(tp),
                               np.concatenate(utils.unit_to_tp(utils.tp_to_unit(*tp))),
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.concatenate(tp),
                               np.concatenate(utils.lb_to_tp(*utils.tp_to_lb(*tp))),
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.concatenate(lb),
                               np.concatenate(utils.unit_to_lb(utils.lb_to_unit(*lb))),
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.concatenate(lb),
                               np.concatenate(utils.tp_to_lb(*utils.lb_to_tp(*lb))),
                               rtol=0, atol=1e-10)

    # check tangent plane round tripping.
    # this is just orthographic, so if you're on the wrong side of the globe
    # you won't round trip back to the right side.
    # so we need to make up some lcen, bcen to project from.
    noise = np.random.randn(npts, 3)*0.01
    uv2 = normed(uv + noise)
    lcen, bcen = utils.unit_to_lb(uv2)
    xy = utils.lb_to_tan(*lb, lcen=lcen, bcen=bcen)

    # vector lcen, bcen
    np.testing.assert_allclose(
        np.concatenate(lb),
        np.concatenate(utils.tan_to_lb(*utils.lb_to_tan(*lb, lcen=lcen, bcen=bcen),
                                     lcen=lcen, bcen=bcen)))
    np.testing.assert_allclose(
        np.concatenate(xy),
        np.concatenate(utils.lb_to_tan(*utils.tan_to_lb(*xy, lcen=lcen, bcen=bcen),
                                     lcen=lcen, bcen=bcen)))

    # single lcen, bcen; careful to choose all points to be on same hemisphere
    uv2 = uv.copy()
    uv2[:, 0] = np.abs(uv2[:, 0])
    lcen, bcen = (0, 0)
    lb2 = utils.unit_to_lb(uv2)
    xy2 = utils.lb_to_tan(*lb2, lcen=lcen, bcen=bcen)

    np.testing.assert_allclose(
        np.concatenate(lb2),
        np.concatenate(utils.tan_to_lb(*utils.lb_to_tan(*lb2, lcen=lcen, bcen=bcen),
                                     lcen=lcen, bcen=bcen)))
    np.testing.assert_allclose(
        np.concatenate(xy2),
        np.concatenate(utils.lb_to_tan(*utils.tan_to_lb(*xy2, lcen=lcen, bcen=bcen),
                                     lcen=lcen, bcen=bcen)))


if __name__ == '__main__':
    test_catalog_to_apparent()
    test_catalog_to_apparent_SOFA()
    test_angular_conversions()
