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


def test_newton_raphson():
    f = lambda x: x**2 - 2
    fprime = lambda x: 2 * x
    root = utils.newton_raphson(1.0, f, fprime)
    assert np.isclose(root, np.sqrt(2), atol=1e-10)


def test_find_extrema_brackets():
    y = np.array([1, 2, 1, 0, -1, -2, -1])
    brackets = utils.find_extrema_brackets(y)
    assert len(brackets) > 0


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

    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    interpolated = utils.interpolate_points_between(points, 2)
    np.testing.assert_allclose(
        interpolated,
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
    )


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
