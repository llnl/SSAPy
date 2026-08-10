import numpy as np
from astropy.time import Time
import astropy.units as u
from pathlib import Path

import ssapy
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


def test_lazy_property():
    class Foo:
        @utils.lazy_property
        def val(self):
            return 42
    f = Foo()
    assert f.val == 42
    f.__dict__['val'] = 100
    assert f.val == 100


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
