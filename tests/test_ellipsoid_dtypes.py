import numpy as np

from ssapy.ellipsoid import Ellipsoid


def test_sphere_to_cart_coerces_integer_inputs_to_float64():
    ellipsoid = Ellipsoid(Req=6378137.0, f=1 / 298.257223563)
    lon = np.array([0, 0], dtype=np.int64)
    lat = np.array([0, 0], dtype=np.int64)
    height = np.array([0, 1000], dtype=np.int64)

    x, y, z = ellipsoid.sphereToCart(lon, lat, height)

    assert x.dtype == np.float64
    assert y.dtype == np.float64
    assert z.dtype == np.float64
    np.testing.assert_allclose(x, [6378137.0, 6379137.0], atol=1e-9)
    np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(z, [0.0, 0.0], atol=1e-12)


def test_cart_to_sphere_coerces_integer_inputs_to_float64():
    ellipsoid = Ellipsoid(Req=6378137.0, f=1 / 298.257223563)
    x = np.array([6378137, 6379137], dtype=np.int64)
    y = np.array([0, 0], dtype=np.int64)
    z = np.array([0, 0], dtype=np.int64)

    lon, lat, height = ellipsoid.cartToSphere(x, y, z)

    assert lon.dtype == np.float64
    assert lat.dtype == np.float64
    assert height.dtype == np.float64
    np.testing.assert_allclose(lon, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(lat, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(height, [0.0, 1000.0], atol=1e-6)
