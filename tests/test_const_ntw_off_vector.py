import numpy as np

from ssapy.accel import AccelConstNTW, AccelKepler


def test_inactive_const_ntw_returns_zero_vector():
    burn = AccelConstNTW([0.0, 1e-5, 0.0], time_breakpoints=[10.0, 20.0])
    out = burn(np.array([7000e3, 0.0, 0.0]),
               np.array([0.0, 7500.0, 0.0]),
               0.0)

    assert out.shape == (3,)
    np.testing.assert_array_equal(out, np.zeros(3))


def test_inactive_const_ntw_can_be_first_term_in_acceleration_sum():
    burn = AccelConstNTW([0.0, 1e-5, 0.0], time_breakpoints=[10.0, 20.0])
    kepler = AccelKepler()
    combined = burn + kepler

    r = np.array([7000e3, 0.0, 0.0])
    v = np.array([0.0, 7500.0, 0.0])

    expected = kepler(r, v, 0.0)
    actual = combined(r, v, 0.0)

    np.testing.assert_allclose(actual, expected)
