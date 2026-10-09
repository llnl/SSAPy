import numpy as np
import pytest

from ssapy import utils


@pytest.mark.parametrize("offset", [0.0, 0.05, 0.33, 0.8])
def test_tangent_plane_rates_are_the_derivative_of_the_projection(offset):
    # The projected rates must be the time derivative of the projected
    # position: a central difference of lb_to_tan for a point moving at
    # (mul, mub) on the sphere (mul = cos(b) dl/dt) agrees to 1e-8.
    lcen, bcen = np.array([1.0]), np.array([0.4])
    lb, b = np.array([1.0 + offset]), np.array([0.4 - 0.6 * offset])
    mul, mub = np.array([0.01]), np.array([-0.02])
    _x, _y, vx, vy = utils.lb_to_tan(lb, b, mul=mul, mub=mub, lcen=lcen, bcen=bcen)
    h = 1e-6
    xp, yp = utils.lb_to_tan(lb + mul / np.cos(b) * h, b + mub * h, lcen=lcen, bcen=bcen)
    xm, ym = utils.lb_to_tan(lb - mul / np.cos(b) * h, b - mub * h, lcen=lcen, bcen=bcen)
    np.testing.assert_allclose(vx, (xp - xm) / (2 * h), rtol=1e-8)
    np.testing.assert_allclose(vy, (yp - ym) / (2 * h), rtol=1e-8)
