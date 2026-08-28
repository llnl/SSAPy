import numpy as np

import ssapy.accel as accel
from ssapy.constants import EARTH_RADIUS


def test_solar_radiation_pressure_is_zero_in_cylindrical_earth_shadow(monkeypatch):
    sun_position = np.array([1.5e11, 0.0, 0.0])
    monkeypatch.setattr(accel, 'sunPos', lambda t: sun_position)

    model = accel.AccelSolRad(area=2.0, mass=100.0, CR=1.3)

    shadowed = np.array([-7.0e6, 0.0, 0.0])
    outside_shadow = np.array([-7.0e6, EARTH_RADIUS + 1.0e3, 0.0])
    sunward = np.array([7.0e6, 0.0, 0.0])

    np.testing.assert_array_equal(model(shadowed, None, 0.0), np.zeros(3))
    assert np.linalg.norm(model(outside_shadow, None, 0.0)) > 0.0
    assert np.linalg.norm(model(sunward, None, 0.0)) > 0.0


def test_solar_radiation_shadow_boundary_uses_earth_radius(monkeypatch):
    sun_position = np.array([1.5e11, 0.0, 0.0])
    monkeypatch.setattr(accel, 'sunPos', lambda t: sun_position)

    model = accel.AccelSolRad(area=1.0, mass=10.0, CR=1.0)
    on_cylinder = np.array([-1.0e7, EARTH_RADIUS, 0.0])
    just_outside = np.array([-1.0e7, EARTH_RADIUS + 1.0, 0.0])

    np.testing.assert_array_equal(model(on_cylinder, None, 0.0), np.zeros(3))
    assert np.linalg.norm(model(just_outside, None, 0.0)) > 0.0
