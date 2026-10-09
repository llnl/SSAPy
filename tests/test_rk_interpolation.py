import numpy as np
import pytest

import ssapy
from ssapy import AccelKepler


@pytest.mark.parametrize("h", [40.0, 20.0])
def test_rk8_is_as_accurate_between_steps_as_at_them(h):
    # Reference: the Keplerian propagator. RK8 with a fixed step is accurate
    # to ~1e-8 m at its step boundaries over 6000 s in LEO; queried between
    # boundaries it must stay within 1e-6 m in position and 1e-6 m/s in
    # velocity (measured 1.1e-7 m at h = 40 s). The cubic spline used before
    # gave 2.5e-2 m at h = 40 s.
    t0 = 1.4e9
    orbit = ssapy.Orbit.fromKeplerianElements(7500e3, 0.05, 0.9, 0.3, 1.2, 0.4, t=t0)
    times = t0 + np.array([0.0, 1234.5, 3333.3, 5999.9])
    r_ref, v_ref = ssapy.rv(orbit, times, propagator=ssapy.KeplerianPropagator())
    r, v = ssapy.rv(orbit, times, propagator=ssapy.RK8Propagator(AccelKepler(), h=h))
    assert np.max(np.linalg.norm(r - r_ref, axis=1)) < 1e-6
    assert np.max(np.linalg.norm(v - v_ref, axis=1)) < 1e-6
