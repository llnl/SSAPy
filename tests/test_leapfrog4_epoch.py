import numpy as np
import pytest

import ssapy
from ssapy import AccelKepler


@pytest.mark.parametrize("t0", [0.0, 1.4e9, 3.8e9])
def test_leapfrog4_is_fourth_order_at_real_gps_epochs(t0):
    # Reference: the Keplerian propagator. Halving the step must cut the error
    # by ~16x at any epoch (accept >= 12x) down to h = 5 s, where the error over
    # 6000 s is under 0.05 m. Summing substep times onto a large GPS epoch used
    # to stall the error near 2 m.
    orbit = ssapy.Orbit.fromKeplerianElements(7500e3, 0.05, 0.9, 0.3, 1.2, 0.4, t=t0)
    times = t0 + np.arange(0.0, 6000.0 + 1e-9, 600.0)
    r_ref, _ = ssapy.rv(orbit, times, propagator=ssapy.KeplerianPropagator())
    errors = []
    for h in (20.0, 10.0, 5.0):
        r, _ = ssapy.rv(orbit, times, propagator=ssapy.Leapfrog4Propagator(AccelKepler(), h=h))
        errors.append(np.max(np.linalg.norm(r - r_ref, axis=1)))
    assert errors[0] / errors[1] > 12 and errors[1] / errors[2] > 12
    assert errors[2] < 0.05
