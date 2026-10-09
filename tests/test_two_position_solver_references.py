import numpy as np
import pytest

import ssapy

T1 = 1.4e9
CASES = [  # (a [m], e, i [rad], arc [s])
    (7000e3, 0.01, 0.9, 600.0),       # LEO, 38 deg
    (7000e3, 0.01, 0.9, 4400.0),      # LEO, 270 deg (long way)
    (42164e3, 0.001, 0.1, 21600.0),   # GEO, 90 deg
    (20000e3, 0.4, 1.0, 9000.0),      # eccentric MEO, 134 deg
    (24400e3, 0.73, 0.5, 20000.0),    # GTO, 166 deg
]


def _endpoints(a, e, inc, arc):
    orbit = ssapy.Orbit.fromKeplerianElements(a, e, inc, 0.5, 1.0, 0.3, t=T1)
    r1, v1 = ssapy.rv(orbit, T1)
    r2, _ = ssapy.rv(orbit, T1 + arc)
    kappa_sign = 1 if np.dot(np.cross(r1, r2), np.cross(r1, v1)) > 0 else -1
    return r1, r2, v1, kappa_sign


@pytest.mark.parametrize("a, e, inc, arc", CASES)
def test_shefer_recovers_the_keplerian_velocity_for_short_and_long_arcs(a, e, inc, arc):
    # Reference: the Keplerian orbit that produced both positions. Shefer's
    # method recovers the departure velocity to 1e-9 m/s on every arc
    # (measured <= 6e-12 m/s).
    r1, r2, v1, kappa_sign = _endpoints(a, e, inc, arc)
    solved = ssapy.SheferTwoPosOrbitSolver(r1, r2, T1, T1 + arc, kappaSign=kappa_sign).solve()
    assert np.linalg.norm(solved.v - v1) < 1e-9


def test_gauss_warns_instead_of_returning_a_wrong_long_arc_orbit():
    # Gauss's iteration recovers a 38 deg LEO arc to 1e-9 m/s without a
    # warning, but diverges for a quarter GEO orbit; it must now warn there
    # (it used to return a velocity 2.7 km/s off silently).
    r1, r2, v1, kappa_sign = _endpoints(*CASES[0])
    solved = ssapy.GaussTwoPosOrbitSolver(r1, r2, T1, T1 + CASES[0][3], kappaSign=kappa_sign).solve()
    assert np.linalg.norm(solved.v - v1) < 1e-9
    r1, r2, v1, kappa_sign = _endpoints(*CASES[2])
    with pytest.warns(RuntimeWarning, match="did not converge"):
        ssapy.GaussTwoPosOrbitSolver(r1, r2, T1, T1 + CASES[2][3], kappaSign=kappa_sign).solve()


@pytest.mark.parametrize("a, e", [(7000e3, 0.01), (26560e3, 0.3), (42164e3, 0.001), (24400e3, 0.73)])
def test_danchick_and_shefer_cover_the_whole_revolution(a, e):
    # Reference: Keplerian truth. Over arcs from 2% to 98% of a period (both
    # short- and long-way, including fast perigee passages on a GTO) both
    # solvers recover the departure velocity to 1e-7 m/s (measured <= 3.6e-9).
    # Danchick's method used to raise "Invalid x" on 19 of these 100 arcs.
    orbit = ssapy.Orbit.fromKeplerianElements(a, e, 0.9, 0.5, 1.0, 0.3, t=T1)
    for fraction in np.linspace(0.02, 0.98, 25):
        arc = fraction * orbit.period
        r1, v1 = ssapy.rv(orbit, T1)
        r2, _ = ssapy.rv(orbit, T1 + arc)
        kappa_sign = 1 if np.dot(np.cross(r1, r2), np.cross(r1, v1)) > 0 else -1
        for solver in (ssapy.DanchickTwoPosOrbitSolver, ssapy.SheferTwoPosOrbitSolver):
            solved = solver(r1, r2, T1, T1 + arc, kappaSign=kappa_sign).solve()
            assert np.linalg.norm(solved.v - v1) < 1e-7, (solver.__name__, fraction)
