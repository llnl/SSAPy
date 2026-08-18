import numpy as np
import astropy.units as u
from astropy.time import Time
import pytest

import ssapy
import ssapy.orbit_solver as orbit_solver
from ssapy.utils import normed
from .ssapy_test_helpers import timer, checkAngle


@timer
def testGauss():
    # Testing out Gauss algorithm.  Works for small fractions of an orbital period.
    np.random.seed(5)
    for _ in range(1000):
        a = np.random.uniform(7e6, 5e7)  # Roughly LEO to GEO
        e = np.random.uniform(0.02, 0.98)
        pa = np.random.uniform(0, 2*np.pi)
        raan = np.random.uniform(0, 2*np.pi)
        i = np.random.uniform(0, np.pi)
        trueAnomaly = np.random.uniform(0, 2*np.pi)
        orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, 0)
        # Pick two points close together in time, infer the orbit!
        t1 = np.random.uniform(0, orbit.period)
        t2 = np.random.uniform(t1, t1 + 0.001*orbit.period)
        r1, _ = ssapy.rv(orbit, t1)
        r2, _ = ssapy.rv(orbit, t2)
        solver = ssapy.GaussTwoPosOrbitSolver(r1, r2, t1, t2)
        orbit2 = solver.solve().at(0)

        np.testing.assert_allclose(orbit.a, orbit2.a, atol=1, rtol=0)
        np.testing.assert_allclose(orbit.e, orbit2.e, atol=1e-9, rtol=0)
        checkAngle(orbit.i, orbit2.i, atol=1e-9, rtol=0)
        checkAngle(orbit.pa, orbit2.pa, atol=1e-9, rtol=0)
        checkAngle(orbit.raan, orbit2.raan, atol=1e-9, rtol=0)
        checkAngle(orbit.trueAnomaly, orbit2.trueAnomaly, atol=2e-7, rtol=0)


@timer
def testDanchick():
    np.random.seed(57)
    for _ in range(1000):
        dnu = np.inf
        # Only valid for dnu < pi
        while dnu > np.pi:
            a = np.random.uniform(7e6, 5e7)  # Roughly LEO to GEO
            e = np.random.uniform(0.02, 0.98)
            pa = np.random.uniform(0, 2*np.pi)
            raan = np.random.uniform(0, 2*np.pi)
            i = np.random.uniform(0, np.pi)
            trueAnomaly = np.random.uniform(0, 2*np.pi)
            orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, 0)
            t1 = np.random.uniform(0, orbit.period)
            t2 = np.random.uniform(t1, t1 + 0.001*orbit.period)
            nu1 = orbit.at(t1).trueAnomaly
            nu2 = orbit.at(t2).trueAnomaly
            dnu = nu2 - nu1
        r1, v1 = ssapy.rv(orbit, t1)
        r2, v2 = ssapy.rv(orbit, t2)

        orbit2 = ssapy.DanchickTwoPosOrbitSolver(
            r1, r2, t1, t2).solve()
        np.testing.assert_allclose(r1, ssapy.rv(orbit2, t1)[0], atol=1e-2, rtol=0)
        np.testing.assert_allclose(r2, ssapy.rv(orbit2, t2)[0], atol=1e-2, rtol=0)
        np.testing.assert_allclose(v1, ssapy.rv(orbit2, t1)[1], atol=1e-2, rtol=0)
        np.testing.assert_allclose(v2, ssapy.rv(orbit2, t2)[1], atol=1e-2, rtol=0)


@timer
def testShefer():
    np.random.seed(577)
    nTest = 1000
    nRobust = 0
    for _ in range(nTest):
        a = np.random.uniform(7e6, 5e7)  # Roughly LEO to GEO
        # TODO: Allow e > 1
        e = np.random.uniform(0.02, 0.98)
        pa = np.random.uniform(0, 2*np.pi)
        raan = np.random.uniform(0, 2*np.pi)
        i = np.random.uniform(0, np.pi)
        trueAnomaly = np.random.uniform(0, 2*np.pi)
        orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, 0)
        t1 = np.random.uniform(0, orbit.period)
        t2 = np.random.uniform(t1, t1 + 0.001*orbit.period)

        r1, v1 = ssapy.rv(orbit, t1)
        r2, v2 = ssapy.rv(orbit, t2)
        orbit2 = ssapy.SheferTwoPosOrbitSolver(
            r1, r2, t1, t2, robust=True, nExam=200).solve()

        np.testing.assert_allclose(r1, ssapy.rv(orbit2, t1)[0], atol=1e-2, rtol=0)
        np.testing.assert_allclose(r2, ssapy.rv(orbit2, t2)[0], atol=1e-2, rtol=0)
        np.testing.assert_allclose(v1, ssapy.rv(orbit2, t1)[1], atol=1e-1, rtol=0)
        np.testing.assert_allclose(v2, ssapy.rv(orbit2, t2)[1], atol=1e-1, rtol=0)

    # Now let's try adding a few orbit wraps
    for _ in range(nTest):
        a = np.random.uniform(7e6, 5e7)  # Roughly LEO to GEO
        e = np.random.uniform(0.02, 0.98)
        pa = np.random.uniform(0, 2*np.pi)
        raan = np.random.uniform(0, 2*np.pi)
        i = np.random.uniform(0, np.pi)
        trueAnomaly = np.random.uniform(0, 2*np.pi)
        orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, 0)
        lam = np.random.randint(1, 5)
        t1 = np.random.uniform(0, 1)*orbit.period
        dt = np.random.uniform(lam, lam+1)*orbit.period
        t2 = t1 + dt
        r1, v1 = ssapy.rv(orbit, t1)
        r2, v2 = ssapy.rv(orbit, t2)

        orbit2 = ssapy.SheferTwoPosOrbitSolver(
            r1, r2, t1, t2, lam=lam).solve()
        r1_test, _ = ssapy.rv(orbit2, t1)
        r2_test, _ = ssapy.rv(orbit2, t2)
        # Sometimes the initial guess doesn't work well, so check we can solve robustly in those
        # cases.
        try:
            np.testing.assert_allclose(r1, r1_test, atol=1, rtol=0)
            np.testing.assert_allclose(r2, r2_test, atol=1, rtol=0)
        except AssertionError:
            nRobust += 1
            orbit2 = ssapy.SheferTwoPosOrbitSolver(
                r1, r2, t1, t2, lam=lam, robust=True, nExam=500).solve()
            r1_test, _ = ssapy.rv(orbit2, t1)
            r2_test, _ = ssapy.rv(orbit2, t2)
            np.testing.assert_allclose(r1, r1_test, atol=1, rtol=0)
            np.testing.assert_allclose(r2, r2_test, atol=1, rtol=0)

    print("Reverted to robust methods {} of {} times.".format(nRobust, nTest))


@timer
def testKappaSignPlane():
    np.random.seed(5772)
    for _ in range(1000):
        a = np.random.uniform(7e6, 5e7)  # Roughly LEO to GEO
        e = np.random.uniform(0.02, 0.98)
        pa = np.random.uniform(0, 2*np.pi)
        raan = np.random.uniform(0, 2*np.pi)
        i = np.random.uniform(0, np.pi)
        trueAnomaly = np.random.uniform(0, 2*np.pi)
        orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, 0)
        t1 = np.random.uniform(0, orbit.period)
        t2 = np.random.uniform(0, orbit.period)
        r1, _ = ssapy.rv(orbit, t1)
        r2, _ = ssapy.rv(orbit, t2)
        solver1 = ssapy.SheferTwoPosOrbitSolver(r1, r2, t1, t2)
        solver2 = ssapy.SheferTwoPosOrbitSolver(
            r1, r2, t1, t2, kappaSign=-1)
        # Orbital plane is fixed regardless of kappaSign, but which node is labeled
        # ascending/descending are flipped, as is the inclination angle.
        np.testing.assert_allclose(
            solver1.raan, (solver2.raan+np.pi) % (2*np.pi), rtol=0, atol=1e-9)
        np.testing.assert_allclose(solver1.i, np.pi-solver2.i, rtol=0, atol=1e-9)


@timer
def testThreeAngles():
    np.random.seed(57721)
    failedSolve = 0
    failedTest = 0
    ntest = 1000
    for _ in range(ntest):
        a = np.random.uniform(7e6, 5e7)  # Roughly LEO to GEO
        e = np.random.uniform(0.02, 0.98)
        pa = np.random.uniform(0, 2*np.pi)
        raan = np.random.uniform(0, 2*np.pi)
        i = np.random.uniform(0, np.pi)
        trueAnomaly = np.random.uniform(0, 2*np.pi)
        orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, 0)
        # Observation points.  Roughly uniformly distributed within cube
        # surrounding earth
        R1 = np.random.uniform(-7e6, 7e6, size=3)
        R2 = np.random.uniform(-7e6, 7e6, size=3)
        R3 = np.random.uniform(-7e6, 7e6, size=3)
        # Algorithm only seems stable for relatively small time separations
        t1 = np.random.uniform(0, orbit.period)
        dt21 = np.random.uniform(0, orbit.period/10)
        t2 = t1 + dt21
        dt32 = np.random.uniform(0, orbit.period/10)
        t3 = t2 + dt32
        r1, _ = ssapy.rv(orbit, t1)
        r2, _ = ssapy.rv(orbit, t2)
        r3, _ = ssapy.rv(orbit, t3)
        e1 = normed(r1 - R1)
        e2 = normed(r2 - R2)
        e3 = normed(r3 - R3)
        solver = ssapy.ThreeAngleOrbitSolver(
            e1, e2, e3, R1, R2, R3, t1, t2, t3)

        # Basic dot/cross product orthogonality check
        np.testing.assert_allclose(
            np.dot(solver.d1, solver.e2), 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            np.dot(solver.d1, solver.e3), 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            np.dot(solver.d2, solver.e1), 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            np.dot(solver.d2, solver.e3), 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            np.dot(solver.d3, solver.e1), 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            np.dot(solver.d3, solver.e2), 0, rtol=0, atol=1e-10)

        assert solver.t21 > 0
        assert solver.t32 > 0
        assert solver.t31 > 0

        try:
            orbit2 = solver.solve()
        except ValueError:
            failedSolve += 1
            continue
        try:
            r1_test, _ = ssapy.rv(orbit2, t1)
            r2_test, _ = ssapy.rv(orbit2, t2)
            r3_test, _ = ssapy.rv(orbit2, t3)
        except RuntimeError:
            # Count these as failures too
            failedTest += 1
            continue
        e1_test = normed(r1_test-R1)
        e2_test = normed(r2_test-R2)
        e3_test = normed(r3_test-R3)
        try:
            np.testing.assert_allclose(
                e1, e1_test, rtol=0, atol=5e-6)  # ~arcsec-ish
            np.testing.assert_allclose(e2, e2_test, rtol=0, atol=5e-6)
            np.testing.assert_allclose(e3, e3_test, rtol=0, atol=5e-6)
        except AssertionError:
            failedTest += 1
            # print("----- Failed test in three angle orbit solver -----")
            # print("Times deltas:", dt21, dt32)
            # print(orbit)
            # print(orbit2)
            continue

    print("ThreeAngleOrbitSolver failed to solve {} times out of {}".format(
        failedSolve, ntest))
    print("ThreeAngleOrbitSolver failed test {} times out of {}".format(
        failedTest, ntest))


@timer
def test_MG_2_6():
    """Exercise 2.6 from Montenbruck and Gill

    Tests orbit determination from two position vectors
    """
    # Elements provided in MG as a solution
    a_ref = 28196776.0 # meters
    e_ref = 0.7679436
    i_ref = np.deg2rad(20.315)
    Omega_ref = np.deg2rad(359.145)
    omega_ref = np.deg2rad(179.425)
    M0_ref = np.deg2rad(29.236)

    # Specify satellite positions at two times
    r1 = np.array([11959978.0, -16289478.0, -5963827.0])
    r2 = np.array([39863390.0, -13730547.0, -4862350.0])
    t1 = Time(2455198.0, format='jd')
    t2 = t1 + 2.5*u.hour

    orbit = ssapy.SheferTwoPosOrbitSolver(r1, r2, t1, t2).solve()

    # Test that determined elements are close to reference values
    np.testing.assert_allclose(orbit.a, a_ref, atol=1e-1, rtol=0)
    np.testing.assert_allclose(orbit.e, e_ref, atol=1e-6, rtol=0)
    np.testing.assert_allclose(orbit.i, i_ref, atol=1e-5, rtol=0)
    np.testing.assert_allclose(orbit.pa, omega_ref, atol=1e-5, rtol=0)
    np.testing.assert_allclose(orbit.raan, Omega_ref, atol=1e-5, rtol=0)
    np.testing.assert_allclose(orbit.meanAnomaly, M0_ref, atol=1e-5, rtol=0)


def test_two_position_solver_edge_branches(monkeypatch):
    class ProbeSolver(ssapy.TwoPosOrbitSolver):
        def _getP(self):
            return ssapy.TwoPosOrbitSolver._getP(self)

    with pytest.raises(NotImplementedError):
        ProbeSolver(
            np.array([7000e3, 0.0, 0.0]),
            np.array([0.0, 7000e3, 0.0]),
            0.0,
            10.0,
        )._getP()

    gauss = object.__new__(ssapy.GaussTwoPosOrbitSolver)
    gauss.kappa = gauss.sigma = gauss.tau = gauss.mu = 1.0
    gauss.eps = 0.0
    gauss.maxiter = 1
    gauss.m = 1.0
    gauss.ell = 2.0
    assert np.isfinite(ssapy.GaussTwoPosOrbitSolver._getP(gauss))
    gauss.ell = 1.0
    assert np.isfinite(ssapy.GaussTwoPosOrbitSolver._getP(gauss))

    danchick = object.__new__(ssapy.DanchickTwoPosOrbitSolver)
    danchick.kappa = danchick.sigma = danchick.tau = danchick.mu = 1.0
    danchick.eps = 1e-12
    danchick.maxiter = 20
    danchick.m = 1.0
    danchick.ell = 0.0
    danchick.cos2f = -1.0
    assert np.isfinite(ssapy.DanchickTwoPosOrbitSolver._getP(danchick))

    danchick.maxiter = 3
    danchick.m = 10.0
    danchick.ell = 0.0
    monkeypatch.setattr(ssapy.DanchickTwoPosOrbitSolver, 'X', staticmethod(lambda g: 1.0))
    monkeypatch.setattr(ssapy.DanchickTwoPosOrbitSolver, 'dXdg', staticmethod(lambda g: 0.0))
    with pytest.raises(RuntimeError, match='Invalid x'):
        ssapy.DanchickTwoPosOrbitSolver._getP(danchick)

    danchick.cos2f = np.nan
    with pytest.raises(ValueError, match='Invalid value of cos2f'):
        ssapy.DanchickTwoPosOrbitSolver._getP(danchick)

    monkeypatch.undo()
    danchick = object.__new__(ssapy.DanchickTwoPosOrbitSolver)
    danchick.kappa = danchick.sigma = danchick.tau = danchick.mu = 1.0
    danchick.eps = 1e-12
    danchick.maxiter = 1
    danchick.m = 3.0
    danchick.ell = 1.0
    danchick.cos2f = 1.0
    calls = {'n': 0}
    real_sqrt = np.sqrt

    def fake_sqrt(x):
        calls['n'] += 1
        if calls['n'] <= 2:
            return 1.0
        return real_sqrt(x)

    monkeypatch.setattr(np, 'sqrt', fake_sqrt)
    with pytest.raises(RuntimeError, match='Invalid x'):
        ssapy.DanchickTwoPosOrbitSolver._getP(danchick)


def test_shefer_static_and_error_branches(monkeypatch):
    val, grad = ssapy.SheferTwoPosOrbitSolver.X(-0.25 * u.one)
    assert np.isfinite(val)
    assert np.isfinite(grad)

    val0, grad0 = ssapy.SheferTwoPosOrbitSolver.X(0.0)
    np.testing.assert_allclose([val0, grad0], [4.0 / 3.0, 8.0 / 5.0])
    assert np.isinf(ssapy.SheferTwoPosOrbitSolver.X(1.0))

    shefer = object.__new__(ssapy.SheferTwoPosOrbitSolver)
    shefer.lam = 0
    shefer.rbar = 10.0
    shefer.kappa = 2.0
    shefer.tau = 1.0
    shefer.mu = 1.0
    shefer.eps = 1e-12
    shefer.maxiter = 10

    monkeypatch.setattr(np, 'roots', lambda poly: np.array([1.0, 2.0, -3.0]))
    with pytest.raises(RuntimeError, match='more than one positive'):
        ssapy.SheferTwoPosOrbitSolver._getInitialXGuess(shefer)

    monkeypatch.setattr(np, 'roots', lambda poly: np.array([-1.0, -2.0, -3.0]))
    with pytest.raises(RuntimeError, match='no positive real roots'):
        ssapy.SheferTwoPosOrbitSolver._getInitialXGuess(shefer)

    monkeypatch.setattr(np, 'roots', lambda poly: np.array([1.0, 2.0, -3.0]))
    with pytest.raises(RuntimeError, match='more than one positive'):
        ssapy.SheferTwoPosOrbitSolver._getInitialXiGuess(shefer)

    monkeypatch.setattr(np, 'roots', lambda poly: np.array([-1.0, -2.0, -3.0]))
    with pytest.raises(RuntimeError, match='no positive real roots'):
        ssapy.SheferTwoPosOrbitSolver._getInitialXiGuess(shefer)


def test_shefer_robust_failure_and_three_angle_time_conversion(monkeypatch):
    shefer = object.__new__(ssapy.SheferTwoPosOrbitSolver)
    shefer.robust = True
    shefer.r1 = np.array([10.0, 0.0, 0.0])
    shefer.r2 = np.array([0.0, 10.0, 0.0])
    shefer.t1 = 10.0
    shefer.t2 = 20.0
    fake_orbit = object()

    monkeypatch.setattr(orbit_solver.TwoPosOrbitSolver, 'solve', lambda self: fake_orbit)
    monkeypatch.setattr(orbit_solver, 'rv', lambda orbit, t: (np.zeros(3), np.zeros(3)))
    monkeypatch.setattr(shefer, '_getAllP', lambda: [1.0, 2.0])
    monkeypatch.setattr(shefer, '_finishOrbit', lambda p: fake_orbit)

    with pytest.raises(RuntimeError, match='Cannot find orbit'):
        ssapy.SheferTwoPosOrbitSolver.solve(shefer)

    t1 = Time(1.0, format='gps')
    t2 = Time(2.0, format='gps')
    t3 = Time(3.0, format='gps')
    solver = ssapy.ThreeAngleOrbitSolver(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.zeros(3),
        np.ones(3),
        np.ones(3) * 2,
        t1,
        t2,
        t3,
    )
    assert solver.t1 == pytest.approx(t1.gps)
    assert solver.t2 == pytest.approx(t2.gps)
    assert solver.t3 == pytest.approx(t3.gps)


if __name__ == '__main__':
    testGauss()
    testDanchick()
    testShefer()
    testKappaSignPlane()
    testThreeAngles()
    test_MG_2_6()
