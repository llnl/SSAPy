import numpy as np
import pytest
from astropy.time import Time
import astropy.units as u
from collections import deque
from types import SimpleNamespace

import ssapy
from ssapy.constants import EARTH_RADIUS, MOON_RADIUS
from ssapy.propagator import (Propagator, RKPropagator, default_numerical,
                              impact_event, moon_impact_event)
from ssapy.utils import moonPos, norm


def test_abstract_propagator_classes_require_concrete_methods():
    with pytest.raises(TypeError):
        Propagator()
    with pytest.raises(TypeError):
        RKPropagator()


def test_abstract_propagator_hook_defaults_are_noops():
    class ConcretePropagator(Propagator):
        def _getRVOne(self, orbit, time):
            return super()._getRVOne(orbit, time)

    class ConcreteRKPropagator(RKPropagator):
        def _prop(self, *args, **kwargs):
            return super()._prop()

    assert ConcretePropagator()._getRVOne(None, None) is None
    assert ConcreteRKPropagator()._prop() is None


def test_base_get_rv_many_trims_to_shortest_result():
    class ShorteningPropagator(Propagator):
        def _getRVOne(self, orbit, time):
            n = orbit
            return np.ones((n, 3)) * n, np.ones((n, 3)) * -n

    r, v = ShorteningPropagator()._getRVMany([3, 1], np.arange(3.0))

    assert r.shape == (2, 1, 3)
    assert v.shape == (2, 1, 3)
    np.testing.assert_allclose(r[:, 0, 0], [3.0, 1.0])
    np.testing.assert_allclose(v[:, 0, 0], [-3.0, -1.0])


def test_chained_at_reuses_root_propagation_context():
    orbit = ssapy.Orbit(
        np.array([7.0e6, 0.0, 0.0]),
        np.array([0.0, 7.5e3, 0.0]),
        0.0,
    )
    propagator = ssapy.RK4Propagator(ssapy.AccelKepler(), h=70.0)
    times = np.array([0.0, 1000.0, 2500.0, 4000.0, 5500.0])
    r_expected, v_expected = ssapy.rv(orbit, times, propagator=propagator)

    current = orbit
    states = []
    for time in times:
        current = current.at(time, propagator=propagator)
        states.append((current.r, current.v))

    r_chained = np.array([state[0] for state in states])
    v_chained = np.array([state[1] for state in states])
    np.testing.assert_allclose(r_chained, r_expected, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(v_chained, v_expected, rtol=0.0, atol=1e-11)


@pytest.mark.parametrize(
    "orbit",
    [
        ssapy.Orbit.fromKeplerianElements(
            42_164e3,
            0.01,
            np.deg2rad(7.0),
            0.2,
            0.3,
            0.4,
            Time("J2000", scale="utc"),
        ),
        ssapy.Orbit.fromKeplerianElements(
            -5.0e9,
            1.02,
            np.deg2rad(20.0),
            0.2,
            0.3,
            0.4,
            Time("J2000", scale="utc"),
        ),
    ],
)
def test_keplerian_get_rv_one_matches_public_rv(orbit):
    times = Time(orbit.t, format="gps") + np.array([-60.0, 0.0, 120.0]) * u.s
    propagator = ssapy.KeplerianPropagator()

    r_direct, v_direct = propagator._getRVOne(orbit, times.gps)
    r_public, v_public = ssapy.rv(orbit, times, propagator=propagator)

    np.testing.assert_allclose(r_direct, r_public, rtol=0, atol=1e-8)
    np.testing.assert_allclose(v_direct, v_public, rtol=0, atol=1e-11)


def test_series_propagator_orders_match_closed_form_motion():
    orbit = ssapy.Orbit(
        np.array([7_000e3, 0.0, 0.0]),
        np.array([0.0, 7_500.0, 0.0]),
        0.0,
    )
    times = np.array([0.0, 1.0, 2.0])

    r0, v0 = ssapy.SeriesPropagator(0)._getRVOne(orbit, times)
    np.testing.assert_allclose(r0, np.broadcast_to(orbit.r, (3, 3)))
    np.testing.assert_allclose(v0, np.broadcast_to(orbit.v, (3, 3)))

    r1, v1 = ssapy.SeriesPropagator(1)._getRVOne(orbit, times)
    np.testing.assert_allclose(r1, orbit.r + times[:, None] * orbit.v)
    np.testing.assert_allclose(v1, np.broadcast_to(orbit.v, (3, 3)))

    accel = -orbit.mu * orbit.r / norm(orbit.r) ** 3
    r2, v2 = ssapy.SeriesPropagator(2)._getRVOne(orbit, times)
    np.testing.assert_allclose(
        r2,
        orbit.r + times[:, None] * (orbit.v + 0.5 * times[:, None] * accel),
    )
    np.testing.assert_allclose(v2, orbit.v + times[:, None] * accel)

    with pytest.raises(NotImplementedError):
        ssapy.SeriesPropagator(4)._getRVOne(orbit, times)


def test_propagator_equality_hash_and_repr_contracts():
    accel = ssapy.AccelKepler()

    assert ssapy.SeriesPropagator() == ssapy.SeriesPropagator(2)
    assert ssapy.SeriesPropagator(1) != ssapy.SeriesPropagator(2)
    assert ssapy.SeriesPropagator() != object()
    assert repr(ssapy.SeriesPropagator()) == "SeriesPropagator()"
    assert repr(ssapy.SeriesPropagator(1)) == "SeriesPropagator(1)"

    assert ssapy.SGP4Propagator() == ssapy.SGP4Propagator()
    assert ssapy.SGP4Propagator(Time(0.0, format="gps")).t == 0.0
    assert repr(ssapy.SGP4Propagator()) == "SGP4Propagator()"
    assert ssapy.SGP4Propagator() != object()
    assert ssapy.SGP4Propagator(truncate=False) != ssapy.SGP4Propagator(truncate=True)
    assert hash(ssapy.SGP4Propagator(truncate=False)) != hash(
        ssapy.SGP4Propagator(truncate=True)
    )

    scipy_a = ssapy.SciPyPropagator(accel, {"rtol": 1e-9})
    scipy_b = ssapy.SciPyPropagator(accel, {"rtol": 1e-9})
    scipy_c = ssapy.SciPyPropagator(accel, {"rtol": 1e-8})
    assert scipy_a == scipy_b
    assert scipy_a != scipy_c
    assert scipy_a != object()
    assert hash(scipy_a) == hash(scipy_b)
    assert repr(scipy_a) == "SciPyPropagator({!r}, {!r})".format(
        accel,
        {"rtol": 1e-9},
    )

    for cls in [ssapy.RK4Propagator, ssapy.RK8Propagator]:
        prop_a = cls(accel, h=10.0)
        prop_b = cls(accel, h=10.0)
        prop_c = cls(accel, h=20.0)
        assert prop_a == prop_b
        assert prop_a != prop_c
        assert prop_a != object()
        assert hash(prop_a) == hash(prop_b)
        assert repr(prop_a) == "{}({!r}, {!r})".format(cls.__name__, accel, 10.0)

    rk78_a = ssapy.RK78Propagator(accel, h=10.0, tol=(1e-6,) * 6)
    rk78_b = ssapy.RK78Propagator(accel, h=10.0, tol=(1e-6,) * 6)
    rk78_c = ssapy.RK78Propagator(accel, h=10.0, tol=(1e-5,) * 6)
    assert rk78_a == rk78_b
    assert rk78_a != rk78_c
    assert rk78_a != object()
    assert hash(rk78_a) == hash(rk78_b)
    assert repr(rk78_a) == "RK78Propagator({!r}, {!r}, {!r})".format(
        accel,
        10.0,
        (1e-6,) * 6,
    )

    leapfrog_a = ssapy.LeapfrogPropagator(accel, h=10.0)
    leapfrog_b = ssapy.LeapfrogPropagator(accel, h=10.0)
    leapfrog_c = ssapy.LeapfrogPropagator(accel, h=20.0)
    assert leapfrog_a == leapfrog_b
    assert leapfrog_a != leapfrog_c
    assert leapfrog_a != object()
    assert hash(leapfrog_a) == hash(leapfrog_b)
    assert repr(leapfrog_a) == "LeapfrogPropagator({!r}, {!r})".format(accel, 10.0)

    leapfrog4_a = ssapy.Leapfrog4Propagator(accel, h=10.0)
    leapfrog4_b = ssapy.Leapfrog4Propagator(accel, h=10.0)
    leapfrog4_c = ssapy.Leapfrog4Propagator(accel, h=20.0)
    assert leapfrog4_a == leapfrog4_b
    assert leapfrog4_a != leapfrog4_c
    assert leapfrog4_a != object()
    assert hash(leapfrog4_a) == hash(leapfrog4_b)
    assert repr(leapfrog4_a) == "Leapfrog4Propagator({!r}, {!r})".format(accel, 10.0)


def test_scipy_concatenate_ode_solution_edge_cases():
    from scipy.integrate._ivp.base import ConstantDenseOutput
    from scipy.integrate._ivp.common import OdeSolution

    sol0 = OdeSolution(
        np.array([1.0, 0.0]),
        [ConstantDenseOutput(1.0, 0.0, np.array([1.0]))],
    )
    sol1 = OdeSolution(
        np.array([1.0, 2.0]),
        [ConstantDenseOutput(1.0, 2.0, np.array([2.0]))],
    )
    merged = ssapy.SciPyPropagator._concatenateOdeSolutions(sol0, sol1)
    np.testing.assert_allclose(merged.ts, [0.0, 1.0, 2.0])

    sol1_desc = OdeSolution(
        np.array([2.0, 1.0]),
        [ConstantDenseOutput(2.0, 1.0, np.array([2.0]))],
    )
    merged = ssapy.SciPyPropagator._concatenateOdeSolutions(sol0, sol1_desc)
    np.testing.assert_allclose(merged.ts, [0.0, 1.0, 2.0])

    degenerate0 = OdeSolution(
        np.array([0.0, 0.0]),
        [ConstantDenseOutput(0.0, 0.0, np.array([0.0]))],
    )
    assert ssapy.SciPyPropagator._concatenateOdeSolutions(degenerate0, sol1) is sol1

    degenerate1 = OdeSolution(
        np.array([2.0, 2.0]),
        [ConstantDenseOutput(2.0, 2.0, np.array([0.0]))],
    )
    assert ssapy.SciPyPropagator._concatenateOdeSolutions(sol1, degenerate1) is sol1


def test_scipy_piecewise_ivp_reports_impact_and_failure(monkeypatch, capsys):
    from scipy.integrate._ivp.base import ConstantDenseOutput
    from scipy.integrate._ivp.common import OdeSolution
    import scipy.integrate

    accel = ssapy.AccelKepler()
    accel.time_breakpoints = np.array([-np.inf, np.inf])
    prop = ssapy.SciPyPropagator(accel)
    sol0 = OdeSolution(
        np.array([0.0, 0.0]),
        [ConstantDenseOutput(0.0, 0.0, np.zeros(6))],
    )

    def fake_success(fp, t_span, y0, dense_output=True, events=None, **kwargs):
        return SimpleNamespace(
            success=True,
            message="ok",
            t_events=[np.array([0.5])],
            sol=OdeSolution(
                np.array(t_span, dtype=float),
                [ConstantDenseOutput(t_span[0], t_span[1], np.ones(6))],
            ),
        )

    monkeypatch.setattr(scipy.integrate, "solve_ivp", fake_success)
    prop._solve_piecewise_ivp(lambda t, s: s, [0.0, 1.0], sol0)
    assert "Impact detected" in capsys.readouterr().out

    def fake_failure(fp, t_span, y0, dense_output=True, events=None, **kwargs):
        return SimpleNamespace(success=False, message="integration failed", t_events=[])

    monkeypatch.setattr(scipy.integrate, "solve_ivp", fake_failure)
    with pytest.raises(ValueError, match="integration failed"):
        prop._solve_piecewise_ivp(lambda t, s: s, [0.0, 1.0], sol0)


def test_scipy_get_rv_one_returns_empty_when_solution_does_not_cover_query(monkeypatch):
    from scipy.integrate._ivp.base import ConstantDenseOutput
    from scipy.integrate._ivp.common import OdeSolution

    prop = ssapy.SciPyPropagator(ssapy.AccelKepler())
    orbit = ssapy.Orbit(np.array([7.0e6, 0.0, 0.0]), np.array([0.0, 7.5e3, 0.0]), 0.0)
    stuck_solution = OdeSolution(
        np.array([0.0, 0.0]),
        [ConstantDenseOutput(0.0, 0.0, np.hstack([orbit.r, orbit.v]))],
    )

    monkeypatch.setattr(prop, "_solve_piecewise_ivp", lambda fp, t_span, sol: stuck_solution)
    r, v = prop._getRVOne(orbit, np.array([1.0]))

    assert r.shape == (0, 3)
    assert v.shape == (0, 3)


def test_scipy_burnup_returns_valid_state_prefix():
    accel = ssapy.AccelKepler() + ssapy.AccelDrag(
        CD=2.2, area=1.0, mass=100.0)
    prop = ssapy.SciPyPropagator(accel, {"rtol": 1e-8, "max_step": 5.0})
    orbit = ssapy.Orbit(
        np.array([EARTH_RADIUS + 200e3, 0.0, 0.0]),
        np.array([-1000.0, 0.0, 0.0]),
        0.0,
    )
    times = np.arange(0.0, 201.0, 20.0)

    r, v = ssapy.rv(orbit, times, propagator=prop)

    assert 0 < len(r) < len(times)
    assert np.all(np.linalg.norm(r, axis=1) - EARTH_RADIUS >= 100e3)
    assert v.shape == r.shape


def test_scipy_moon_collision_returns_valid_state_prefix():
    moon_r0 = moonPos(0.0)
    moon_v0 = moonPos(1.0) - moon_r0
    orbit = ssapy.Orbit(
        moon_r0 + np.array([MOON_RADIUS + 1000.0, 0.0, 0.0]),
        moon_v0 + np.array([-100.0, 0.0, 0.0]),
        0.0,
    )
    prop = ssapy.SciPyPropagator(
        ssapy.AccelKepler(), {"rtol": 1e-9, "max_step": 0.1})
    times = np.arange(0.0, 21.0, 1.0)

    r, v = ssapy.rv(orbit, times, propagator=prop)

    assert 0 < len(r) < len(times)
    distances = np.array([
        np.linalg.norm(state - moonPos(time)) - MOON_RADIUS
        for state, time in zip(r, times)
    ])
    assert np.all(distances >= -1.0)
    assert v.shape == r.shape


def test_rk_get_rv_one_single_point_cache_paths():
    class StaticRK(RKPropagator):
        _minPoints = 2
        h = 1.0

        def _prop(self, times, states, h, tthresh, propkw):
            return h

    orbit = ssapy.Orbit(np.array([7.0e6, 0.0, 0.0]), np.array([0.0, 7.5e3, 0.0]), 0.0)
    prop = StaticRK()

    r, v = prop._getRVOne(orbit, np.array([1.0]))
    assert r.shape == (0, 3)
    assert v.shape == (0, 3)

    r, v = prop._getRVOne(orbit, np.array([0.0, 0.0]))
    np.testing.assert_allclose(r, np.broadcast_to(orbit.r, (2, 3)))
    np.testing.assert_allclose(v, np.broadcast_to(orbit.v, (2, 3)))


def test_rk_get_rv_one_reuses_cache_and_handles_empty_filtered_query():
    class OneStepRK(RKPropagator):
        _minPoints = 2
        h = 1.0

        def _prop(self, times, states, h, tthresh, propkw):
            if h > 0 and len(times) == 1:
                times.append(times[-1] + h)
                states.append(states[-1] + np.ones(6))
            return h

    orbit = ssapy.Orbit(np.array([7.0e6, 0.0, 0.0]), np.array([0.0, 7.5e3, 0.0]), 0.0)
    prop = OneStepRK()

    r, v = prop._getRVOne(orbit, np.array([0.0, 1.0]))
    assert r.shape == (2, 3)
    r, v = prop._getRVOne(orbit, np.array([0.5]))
    assert r.shape == (1, 3)

    r, v = prop._getRVOne(orbit, np.array([-0.5]))
    assert r.shape == (0, 3)
    assert v.shape == (0, 3)


def test_rk_family_collision_stops_are_reported(capsys):
    accel = ssapy.AccelKepler()
    state = np.array([EARTH_RADIUS - 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for cls in [
        ssapy.RK4Propagator,
        ssapy.RK8Propagator,
        ssapy.RK78Propagator,
        ssapy.LeapfrogPropagator,
        ssapy.Leapfrog4Propagator,
    ]:
        prop = cls(accel, h=1.0)
        prop._prop(deque([0.0]), deque([state]), -1.0, -1.0, {})

    assert capsys.readouterr().out.count("Collision with Earth detected") == 5


def test_impact_event_and_default_numerical_factory():
    state = np.array([EARTH_RADIUS + 123.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert impact_event(0.0, state) == 123.0
    assert impact_event.terminal is True
    assert impact_event.direction == -1

    moon_state = np.hstack([moonPos(0.0) + [MOON_RADIUS + 123.0, 0.0, 0.0],
                             np.zeros(3)])
    assert moon_impact_event(0.0, moon_state) == pytest.approx(123.0)
    assert moon_impact_event.terminal is True
    assert moon_impact_event.direction == -1

    accel = ssapy.AccelKepler()
    propagator = default_numerical(10.0, cls=ssapy.RK4Propagator, accel=accel)
    assert propagator == ssapy.RK4Propagator(accel, 10.0)


def test_default_numerical_includes_scalar_extra_accel(monkeypatch):
    import ssapy.body
    import ssapy.gravity

    zero = ssapy.AccelKepler(mu=0.0)
    monkeypatch.setattr(ssapy.body, "get_body", lambda name: SimpleNamespace(mu=1.0))
    monkeypatch.setattr(ssapy.gravity, "AccelHarmonic", lambda earth, n, m: zero)
    monkeypatch.setattr(ssapy.gravity, "AccelThirdBody", lambda body: zero)

    extra = ssapy.AccelKepler(mu=2.0)
    prop = default_numerical(5.0, cls=ssapy.RK4Propagator, extra_accel=extra)

    assert isinstance(prop, ssapy.RK4Propagator)
    assert prop.h == 5.0
    assert prop.accel.accels[-1] is extra
