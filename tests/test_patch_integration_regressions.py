from collections import deque
from dataclasses import dataclass, field
import unittest
from unittest.mock import patch

import numpy as np

from ssapy.constants import EARTH_RADIUS
from ssapy.propagator import (Propagator, SciPyPropagator, RK4Propagator, RK8Propagator,
                              RK78Propagator, LeapfrogPropagator, Leapfrog4Propagator)
from ssapy.compute import rv, HashableArrayContainer, _countTime


@dataclass(eq=False)
class LinearState:
    r: np.ndarray
    v: np.ndarray
    t: float = 0.
    propkw: dict = field(default_factory=dict)


class ZeroAcceleration:
    time_breakpoints = np.array([-np.inf, np.inf])
    def __call__(self, r, v, t, **kwargs):
        return np.zeros(3)


class EpochCoverageTests(unittest.TestCase):
    def states(self):
        radius = EARTH_RADIUS + 100e3
        return [LinearState(np.array([radius + 1000, 0, 0]), np.array([100., 0, 0])),
                LinearState(np.array([radius + 1500, 0, 0]), np.array([-100., 0, 0]))]

    def test_forward_backward_batch_matches_retained_epochs(self):
        # One arc ends backward at -10 s, the other forward at +15 s.
        # Include out-of-order and repeated query epochs to forbid prefix heuristics.
        times = np.array([20., -20., 0., 5., -5., 0.])
        for prop in (SciPyPropagator(ZeroAcceleration(), {'max_step': 1.}),
                     RK4Propagator(ZeroAcceleration(), 1.)):
            with self.subTest(propagator=type(prop).__name__):
                states = self.states()
                r, v, valid = prop._getRVManyWithMask(states, times)
                np.testing.assert_array_equal(valid, [False, False, True, True, True, True])
                for index, state in enumerate(states):
                    expected = state.r + times[valid, None] * state.v
                    np.testing.assert_allclose(r[index], expected, atol=1e-6, rtol=0)
                    np.testing.assert_allclose(v[index], np.broadcast_to(state.v, expected.shape), atol=1e-9)

    def test_public_rv_returns_matching_epochs(self):
        states = self.states()
        times = np.array([-20., -5., 0., 5., 20.])
        with patch('ssapy.compute._countOrbit', return_value=(2, False, states)):
            r, v, retained = rv(states, times, SciPyPropagator(ZeroAcceleration()), return_times=True)
        np.testing.assert_array_equal(retained, [-5., 0., 5.])
        np.testing.assert_allclose(r[0], states[0].r + retained[:, None] * states[0].v, atol=1e-6)

    def test_legacy_prefix_subclasses_remain_compatible(self):
        class Prefix(Propagator):
            def _getRVOne(self, orbit, time):
                return np.ones((orbit, 3)), np.zeros((orbit, 3))
        self.assertEqual(Prefix()._getRVMany([3, 1], np.arange(3.))[0].shape, (2, 1, 3))

    def test_cached_termination_is_not_restarted_from_rounded_event_state(self):
        for epoch in (1.4e9, 3.8e9):
            for direction in (-1, 1):
                with self.subTest(epoch=epoch, direction=direction):
                    state = LinearState(np.array([EARTH_RADIUS + 101e3, 0, 0]),
                                        np.array([-direction * 123.456, 0, 0]), epoch)
                    prop = SciPyPropagator(ZeroAcceleration(), {'max_step': 3.1})
                    query = epoch + direction * np.arange(0., 25., 2.)
                    first_r, first_v = prop._getRVOne(state, query)
                    self.assertEqual(len(first_r), 5)
                    with patch.object(prop, '_solve_piecewise_ivp', side_effect=AssertionError('restarted terminated arc')):
                        second_r, second_v = prop._getRVOne(state, query)
                    np.testing.assert_array_equal(second_r, first_r)
                    np.testing.assert_array_equal(second_v, first_v)

    def test_termination_at_one_end_does_not_block_other_end(self):
        state = self.states()[0]
        prop = SciPyPropagator(ZeroAcceleration())
        prop._getRVOne(state, np.array([-20., 0.]))
        r, v = prop._getRVOne(state, np.array([0., 20.]))
        np.testing.assert_allclose(r[-1], state.r + 20 * state.v, atol=1e-6)

    def test_empty_scalar_after_termination_is_not_squeezed_to_nonexistent_state(self):
        states = self.states()[:1]
        with patch('ssapy.compute._countOrbit', return_value=(1, True, states)):
            r, v, retained = rv(states[0], -20., SciPyPropagator(ZeroAcceleration()), return_times=True)
        self.assertEqual(r.shape, (0, 3))
        self.assertEqual(retained.shape, (0,))


class ToleranceTests(unittest.TestCase):
    def test_invalid_fixed_steps_are_rejected_before_entering_step_loops(self):
        for cls in (RK4Propagator, RK8Propagator, RK78Propagator,
                    LeapfrogPropagator, Leapfrog4Propagator):
            for h in (0., -1., np.nan, np.inf):
                with self.subTest(propagator=cls.__name__, h=h), self.assertRaises(ValueError):
                    cls(ZeroAcceleration(), h)

    def test_rk78_accepts_documented_array_tolerance_without_cache_failure(self):
        tol = np.array([1e-6] * 3 + [1e-9] * 3)
        prop = RK78Propagator(ZeroAcceleration(), 1., tol)
        same = RK78Propagator(prop.accel, 1., list(tol))
        self.assertEqual(hash(prop), hash(same))
        self.assertEqual(prop, same)
        tol[:] = 100
        self.assertEqual(prop.tol[0], 1e-6)
        state = LinearState(np.array([7e6, 0, 0]), np.array([0., 7500., 0.]))
        r, v = prop._getRVOne(state, np.array([0., 1., 2.]))
        np.testing.assert_allclose(r, state.r + np.arange(3.)[:, None] * state.v, atol=1e-6)

    def test_scipy_component_atol_is_hashable_and_owned(self):
        tolerance = np.ones(6) * 1e-9
        options = {'atol': tolerance, 'rtol': 1e-8}
        prop = SciPyPropagator(ZeroAcceleration(), options)
        key = hash(prop)
        tolerance[:] = 1
        options['rtol'] = 0.1
        self.assertEqual(hash(prop), key)
        self.assertEqual(prop.ode_kwargs['atol'][0], 1e-9)

    def test_invalid_rk78_tolerance_and_nonadvancing_step_fail(self):
        for tol in (0., -1., np.nan, np.inf):
            with self.subTest(tol=tol), self.assertRaises(ValueError):
                RK78Propagator(ZeroAcceleration(), 1, tol)
        for cls in (RK4Propagator, RK8Propagator, RK78Propagator,
                    LeapfrogPropagator, Leapfrog4Propagator):
            prop = cls(ZeroAcceleration(), 1e-12)
            times = deque([1.4e9])
            states = deque([np.array([7e6, 0, 0, 0, 7500, 0.])])
            with self.subTest(propagator=cls.__name__), self.assertRaisesRegex(RuntimeError, 'cannot advance'):
                prop._prop(times, states, 1e-12, 1.4e9 + 1, {})


class CacheInputTests(unittest.TestCase):
    def test_mutating_returned_states_does_not_corrupt_the_rv_cache(self):
        state = LinearState(np.array([7e6, 0, 0]), np.array([0, 7500., 0]))
        prop = SciPyPropagator(ZeroAcceleration())
        times = np.array([0., 1.])
        with patch('ssapy.compute._countOrbit', return_value=(1, True, (state,))):
            first_r, first_v = rv(state, times, prop)
            expected_r, expected_v = first_r.copy(), first_v.copy()
            first_r /= 1000.
            first_v[:] = 0.
            second_r, second_v = rv(state, times, prop)
        np.testing.assert_array_equal(second_r, expected_r)
        np.testing.assert_array_equal(second_v, expected_v)
        self.assertTrue(times.flags.writeable)

    def test_time_lists_are_converted_and_cache_keys_do_not_freeze_callers(self):
        _, _, values = _countTime([0., 1., 2.])
        key = HashableArrayContainer(values)
        old_hash = hash(key)
        values[:] = 7
        self.assertEqual(hash(key), old_hash)
        np.testing.assert_array_equal(key.arr, [0., 1., 2.])

    def test_cache_equality_respects_shape_dtype_and_hash(self):
        key = HashableArrayContainer(np.array([0., 1.]))
        same = HashableArrayContainer(np.array([0., 1.]))
        self.assertEqual(key, same)
        self.assertEqual(hash(key), hash(same))
        self.assertNotEqual(key, HashableArrayContainer(np.array([[0., 1.]])))
        self.assertNotEqual(key, HashableArrayContainer(np.array([0, 1])))
        self.assertNotEqual(key, object())
