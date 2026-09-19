# Copyright 2026 Lawrence Livermore National Security, LLC.
# See the top-level LICENSE file for details.
# SPDX-License-Identifier: MIT

"""Regression tests for the exact light-time convergence tolerance."""

import numpy as np
import pytest
from scipy.optimize import brentq

import ssapy
from ssapy import compute
from ssapy.propagator import KeplerianPropagator


C = 299792458.0


def test_exact_correction_accepts_subpicosecond_oscillation(monkeypatch):
    calls = []

    def oscillating_rv(orbit, times, propagator):
        calls.append(times.copy())
        delay = 1.0 + (2.5e-13 if len(calls) % 2 else 0.0)
        r = np.array([[C * delay, 0.0, 0.0]])
        return r, np.zeros_like(r)

    monkeypatch.setattr(compute, 'rv', oscillating_rv)
    r, v = compute._obsAngleCorrection(
        np.array([[[C, 0.0, 0.0]]]), np.zeros((1, 1, 3)),
        np.zeros((1, 3)), np.zeros((1, 3)), [object()],
        np.array([10.0]), None, 'exact', max_iter=3,
    )
    assert len(calls) == 1
    np.testing.assert_allclose(r[0, 0, 0] / C, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(v, np.zeros((1, 1, 3)))


def test_exact_correction_waits_for_every_orbit(monkeypatch):
    calls = [0, 0]

    def staged_rv(orbit, times, propagator):
        calls[orbit] += 1
        delay = 1.0 + orbit * 1e-4
        if calls[orbit] % 2:
            delay += 2.5e-13
        r = np.array([[C * delay, 0.0, 0.0]])
        return r, np.zeros_like(r)

    monkeypatch.setattr(compute, 'rv', staged_rv)
    r, _ = compute._obsAngleCorrection(
        np.array([[[C, 0.0, 0.0]], [[C, 0.0, 0.0]]]),
        np.zeros((2, 1, 3)), np.zeros((1, 3)), np.zeros((1, 3)),
        [0, 1], np.array([10.0]), None, 'exact', max_iter=3,
    )
    assert calls == [2, 2]
    np.testing.assert_allclose(
        r[:, 0, 0] / C, [1.0, 1.0001], rtol=0, atol=1e-12,
    )


@pytest.mark.parametrize('delay', [1.1, np.nan])
def test_exact_correction_still_rejects_nonconvergence(monkeypatch, delay):
    calls = []

    def oscillating_rv(orbit, times, propagator):
        calls.append(None)
        r = np.array([[C * (delay if len(calls) % 2 else 1.0), 0.0, 0.0]])
        return r, np.zeros_like(r)

    monkeypatch.setattr(compute, 'rv', oscillating_rv)
    with pytest.raises(RuntimeError, match='did not converge'):
        compute._obsAngleCorrection(
            np.array([[[C, 0.0, 0.0]]]), np.zeros((1, 1, 3)),
            np.zeros((1, 3)), np.zeros((1, 3)), [object()],
            np.array([10.0]), None, 'exact', max_iter=3,
        )


@pytest.mark.parametrize('time', [99995496.78556564, 99965412.79906896])
def test_exact_observation_matches_light_time_root(time):
    orbit = ssapy.Orbit(
        np.array([7e6, 1e6, 2e6]), np.array([-1000.0, 7000.0, 1200.0]), 1e8,
    )
    obs_pos = np.array([6.2e6, 1e6, 1e6])
    propagator = KeplerianPropagator()

    def residual(delay):
        r, _ = ssapy.rv(orbit, time - delay, propagator=propagator)
        return delay - np.linalg.norm(r - obs_pos) / C

    delay = brentq(residual, 0.0, 1.0, xtol=1e-15)
    r, v = ssapy.rv(orbit, time - delay, propagator=propagator)
    expected = compute.rvObsToRaDecRate(r, v, obs_pos)[:3]
    kwargs = dict(
        obsPos=obs_pos, obsVel=np.zeros(3), propagator=propagator,
        obsAngleCorrection='exact',
    )
    actual = ssapy.radec(orbit, time, **kwargs)
    np.testing.assert_allclose(actual[:2], expected[:2], rtol=0, atol=1e-10)
    np.testing.assert_allclose(actual[2], expected[2], rtol=0, atol=1e-3)
    expected_direction = (r - obs_pos) / np.linalg.norm(r - obs_pos)
    np.testing.assert_allclose(
        ssapy.dircos(orbit, time, **kwargs), expected_direction,
        rtol=0, atol=1e-10,
    )
