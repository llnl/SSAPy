# Copyright 2026 Lawrence Livermore National Security, LLC.
# See the top-level LICENSE file for details.
# SPDX-License-Identifier: MIT

"""Valid singular covariance matrices may contain deterministic coordinates."""

import numpy as np
import pytest

from ssapy import utils


COVARIANCES = [
    np.diag([0.0, 1.0, 4.0]),
    np.array([[9.0, 0.0, 3.0], [0.0, 0.0, 0.0], [3.0, 0.0, 4.0]]),
    np.diag([1.0, 4.0, 0.0]),
    np.zeros((3, 3)),
    np.ones((3, 3)),
]


@pytest.mark.parametrize('covariance', COVARIANCES)
def test_samples_preserve_singular_covariance(covariance, monkeypatch):
    x = np.array([10.0, -20.0, 30.0])
    before = covariance.copy()
    # Use deterministic draws with zero mean and identity covariance.
    draws = np.sqrt(3.0) * np.vstack([np.eye(3), -np.eye(3)])

    def normal_draws(npts, dimension):
        assert (npts, dimension) == draws.shape
        return draws.copy()

    monkeypatch.setattr(np.random, 'randn', normal_draws)
    with np.errstate(divide='raise', invalid='raise'):
        samples = utils.sample_points(x, covariance, 6)
    np.testing.assert_allclose(samples.mean(axis=0), x, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        np.cov(samples, rowvar=False, ddof=0), covariance,
        rtol=1e-12, atol=1e-12,
    )
    fixed = np.diag(covariance) == 0
    np.testing.assert_array_equal(samples[:, fixed], np.tile(x[fixed], (6, 1)))
    np.testing.assert_array_equal(covariance, before)


@pytest.mark.parametrize('covariance', COVARIANCES)
def test_sigma_points_preserve_singular_covariance(covariance):
    x = np.array([10.0, -20.0, 30.0])
    before = covariance.copy()
    with np.errstate(divide='raise', invalid='raise'):
        points = utils.sigma_points(None, x, covariance, scale=2)
    assert points.shape == (7, 3)
    np.testing.assert_array_equal(points[0], x)
    np.testing.assert_allclose(
        np.cov(points[1:], rowvar=False, ddof=0), 4 * covariance,
        rtol=1e-12, atol=1e-12,
    )
    fixed = np.diag(covariance) == 0
    np.testing.assert_array_equal(points[:, fixed], np.tile(x[fixed], (7, 1)))
    np.testing.assert_array_equal(covariance, before)


def test_zero_variance_with_explicit_fixed_dimensions():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    covariance = np.diag([4.0, 0.0, 9.0])
    fixed = np.array([False, True, False, False])
    with np.errstate(divide='raise', invalid='raise'):
        points = utils.sigma_points(
            None, x, covariance, fixed_dimensions=fixed,
        )
    np.testing.assert_array_equal(points[:, 1:3], np.tile(x[1:3], (7, 1)))
    np.testing.assert_allclose(
        np.cov(points[1:, ~fixed], rowvar=False, ddof=0), covariance,
        rtol=1e-12, atol=1e-12,
    )


def test_unscented_linear_transform_of_singular_covariance():
    x = np.array([1.0, 2.0, 3.0])
    covariance = COVARIANCES[1]
    transform = np.array([[1.0, 2.0, -1.0], [0.0, 1.0, 2.0]])
    with np.errstate(divide='raise', invalid='raise'):
        mean, cov = utils.unscented_transform_mean_covar(
            lambda points: points @ transform.T, x, covariance,
        )
    np.testing.assert_allclose(mean, transform @ x)
    np.testing.assert_allclose(
        cov, transform @ covariance @ transform.T, rtol=1e-12, atol=1e-12,
    )
