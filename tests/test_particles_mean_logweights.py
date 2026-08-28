from types import SimpleNamespace

import numpy as np

from ssapy.particles import Particles


def test_mean_is_stable_for_large_negative_log_weights():
    particles = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    ])
    log_weights = np.array([-1000.0, -1001.0])
    rvprobability = SimpleNamespace(epoch=0.0)

    sample = Particles(
        particles,
        rvprobability,
        ln_weights=log_weights,
    )

    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    expected = np.average(particles, axis=0, weights=weights)

    result = sample.mean()

    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, expected)
