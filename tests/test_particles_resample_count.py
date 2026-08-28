import numpy as np
import pytest

import ssapy.particles as particles_module
from ssapy.particles import Particles


class DummyProbability:
    epoch = 0.0

    def lnprior(self, orbit):
        return 0.0


def make_particles(n=5):
    samples = np.arange(n * 6, dtype=float).reshape(n, 6)
    return Particles(
        samples,
        DummyProbability(),
        lnpriors=np.zeros(n),
        ln_weights=np.zeros(n),
    )


def identity_resample(particles, ln_wts, pod=True):
    return particles.copy(), np.full(len(particles), 1.0 / len(particles))


def test_resample_honors_requested_particle_count(monkeypatch):
    population = make_particles(5)
    monkeypatch.setattr(particles_module.utils, 'resample', identity_resample)
    monkeypatch.setattr(
        np.random,
        'choice',
        lambda n, size, replace: np.arange(size),
    )

    population.resample(3)

    assert population.particles.shape == (3, 6)
    assert population.ln_wts.shape == (3,)
    assert population.num_particles == 3


def test_resample_rejects_more_particles_than_available(monkeypatch):
    population = make_particles(5)
    monkeypatch.setattr(particles_module.utils, 'resample', identity_resample)
    particles_before = population.particles.copy()
    weights_before = population.ln_wts.copy()

    with pytest.raises(ValueError, match='Requested more particles than we have'):
        population.resample(6)

    np.testing.assert_array_equal(population.particles, particles_before)
    np.testing.assert_array_equal(population.ln_wts, weights_before)
