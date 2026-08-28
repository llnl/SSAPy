import numpy as np

from ssapy.particles import Particles


class DummyProbability:
    epoch = 0.0

    def lnprior(self, orbit):
        return orbit.r[0] * 1e-6

    def lnlike(self, orbit):
        return -0.5


def make_particles(offset=0.0):
    samples = np.array([
        [7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0],
        [7.1e6, 0.0, 0.0, 0.0, 7.4e3, 0.0],
    ]) + offset
    return Particles(samples, DummyProbability(), ln_weights=np.zeros(2))


def test_reset_to_pseudo_prior_invalidates_derived_caches():
    population = make_particles()
    initial = population.initial_particles.copy()

    population.particles = population.particles + 1.0e5
    population._orbits = None
    population._lnpriors = None
    assert population.orbits[0].r[0] != initial[0, 0]
    _ = population.lnpriors

    population.resample(1)
    population.reset_to_pseudo_prior()

    np.testing.assert_allclose(population.orbits[0].r, initial[0, :3])
    assert population.num_particles == len(initial)
    assert len(population.lnpriors) == len(initial)
    assert population.lnpriors[0] == population.rvprobability.lnprior(population.orbits[0])


def test_reweight_invalidates_caches_after_appending_particles():
    population = make_particles()
    other = make_particles(offset=1000.0)

    # Populate both derived caches before reweight changes the particle array.
    assert len(population.orbits) == 2
    assert len(population.lnpriors) == 2

    # Avoid testing propagation here; this regression is about cache lifetime.
    other.move = lambda epoch: other.particles.copy()
    population.reweight(other)

    assert population.particles.shape[0] == 4
    assert len(population.orbits) == 4
    assert len(population.lnpriors) == 4
    np.testing.assert_allclose(population.orbits[-1].r, other.particles[-1, :3])
