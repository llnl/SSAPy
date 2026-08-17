import numpy as np

from ssapy.orbit import Orbit


def _custom_mu_vector_orbit():
    mu = 4.9048695e12
    radii = np.array([1.8e6, 2.0e6])
    r = np.column_stack((radii, np.zeros(2), np.zeros(2)))
    v = np.column_stack((np.zeros(2), np.sqrt(mu / radii), np.zeros(2)))
    orbit = Orbit(r, v, np.array([0.0, 10.0]), mu=mu)
    return orbit, mu, radii


def test_vector_orbit_index_preserves_custom_mu():
    orbit, mu, radii = _custom_mu_vector_orbit()

    selected = orbit[0]

    assert selected.mu == mu
    np.testing.assert_allclose(selected.a, radii[0], rtol=1e-12)


def test_vector_orbit_iteration_preserves_custom_mu():
    orbit, mu, radii = _custom_mu_vector_orbit()

    selected = list(orbit)

    assert all(item.mu == mu for item in selected)
    np.testing.assert_allclose([item.a for item in selected], radii, rtol=1e-12)
