import numpy as np

from ssapy.linker import Linker


def test_linker_honors_a_orbit_hyperparameter():
    linker = Linker([object(), object(), object()], a_orbit=0.25)

    np.testing.assert_allclose(linker._a_orbit[0], [0.25])
    np.testing.assert_allclose(linker._a_orbit[1], [0.25, 0.25])
    np.testing.assert_allclose(linker._a_orbit[2], [0.25, 0.25, 0.25])


def test_linker_a_orbit_changes_dirichlet_conditional(monkeypatch):
    linker = Linker([object(), object(), object()], a_orbit=0.4)
    captured = {}

    def fake_dirichlet(alpha):
        captured['alpha'] = np.asarray(alpha).copy()
        return np.asarray(alpha) / np.sum(alpha)

    monkeypatch.setattr(np.random, 'dirichlet', fake_dirichlet)
    linker.sample_Porbit_conditional_dist(2)

    # Track 2 initially selects orbit 0, so its selector contributes [1, 0, 0]
    # on top of the configured symmetric Dirichlet prior.
    np.testing.assert_allclose(captured['alpha'], [1.4, 0.4, 0.4])
