import numpy as np

from ssapy.linker import Linker


class FakeIOD:
    def __init__(self, model_id, likelihoods):
        self.model_id = model_id
        self._likelihoods = likelihoods

    def draw_orbit(self):
        return self.model_id

    def lnlike(self, model_id):
        return self._likelihoods[model_id]


def test_selector_probabilities_remain_finite_with_tiny_priors(monkeypatch):
    likelihoods = np.array([-400.0, -401.0, -2000.0])
    linker = Linker([
        FakeIOD(0, likelihoods),
        FakeIOD(1, likelihoods),
        FakeIOD(2, likelihoods),
    ])
    linker.p_orbit[2] = np.array([1e-320, 2e-320, 1.0])

    captured = {}

    def fake_multinomial(n, pvals):
        captured['pvals'] = pvals.copy()
        return np.array([1, 0, 0])

    monkeypatch.setattr(np.random, 'multinomial', fake_multinomial)

    linker.sample_orbit_selectors_from_data_conditional(2, verbose=False)
    pvals = captured['pvals']

    assert np.all(np.isfinite(pvals))
    assert np.isclose(np.sum(pvals), 1.0)
    assert pvals[0] > 0.0
    assert pvals[1] > 0.0
    assert pvals[2] < pvals[0]
