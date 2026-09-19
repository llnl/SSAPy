import numpy as np
import pytest

from ssapy.linker import Linker
from ssapy.particles import Particles


class _Probability:
    epoch = 0.0

    def __init__(self, likelihoods):
        self.likelihoods = dict(zip([7.0e6, 8.0e6], likelihoods))

    def lnprior(self, orbit):
        return 0.0

    def lnlike(self, orbit):
        return self.likelihoods[orbit.r[0]]


@pytest.mark.filterwarnings(
    "error:Conversion of an array with ndim > 0 to a scalar is deprecated:DeprecationWarning"
)
@pytest.mark.parametrize("track_index", [0, 1])
@pytest.mark.parametrize("likelihoods", [(-1.0, -2.0), (-1000.0, -1001.0)])
def test_selector_uses_single_likelihood_from_real_particles(
    monkeypatch, track_index, likelihoods
):
    probability = _Probability(likelihoods)
    populations = [
        Particles(
            np.array([[radius, 0.0, 0.0, 0.0, 7500.0, 0.0]]),
            probability,
        )
        for radius in [7.0e6, 8.0e6]
    ]
    linker = Linker(populations)
    linker.p_orbit[1] = [0.25, 0.75]
    captured = {}

    def multinomial(n, pvals):
        captured["n"] = n
        captured["pvals"] = pvals.copy()
        selected = np.zeros(len(pvals), dtype=int)
        selected[-1] = 1
        return selected

    monkeypatch.setattr(np.random, "multinomial", multinomial)
    selected = linker.sample_orbit_selectors_from_data_conditional(
        track_index, verbose=False
    )

    count = track_index + 1
    if max(likelihoods[:count]) < -500.0:
        expected = np.full(count, 1.0 / count)
    else:
        expected = np.exp(likelihoods[:count]) * linker.p_orbit[track_index]
        expected /= expected.sum()
    assert captured["n"] == 1
    np.testing.assert_allclose(captured["pvals"], expected)
    np.testing.assert_array_equal(selected, np.eye(count, dtype=int)[-1])
