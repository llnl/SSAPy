import numpy as np
import pytest

import ssapy


@pytest.fixture(params=["keplerian", "rk4"])
def propagator(request):
    if request.param == "keplerian":
        return ssapy.KeplerianPropagator()
    return ssapy.RK4Propagator(ssapy.AccelKepler(), h=25.0)


@pytest.fixture
def orbits():
    return ssapy.Orbit(
        [[7.0e6, 0.0, 0.0], [8.0e6, 0.0, 0.0]],
        [[0.0, 7500.0, 0.0], [0.0, 7000.0, 500.0]],
        0.0,
    )


def assert_same_state(actual, expected):
    assert actual.r.shape == expected.r.shape
    assert actual.v.shape == expected.v.shape
    np.testing.assert_allclose(actual.r, expected.r, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(actual.v, expected.v, rtol=0.0, atol=1e-11)
    np.testing.assert_array_equal(actual.t, expected.t)


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(0, id="first"),
        pytest.param(1, id="second"),
        pytest.param(-1, id="negative"),
        pytest.param(slice(1, None), id="slice"),
        pytest.param(slice(None, None, -1), id="reverse-slice"),
        pytest.param([1, 0], id="reordered"),
        pytest.param([False, True], id="mask"),
    ],
)
def test_indexed_orbit_keeps_selected_propagation_root(orbits, propagator, index):
    advanced = orbits.at(100.0, propagator=propagator)

    actual = advanced[index].at(300.0, propagator=propagator)
    expected = orbits[index].at(300.0, propagator=propagator)

    assert_same_state(actual, expected)


def test_iterated_orbits_keep_individual_propagation_roots(orbits, propagator):
    advanced = orbits.at(100.0, propagator=propagator)

    for index, orbit in enumerate(advanced):
        actual = orbit.at(300.0, propagator=propagator)
        expected = orbits[index].at(300.0, propagator=propagator)
        assert_same_state(actual, expected)


def test_time_samples_keep_scalar_propagation_root(orbits, propagator):
    original = orbits[1]
    samples = original.at(np.array([100.0, 200.0]), propagator=propagator)
    expected = original.at(300.0, propagator=propagator)

    for index in range(len(samples)):
        actual = samples[index].at(300.0, propagator=propagator)
        assert_same_state(actual, expected)
    for sample in samples:
        actual = sample.at(300.0, propagator=propagator)
        assert_same_state(actual, expected)
