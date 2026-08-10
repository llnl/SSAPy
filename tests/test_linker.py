import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import Longitude, Latitude
from astropy.table import QTable

import ssapy
from ssapy.constants import RGEO, VGEO


class _FakeIOD:
    def __init__(self, name, lnlike_by_theta=None):
        self.name = name
        self.lnlike_by_theta = lnlike_by_theta or {}
        self.fused = []
        self.reset = False
        self.particles = np.array([[float(name), float(name) + 1.0, float(name) + 2.0]])
        self.ln_wts = np.array([0.0])

    @property
    def lnlike(self):
        return lambda theta: self.lnlike_by_theta.get(theta, -1.0)

    def draw_orbit(self):
        return f"theta-{self.name}"

    def fuse(self, other):
        self.fused.append(other.name)

    def reset_to_pseudo_prior(self):
        self.reset = True

def _create_iods_small(num_epochs=2, num_obs_per_track=3, nBurn=10, nStep=10):
    from ssapy.rvsampler import RPrior, APrior
    np.random.seed(42)

    sigma_arcsec = 1.0
    sigma_rad = sigma_arcsec * np.pi / (180. * 3600.)
    time0 = Time(2458316., format='jd')

    lon, lat, elevation = 100.0, 33.0, 1300.0
    observer = ssapy.EarthObserver(lon, lat, elevation)
    orbit = ssapy.Orbit.fromKeplerianElements(RGEO * 0.98, 0.01, 0.001, 0.0, 1.2, 1.03, time0)
    times = time0 + np.linspace(0, 4, num_obs_per_track) * u.h
    coord = ssapy.radec(orbit, times, observer=observer)
    rstation, vstation = observer.getRV(times)

    iods = []
    for _ in range(num_epochs):
        arc = QTable()
        arc['ra'] = Longitude((coord[0] + np.random.randn(len(coord[0])) * sigma_rad) * u.rad)
        arc['dec'] = Latitude((coord[1] + np.random.randn(len(coord[1])) * sigma_rad) * u.rad)
        arc['rStation_GCRF'] = rstation * u.m
        arc['vStation_GCRF'] = vstation * u.m / u.s
        arc['time'] = Time(times)
        arc['sigma'] = sigma_arcsec * np.ones(num_obs_per_track) * u.arcsec

        r, v = ssapy.rv(orbit, time0)
        initializer = ssapy.GaussianRVInitializer(r, v, rSigma=0.1 * RGEO, vSigma=0.1 * VGEO)
        priors = [RPrior(RGEO, RGEO * 0.2), APrior(RGEO, RGEO * 0.2)]
        rvprob = ssapy.RVProbability(arc, time0, priors=priors)

        sampler = ssapy.EmceeSampler(rvprob, initializer, nWalker=10)
        chain, lnprob, lnprior = sampler.sample(nBurn=nBurn, nStep=nStep)
        chain, lnprob, lnprior = ssapy.utils.cluster_emcee_walkers(chain, lnprob, lnprior, thresh_multiplier=4)

        iods.append(ssapy.Particles(chain, rvprob, lnpriors=lnprior))

    return iods


def test_model_selector_params_normalize():
    p = ssapy.ModelSelectorParams(3, 3, init_val=1.0)
    p.normalize()
    for i in range(3):
        assert np.isclose(np.sum(p[i]), 1.0)


def test_selector_params_repr_indexing_and_binary_helpers():
    params = ssapy.ModelSelectorParams(4, 4, init_val=2.0)
    np.testing.assert_allclose(params.params, np.tril(np.ones((4, 4)) * 2.0))
    np.testing.assert_allclose(params[0], [2.0])
    params[2] = [0.1, 0.2, 0.7]
    np.testing.assert_allclose(params[2], [0.1, 0.2, 0.7])
    assert "2.000" in repr(params)

    selectors = ssapy.BinarySelectorParams(3, 3, init_value=0, dtype=int)
    selectors.params[:] = [[1, 0, 0], [1, 0, 0], [0, 0, 0]]

    assert repr(selectors).splitlines()[0] == "1\t0\t0"
    np.testing.assert_array_equal(selectors.get_linked_track_indices(0), [0, 1])
    np.testing.assert_array_equal(selectors.get_unlinked_track_indices(), [1, 2])


def test_linker_lightweight_state_transitions(tmp_path):
    iods = [_FakeIOD(0), _FakeIOD(1), _FakeIOD(2)]
    linker = ssapy.Linker(iods, num_orbits=3)
    default_linker = ssapy.Linker(iods[:2])

    assert "Linker(num_tracks=3, num_orbits=3)" in repr(linker)
    np.testing.assert_array_equal(linker.orbit_selectors.params[:, 0], [1, 1, 1])
    assert default_linker.num_orbits == 2

    np.random.seed(0)
    p_orbit = linker.sample_Porbit_conditional_dist(2)
    assert p_orbit.shape == (3,)
    assert np.isclose(np.sum(p_orbit), 1.0)

    linker.iods[1].lnlike_by_theta = {"theta-0": -1000.0, "theta-1": -1001.0}
    np.random.seed(0)
    selector = linker.sample_orbit_selectors_from_data_conditional(1, verbose=False)
    assert selector.shape == (2,)
    assert np.sum(selector) == 1

    np.random.seed(0)
    selector = linker.sample_orbit_selectors_from_data_conditional(1, verbose=True)
    assert selector.shape == (2,)
    assert np.sum(selector) == 1

    linker.iods[1].lnlike_by_theta = {"theta-0": 0.0, "theta-1": -5.0}
    np.random.seed(0)
    selector = linker.sample_orbit_selectors_from_data_conditional(1, verbose=False)
    assert selector.shape == (2,)
    assert np.sum(selector) == 1

    linker.orbit_selectors.params[:] = [[1, 0, 0], [1, 0, 0], [0, 0, 0]]
    assert linker.update_orbit_parameters() is None
    assert iods[0].fused == [1]
    assert iods[1].reset is True
    assert iods[2].reset is True

    out = tmp_path / "link_step"
    linker.save_step(str(out))
    assert (tmp_path / "link_step_orbit_selectors.txt").exists()
    assert (tmp_path / "link_step_p_orbit.txt").exists()
    assert (tmp_path / "link_step_particles_0.txt").exists()

    calls = []

    def fake_selector(track_ndx, verbose=True):
        selector = np.zeros(track_ndx + 1, dtype=int)
        selector[0] = 1
        return selector

    def fake_p_orbit(track_ndx):
        return np.ones(track_ndx + 1) / (track_ndx + 1)

    linker.sample_orbit_selectors_from_data_conditional = fake_selector
    linker.sample_Porbit_conditional_dist = fake_p_orbit
    linker.update_orbit_parameters = lambda: calls.append("update")
    assert linker.update_params_using_carlin_chib(verbose=True) is None
    assert calls == ["update"]

    calls = []
    linker.update_params_using_carlin_chib = lambda: calls.append("step")
    assert linker.sample(nStep=3) is None
    assert calls == ["step", "step", "step"]
