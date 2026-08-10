import numpy as np
from astropy.time import Time
from astropy import units as u
import math
from functools import partial
import pytest
from astropy.table import QTable

import ssapy
from ssapy.constants import EARTH_MU, RGEO
from ssapy.correlate_tracks import (
    CircVelocityPrior, ZeroRadialVelocityPrior, GaussPrior, VolumeDistancePrior,
    orbit_to_param, make_param_guess, make_optimizer, fit_arc_blind, fit_arc,
    fit_arc_with_gaussian_prior, data_for_satellite, wrap_angle_difference,
    radeczn, param_to_orbit, Track, TrackGauss,TrackBase, MHT, summarize_tracklet,
    summarize_tracklets, iterate_mht, fit_arc_blind_via_track, Hypothesis,
    time_ordered_satIDs, combinatoric_lnprior
)
from ssapy.orbit import Orbit
from ssapy import propagator, rvsampler
from .ssapy_test_helpers import sample_GEO_orbit, sample_LEO_orbit, checkAngle, checkSphere

@pytest.fixture
def sample_data():
    dtype = [
        ('satID', 'int'),
        ('time', 'O'),  # Object type to hold astropy Time instances
        ('rStation_GCRF', 'float', (3,)),
        ('vStation_GCRF', 'float', (3,))
    ]
    data = np.zeros(10, dtype=dtype)
    data['satID'] = [2, 3, 1, 5, 4, 3, 2, 1, 5, 4]
    
    # Generate Time objects based on a reference GPS time
    times = np.linspace(0, 100, 10)
    data['time'] = [Time(t, format='gps') for t in times]
    
    data['rStation_GCRF'] = np.random.rand(10, 3)
    data['vStation_GCRF'] = np.random.rand(10, 3)
    return data

@pytest.fixture
def sample_arc():
    dtype = [('satID', 'int'), ('rStation_GCRF', 'float', (3,)), ('vStation_GCRF', 'float', (3,)),
             ('time', 'float'), ('ra', 'float'), ('dec', 'float'), ('pmra', 'float'), ('pmdec', 'float')]
    arc = np.zeros(10, dtype=dtype)
    arc['satID'] = np.arange(10)
    arc['rStation_GCRF'] = np.random.rand(10, 3)
    arc['vStation_GCRF'] = np.random.rand(10, 3)
    arc['time'] = np.linspace(0, 100, 10)
    arc['ra'] = np.random.rand(10)
    arc['dec'] = np.random.rand(10)
    arc['pmra'] = np.random.rand(10)
    arc['pmdec'] = np.random.rand(10)
    return arc

@pytest.fixture
def sample_guess():
    return np.array([1, 2, 3, 4, 5, 6, 123456789])

@pytest.fixture
def sample_gaussian_prior():
    mu = np.array([1, 2, 3, 4, 5, 6, 123456789])
    cinvcholfac = np.eye(6)
    return mu, cinvcholfac

@pytest.fixture
def sample_propagator():
    return propagator.KeplerianPropagator()

@pytest.fixture
def sample_truth():
    return {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

@pytest.fixture
def sample_hypotheses():
    return [Hypothesis([], nsat=1000)]

@pytest.fixture
def mht_instance(sample_data, sample_truth, sample_hypotheses, sample_propagator):
    return MHT(data=sample_data, nsat=1000, truth=sample_truth,
               hypotheses=sample_hypotheses, propagator=sample_propagator)

@pytest.mark.parametrize("mode, expected_cls", [
    ('rv', rvsampler.LMOptimizer),
    ('equinoctial', rvsampler.EquinoctialLMOptimizer),
])
 
def test_make_optimizer_modes(mode, expected_cls):
    param = list(range(9))
    optimizer = make_optimizer(mode=mode, param=param, lsq=False)
    assert optimizer == expected_cls if not isinstance(optimizer, partial) else optimizer.func == expected_cls

@pytest.mark.parametrize("mode", ['invalid', None])
def test_make_optimizer_invalid_mode(mode):
    with pytest.raises(ValueError):
        make_optimizer(mode=mode, param=[1]*9, lsq=False)


@pytest.mark.parametrize("mode", ['rv', 'equinoctial'])
def test_orbit_to_param_and_back(mode):
    original = sample_GEO_orbit(t=1000)
    params = orbit_to_param(original, mode=mode)
    recovered = param_to_orbit(params, mode=mode)
    np.testing.assert_allclose(recovered.r, original.r, atol=1e-6)
    np.testing.assert_allclose(recovered.v, original.v, atol=1e-6)


def test_param_guess_and_orbitattr_branches():
    orbit = sample_GEO_orbit(t=1000)
    arc = QTable()
    arc['time'] = Time([900.0, 1100.0], format='gps')
    arc['ra'] = [0.1, 0.2] * u.rad
    arc['dec'] = [0.3, 0.4] * u.rad
    arc['rStation_GCRF'] = np.zeros((2, 3)) * u.m
    arc['vStation_GCRF'] = np.zeros((2, 3)) * u.m / u.s

    rvguess = np.hstack([orbit.r, orbit.v])
    rv_guess = make_param_guess(rvguess, arc, mode='rv', orbitattr=['log10area', 'cr'])
    assert rv_guess[-3:-1] == [-1, 1]
    assert np.isclose(rv_guess[-1], 1000.0)

    eq_guess = make_param_guess(rvguess, arc, mode='equinoctial')
    assert len(eq_guess) == 7

    angle_guess = make_param_guess(rvguess, arc, mode='angle')
    assert len(angle_guess) == 13
    with pytest.raises(ValueError, match='unrecognized mode'):
        make_param_guess(rvguess, arc, mode='bad')

    arc['pmra'] = [1e-6, 2e-6] * u.rad / u.s
    pm_angle_guess = make_param_guess(rvguess, arc, mode='angle')
    assert pm_angle_guess[6] == arc['time'][0].gps

    with_attr = ssapy.Orbit(orbit.r, orbit.v, orbit.t, propkw={'area': 100.0, 'cr': 1.5})
    param = orbit_to_param(with_attr, mode='rv', orbitattr=['log10area', 'cr'])
    np.testing.assert_allclose(param[6:8], [2.0, 1.5])
    recovered = param_to_orbit(param, mode='rv', orbitattr=['log10area', 'cr'])
    assert recovered.propkw['area'] == 100.0
    assert recovered.propkw['cr'] == 1.5

    fitonly = orbit_to_param(with_attr, mode='rv', orbitattr=['log10area'], fitonly=True)
    assert fitonly.shape == (7,)

    r_station = np.zeros(3)
    v_station = np.zeros(3)
    angle_param = orbit_to_param(orbit, mode='angle', rStation=r_station, vStation=v_station)
    angle_orbit = param_to_orbit(angle_param, mode='angle')
    np.testing.assert_allclose(angle_orbit.r, orbit.r, rtol=0, atol=1e-6)
    np.testing.assert_allclose(angle_orbit.v, orbit.v, rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match='unknown mode'):
        orbit_to_param(orbit, mode='bad')
    with pytest.raises(ValueError, match='unknown mode'):
        param_to_orbit(param, mode='bad')


@pytest.mark.parametrize("input_angle, wrap_range, center, expected", [
    (3, 2 * math.pi, 0.5, (3 + math.pi) % (2 * math.pi) - math.pi),
    (3, 360, 0.25, (3 + 0.25 * 360) % 360 - 0.25 * 360),
    (1000, 360, 0.5, (1000 + 0.5 * 360) % 360 - 0.5 * 360),
])
 
def test_wrap_angle_difference_values(input_angle, wrap_range, center, expected):
    result = wrap_angle_difference(input_angle, wrap_range, center=center)
    assert pytest.approx(result, rel=1e-6) == expected

 
def test_data_for_satellite_behavior(sample_data):
    result = data_for_satellite(sample_data, [1, 3])
    assert set(result['satID']) <= {1, 3}

 
def test_circ_velocity_prior_properties():
    prior = CircVelocityPrior(sigma=0.2)
    assert isinstance(prior, CircVelocityPrior)
    assert math.isclose(prior.sigma, 0.2)

    orbit = sample_GEO_orbit(t=0)
    chi = prior(orbit, distance=RGEO, chi=True)[0]
    np.testing.assert_allclose(prior(orbit, distance=RGEO)[0], -0.5 * chi**2)

 
def test_zero_radial_velocity_prior_properties():
    prior = ZeroRadialVelocityPrior(sigma=0.3)
    assert isinstance(prior, ZeroRadialVelocityPrior)
    assert math.isclose(prior.sigma, 0.3)

    orbit = sample_GEO_orbit(t=0)
    chi = prior(orbit, distance=RGEO, chi=True)[0]
    np.testing.assert_allclose(prior(orbit, distance=RGEO)[0], -0.5 * chi**2)

 
def test_gauss_prior_properties():
    mu = np.zeros(6)
    cinv = np.eye(6)
    translator = lambda o: np.ones(6)
    prior = GaussPrior(mu, cinv, translator)
    assert np.array_equal(prior.mu, mu)
    assert np.array_equal(prior.cinvcholfac, cinv)
    orbit = sample_LEO_orbit(t=0)
    np.testing.assert_allclose(prior(orbit, chi=True), np.ones(6))
    np.testing.assert_allclose(prior(orbit), -0.5 * np.ones(6))

 
def test_volume_distance_prior_behavior():
    prior = VolumeDistancePrior(scale=RGEO)
    orbit = sample_LEO_orbit(t=0)
    logprob = prior(orbit, 7000e3)
    assert isinstance(logprob, float)
    assert prior(orbit, 7000e3, chi=True) >= 0.0


def test_satellite_selection_and_time_ordering(sample_data):
    selected = data_for_satellite(sample_data, [-1, 1, 3])
    assert set(selected['satID']) <= {1, 3}
    with pytest.raises(ValueError, match="satID 99 not found"):
        data_for_satellite(sample_data, [99])

    timed = QTable()
    timed['satID'] = sample_data['satID']
    timed['time'] = Time([t.gps for t in sample_data['time']], format='gps')
    forward = time_ordered_satIDs(timed, order='forward')
    backward, times = time_ordered_satIDs(timed, with_time=True, order='backward')
    assert forward[0] == timed['satID'][np.argmin(timed['time'].gps)]
    assert backward[0] == timed['satID'][np.argmax(timed['time'].gps)]
    assert len(times) == len(backward)


def test_radeczn_wrap_branches(monkeypatch):
    arc = QTable()
    arc['time'] = Time([10.0, 20.0], format='gps')
    arc['rStation_GCRF'] = np.zeros((2, 3)) * u.m
    arc['vStation_GCRF'] = np.zeros((2, 3)) * u.m / u.s
    arc['satID'] = [1, 2]

    def fake_radec(*args, **kwargs):
        return tuple(np.arange(2.0) + i for i in range(6))

    monkeypatch.setattr(ssapy.compute, 'radec', fake_radec)

    class FakeOrbit:
        def __init__(self, mean_motion, t):
            self.meanMotion = mean_motion
            self.t = t

    out = radeczn([FakeOrbit(0.1, 0.0), FakeOrbit(0.2, 5.0)], arc)
    assert len(out) == 7
    np.testing.assert_allclose(out[-1][0], [1.0, 2.0])

    class VectorOrbit:
        meanMotion = np.array([0.1, 0.2])
        t = np.array([0.0, 5.0])
        r = np.zeros((2, 3))

    out = radeczn(VectorOrbit(), arc)
    assert out[-1].shape == (2, 2)
    np.testing.assert_allclose(out[-1][1], [1.0, 3.0])


def test_tracklet_summaries_and_combinatoric_prior():
    one = QTable()
    one['satID'] = [1]
    one['time'] = Time([0.0], format='gps')
    one['ra'] = [10.0] * u.deg
    one['dec'] = [20.0] * u.deg
    one['sigma'] = [1.0] * u.arcsec
    one['rStation_GCRF'] = np.zeros((1, 3)) * u.m
    one['vStation_GCRF'] = np.full((1, 3), np.nan) * u.m / u.s
    meanpos, dmeanpos, pm, dpm = summarize_tracklet(one)
    assert meanpos[0][0] == one['ra'][0]
    assert dmeanpos[0][0] == one['sigma'][0]
    assert pm == (0.0, 0.0)
    assert dpm == (np.inf, np.inf)

    many = QTable()
    many['satID'] = [2, 2, 1, 1]
    many['time'] = Time([10.0, 20.0, 0.0, 5.0], format='gps')
    many['ra'] = [10.0, 10.001, 30.0, 30.001] * u.deg
    many['dec'] = [20.0, 20.001, 40.0, 40.001] * u.deg
    many['sigma'] = [1.0, 1.0, 2.0, 2.0] * u.arcsec
    many['rStation_GCRF'] = np.array([[0, 0, 0], [10, 0, 0], [5, 0, 0], [7, 0, 0]], dtype=float) * u.m
    many['vStation_GCRF'] = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [1, 0, 0], [1, 0, 0]], dtype=float) * u.m / u.s
    summary = summarize_tracklets(many, posuncfloor=1e-6 * u.deg, pmuncfloor=1e-9 * u.deg / u.s)
    assert len(summary) == 2
    for field in ['dra', 'ddec', 'pmra', 'pmdec', 'dpmra', 'dpmdec', 't_baseline']:
        assert field in summary.colnames
    assert np.all(summary['dra'] > 0 * u.deg)

    assert np.isfinite(combinatoric_lnprior(nsat=10, ntrack=3, ndet=5))


def test_hypothesis_bookkeeping_and_difference(capsys):
    class FakeTrack:
        def __init__(self, name, sat_ids, lnprob):
            self.name = name
            self.satIDs = sat_ids
            self.lnprob = lnprob

        def __repr__(self):
            return f"FakeTrack({self.name})"

    track_a = FakeTrack('a', [1, 2], -1.0)
    track_b = FakeTrack('b', [3], -2.0)
    track_c = FakeTrack('c', [4], -3.0)

    hypothesis = Hypothesis([track_a], nsat=20)
    assert hypothesis.ntracklet() == 2
    assert 'Hypothesis with 1 tracks' in repr(hypothesis)
    assert 'FakeTrack(a)' in hypothesis.summarize(verbose=True)

    appended = Hypothesis.addto(hypothesis, track_b)
    assert appended.tracks == [track_a, track_b]
    replaced = Hypothesis.addto(appended, track_c, oldtrack=track_a)
    assert replaced.tracks == [track_c, track_b]

    replaced.difference(appended)
    output = capsys.readouterr().out
    assert 'Tracks only in 1' in output
    assert 'Tracks only in 2' in output
