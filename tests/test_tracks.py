import numpy as np
from astropy.time import Time
from astropy import units as u
import math
from functools import partial
from types import SimpleNamespace
import pytest
from astropy.table import QTable

import ssapy
import ssapy.correlate_tracks as ct
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


def test_make_optimizer_angle_and_lsq_modes():
    param = list(range(13))
    angle_optimizer = make_optimizer(mode='angle', param=param, lsq=False)
    assert angle_optimizer.func is rvsampler.LMOptimizerAngular
    np.testing.assert_allclose(angle_optimizer.keywords['initObsPos'], param[-6:-3])
    np.testing.assert_allclose(angle_optimizer.keywords['initObsVel'], param[-3:])

    lsq_optimizer = make_optimizer(mode='rv', param=param, lsq=True)
    assert lsq_optimizer.func is rvsampler.LeastSquaresOptimizer
    assert lsq_optimizer.keywords['translatorcls'] is rvsampler.ParamOrbitRV

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
    np.testing.assert_allclose(angle_orbit.r, orbit.r, rtol=0, atol=2e-6)
    np.testing.assert_allclose(angle_orbit.v, orbit.v, rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match='unknown mode'):
        orbit_to_param(orbit, mode='bad')
    with pytest.raises(ValueError, match='unknown mode'):
        param_to_orbit(param, mode='bad')


def test_param_orbit_vector_and_velocity_clipping_branches():
    params = np.array([
        [7000e3, 0.0, 0.0, 0.0, 1e6, 0.0, 100.0],
        [8000e3, 0.0, 0.0, 0.0, 7500.0, 0.0, 110.0],
    ])
    orbit = param_to_orbit(params, mode='rv')
    vmax = np.sqrt(2 * EARTH_MU / ssapy.utils.norm(orbit.r))
    assert orbit.r.shape == (2, 3)
    assert np.all(ssapy.utils.norm(orbit.v) <= vmax)

    hyperbolic = ssapy.Orbit(
        params[:, :3],
        np.array([[0.0, 1e6, 0.0], [0.0, 7500.0, 0.0]]),
        Time(params[:, 6], format='gps'),
    )
    clipped = orbit_to_param(hyperbolic, mode='rv')
    clipped_vmax = np.sqrt(2 * EARTH_MU / ssapy.utils.norm(clipped[:, :3]))
    assert clipped.shape == (2, 7)
    assert np.all(ssapy.utils.norm(clipped[:, 3:6]) <= clipped_vmax)


def test_fit_arc_entry_points_with_mock_optimizer(monkeypatch, capsys):
    arc = _minimal_track_table([1, 2, 3], [0.0, 10.0, 25.0], measurements=True)
    calls = []

    class FakeOptimizer:
        def __init__(self, prob, init, orbitattr=None, **kwargs):
            self.prob = prob
            self.init = np.asarray(init, dtype=float)
            self.orbitattr = orbitattr
            self.kwargs = kwargs
            self.result = SimpleNamespace(
                residual=np.array([1.0, 2.0]),
                hithyperbolicorbit=True,
            )
            calls.append((self.init.copy(), orbitattr, kwargs))

        def optimize(self, **kwargs):
            self.optimize_kwargs = kwargs
            return self.init + len(calls)

    monkeypatch.setattr(
        ct.rvsampler,
        'RVProbability',
        lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
    )
    monkeypatch.setattr(
        ct.rvsampler,
        'circular_guess',
        lambda arc_arg: (np.arange(1.0, 7.0), Time(0.0, format='gps')),
    )
    monkeypatch.setattr(ct, 'make_optimizer', lambda *args, **kwargs: FakeOptimizer)

    chi2, param, result = fit_arc_blind(arc, factor=2, max_nfev=4)
    assert chi2 == pytest.approx(1e9 + 5.0)
    assert param.shape == (7,)
    assert result.hithyperbolicorbit is True
    assert 'hyperbolic orbit in blind' in capsys.readouterr().out

    with pytest.raises(ValueError, match='len\\(arc\\) must be > 0'):
        fit_arc_blind(arc[:0])

    monkeypatch.setattr(
        ct.rvsampler,
        'circular_guess',
        lambda arc_arg: (np.arange(1.0, 7.0), Time(99.0, format='gps')),
    )
    with pytest.raises(ValueError, match='inconsistent epoch'):
        fit_arc_blind(arc)
    monkeypatch.setattr(
        ct.rvsampler,
        'circular_guess',
        lambda arc_arg: (np.arange(1.0, 7.0), Time(0.0, format='gps')),
    )

    guess = np.arange(1.0, 8.0)
    guess[6] = 0.0
    chi2, param, result = fit_arc(arc, guess, max_nfev=5)
    assert chi2 == pytest.approx(1e9 + 5.0)
    assert param.shape == (7,)
    assert result.hithyperbolicorbit is True

    angle_mu = np.arange(13.0)
    angle_mu[6] = 0.0
    chi2, param, result = fit_arc_with_gaussian_prior(
        arc, angle_mu, np.eye(6), mode='angle', max_nfev=6)
    assert chi2 == pytest.approx(1e9 + 5.0)
    assert param.shape == (13,)
    assert result.hithyperbolicorbit is True


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

    scalar_arc = arc[0]

    class ScalarOrbit:
        meanMotion = 0.5
        t = 8.0
        r = np.zeros(3)

    out = radeczn(ScalarOrbit(), scalar_arc)
    assert out[-1] == pytest.approx(1.0)


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


def _minimal_track_table(satids=(1,), gps=(0.0,), measurements=False):
    table = QTable()
    table['satID'] = np.asarray(satids)
    table['time'] = Time(np.asarray(gps, dtype=float), format='gps')
    table['rStation_GCRF'] = np.zeros((len(satids), 3)) * u.m
    table['vStation_GCRF'] = np.zeros((len(satids), 3)) * u.m / u.s
    if measurements:
        table['ra'] = np.full(len(satids), 0.1) * u.rad
        table['dec'] = np.full(len(satids), 0.2) * u.rad
        table['pmra'] = np.full(len(satids), 1e-4) * u.rad / u.s
        table['pmdec'] = np.full(len(satids), -2e-4) * u.rad / u.s
        table['dra'] = np.full(len(satids), 1e-5) * u.rad
        table['ddec'] = np.full(len(satids), 1e-5) * u.rad
        table['dpmra'] = np.full(len(satids), 1e-7) * u.rad / u.s
        table['dpmdec'] = np.full(len(satids), 1e-7) * u.rad / u.s
    return table


def _positive_sigma_cloud(center, delta=0.1):
    sigma = [np.asarray(center, dtype=float)]
    for i in range(6):
        high = sigma[0].copy()
        low = sigma[0].copy()
        high[i] += delta
        low[i] -= delta
        sigma.extend([high, low])
    return np.asarray(sigma)


def test_trackbase_predict_keeps_fixed_epoch_dimension():
    class LinearTrack(TrackBase):
        def propagaterdz(self, param, arc0=None, return_nwrap=False):
            self.seen_param = np.asarray(param)
            rows = [param[:, 0], param[:, 1], param[:, 2], param[:, 3]]
            if return_nwrap:
                rows.append(param[:, 4])
            return np.asarray(rows)

    data = _minimal_track_table()
    track = LinearTrack([1], data, volume=10.0)
    track.param = np.array([0.1, 0.2, 1e-4, -2e-4, 3.0, 4.0, 100.0])
    track.covar = np.eye(6) * 1e-4

    mean, covar, sigma = track.predict(
        data[0], return_sigma=True, return_nwrap=False)

    np.testing.assert_allclose(mean, track.param[:4])
    assert covar.shape == (4, 4)
    assert sigma.shape == (4, 13)
    np.testing.assert_allclose(track.seen_param[:, 6], 100.0)


def test_trackbase_predict_multiple_times_and_base_propagation(monkeypatch):
    class MultiTimeTrack(TrackBase):
        def propagaterdz(self, param, arc0=None, return_nwrap=False):
            t = np.atleast_1d(arc0['time'].gps)
            rows = [
                param[:, 0, None] + 0 * t,
                param[:, 1, None] + 0 * t,
                param[:, 2, None] + t,
                param[:, 3, None] - t,
            ]
            if return_nwrap:
                rows.append(param[:, 4, None] + 0 * t)
            return np.asarray(rows)

    data = _minimal_track_table([1, 2], [10.0, 20.0])
    track = MultiTimeTrack([1], data, volume=10.0)
    track.param = np.array([0.1, 0.2, 1e-4, -2e-4, 3.0, 4.0, 100.0])
    track.covar = np.eye(6) * 1e-4

    mean, covar = track.predict(data, return_sigma=False, return_nwrap=False)
    assert mean.shape == (4, 2)
    assert covar.shape == (2, 4, 4)

    attr_track = MultiTimeTrack([1], data, volume=10.0, orbitattr=['cr'])
    attr_track.param = np.array([0.1, 0.2, 1e-4, -2e-4, 3.0, 4.0, 1.2, 100.0])
    attr_track.covar = np.eye(7) * 1e-4
    mean, covar = attr_track.predict(data, return_sigma=False, return_nwrap=False)
    assert mean.shape == (4, 2)

    base = TrackBase([1], data, volume=10.0)
    base.param = track.param.copy()
    base.covar = np.eye(6)
    base.mode = 'rv'
    base.orbitattr = None
    fake_orbit = object()
    monkeypatch.setattr(ct, 'param_to_orbit', lambda param, **kwargs: fake_orbit)
    monkeypatch.setattr(
        ct,
        'radeczn',
        lambda orbit, arc0, **kwargs:
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
    )
    np.testing.assert_allclose(
        base.propagaterdz(base.param, arc0=data[0], return_nwrap=True),
        [1.0, 2.0, 4.0, 5.0, 7.0],
    )
    assert base.update(Time(200.0, format='gps')) is None


def test_trackbase_lnprob_gate_and_repr():
    class GateTrack(TrackBase):
        def predict(self, arc0, return_sigma=False, return_nwrap=True):
            assert return_nwrap is True
            mean = np.array([0.1, 0.2, 1e-4, -2e-4, 0.0])
            covar = np.diag([1e-8, 1e-8, 1e-12, 1e-12, 0.25])
            return mean, covar

    data = _minimal_track_table(measurements=True)
    track = GateTrack([1], data, volume=10.0)
    track.chi2 = 2.0
    track.covar = np.eye(6) * 0.25

    determinant = np.prod(np.linalg.svd(2 * np.pi * track.covar)[1])
    expected = -np.log(track.volume) + 0.5 * np.log(determinant) - 1.0
    assert track.lnprob == pytest.approx(expected)
    assert 'Track, chi2:' in repr(track)

    chi2, nwrapsig = track.gate(data[0:1], return_nwrap=True)
    assert chi2 == pytest.approx(0.0, abs=1e-20)
    assert nwrapsig == pytest.approx(0.5)

    broad = GateTrack([1], data, volume=10.0)
    broad.chi2 = 0.0
    broad.covar = np.full((6, 6), np.inf)
    assert broad.lnprob == pytest.approx(-np.log(10.0))

    singular = GateTrack([1], data, volume=10.0)
    singular.chi2 = 2.0
    singular.covar = np.zeros((6, 6))
    assert np.isfinite(singular.lnprob)

    huge = GateTrack([1], data, volume=10.0)
    huge.chi2 = 0.0
    huge.covar = np.eye(6) * 1e20
    assert huge.lnprob == pytest.approx(-np.log(10.0))

    wide_gate = GateTrack([1], data, volume=10.0)
    wide_gate.predict = lambda arc0, return_sigma=False, return_nwrap=True: (
        np.array([0.1, 0.2, 1e-4, -2e-4, 0.0]),
        np.diag([1.0, 1.0, 1e-12, 1e-12, 0.25]),
    )
    assert wide_gate.gate(data[0:1]) == pytest.approx(0.0)

    singular_gate = GateTrack([1], data, volume=10.0)
    data_zero_pm = data.copy()
    data_zero_pm['dpmra'] = np.zeros(len(data)) * u.rad / u.s
    data_zero_pm['dpmdec'] = np.zeros(len(data)) * u.rad / u.s
    singular_gate.predict = lambda arc0, return_sigma=False, return_nwrap=True: (
        np.array([0.1, 0.2, 1e-4, -2e-4, 0.0]),
        np.diag([1e-8, 1e-8, 0.0, 0.0, 0.25]),
    )
    assert singular_gate.gate(data_zero_pm[0:1]) == pytest.approx(0.0)


def test_track_uses_fitters_and_gaussian_approximation(monkeypatch):
    data = _minimal_track_table([1, 2, 3, 4], [0.0, 10.0, 20.0, 30.0])
    calls = []
    param = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0])

    def fake_fit_arc_blind(arc, **kwargs):
        calls.append(('blind', tuple(arc['satID']), kwargs['mode']))
        return 4.0, param.copy(), SimpleNamespace(covar=np.eye(6), success=True)

    def fake_fit_arc(arc, guess, **kwargs):
        calls.append(('fit', tuple(arc['satID']), tuple(guess), kwargs['mode']))
        return 5.0, param + 1.0, SimpleNamespace(covar=np.eye(6) * 2, success=True)

    monkeypatch.setattr(ct, 'fit_arc_blind', fake_fit_arc_blind)
    monkeypatch.setattr(ct, 'fit_arc', fake_fit_arc)

    track = Track([1], data)
    assert track.success is True
    assert track.gaussian_approximation() is track

    added = track.addto(2)
    assert added.satIDs == [1, 2]
    assert calls[-1][0] == 'fit'

    long_track = Track([1, 2, 3, 4], data)
    gaussian = long_track.gaussian_approximation()
    assert isinstance(gaussian, TrackGauss)

    replacement_propagator = object()
    refit_gaussian = long_track.gaussian_approximation(
        propagator=replacement_propagator)
    assert isinstance(refit_gaussian, TrackGauss)
    assert refit_gaussian.propagator is replacement_propagator
    assert any(call[0] == 'blind' for call in calls)


def test_track_missing_fit_covar_uses_parameter_sized_fallback(monkeypatch):
    data = _minimal_track_table([1, 2, 3, 4], [0.0, 10.0, 20.0, 30.0])
    param = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0])

    monkeypatch.setattr(
        ct, 'fit_arc_blind',
        lambda arc, **kwargs: (4.0, param.copy(), SimpleNamespace(success=True)))

    track = Track([1, 2, 3, 4], data)
    assert track.covar.shape == (6, 6)
    assert np.all(~np.isfinite(track.covar))

    gaussian = track.gaussian_approximation()
    assert isinstance(gaussian, TrackGauss)
    assert gaussian.covar.shape == (6, 6)
    assert gaussian.chi2 >= 1e9


def test_trackgauss_update_at_and_missing_addto_covar(monkeypatch):
    data = _minimal_track_table([1, 2], [0.0, 20.0])
    start = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0])
    track = TrackGauss([1], data, start.copy(), np.eye(6) * 0.01, 7.0)

    track.update(Time(0.0, format='gps'))
    np.testing.assert_allclose(track.param, start)

    class OrbitCloud:
        def __init__(self, sigma):
            self.sigma = sigma

        def at(self, t, propagator=None):
            return SimpleNamespace(t=t, propagator=propagator)

    def fake_param_to_orbit(sigma, mode='rv', orbitattr=None):
        return OrbitCloud(sigma)

    def fake_orbit_to_param(orbit, mode='rv', rStation=None, vStation=None,
                            orbitattr=None):
        assert not isinstance(rStation, u.Quantity)
        center = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, orbit.t.gps])
        return _positive_sigma_cloud(center)

    monkeypatch.setattr(ct, 'param_to_orbit', fake_param_to_orbit)
    monkeypatch.setattr(ct, 'orbit_to_param', fake_orbit_to_param)

    track.update(Time(10.0, format='gps'),
                 rStation=np.ones(3) * u.m,
                 vStation=np.ones(3) * u.m / u.s)
    assert track.param[6] == pytest.approx(10.0)
    assert track.covar.shape == (6, 6)

    shifted = track.at(Time(20.0, format='gps'),
                       rStation=np.ones(3) * u.m,
                       vStation=np.ones(3) * u.m / u.s)
    assert isinstance(shifted, TrackGauss)
    assert shifted.param[6] == pytest.approx(20.0)
    assert track.param[6] == pytest.approx(10.0)

    def fake_fit_arc_with_gaussian_prior(arc, param, cinvcholfac, **kwargs):
        assert tuple(arc['satID']) in [(1,), (2,)]
        return 3.0, np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 20.0]), SimpleNamespace()

    monkeypatch.setattr(ct, 'fit_arc_with_gaussian_prior',
                        fake_fit_arc_with_gaussian_prior)
    added = track.addto(2)
    assert added.satIDs == [1, 2]
    assert added.covar.shape == (6, 6)
    assert np.all(~np.isfinite(added.covar))
    assert added.chi2 >= 1e9

    same_epoch = TrackGauss([1], data, start.copy(), np.eye(6) * 0.01, 7.0)
    assert same_epoch.gaussian_approximation(same_epoch.propagator) is same_epoch
    with pytest.raises(NotImplementedError):
        same_epoch.gaussian_approximation(object())
    same_added = same_epoch.addto(1)
    assert same_added.satIDs == [1, 1]


class _FakeMHTTrack:
    def __init__(self, satids, lnprob=-1.0, chi2=0.0,
                 gate_result=(1.0, 0.1), added_chi2_delta=1.0):
        self.satIDs = list(satids)
        self.lnprob = lnprob
        self.chi2 = chi2
        self.times = np.asarray(self.satIDs, dtype=float) - 1.0
        self.gate_result = gate_result
        self.added_chi2_delta = added_chi2_delta
        self.updated = False

    def __repr__(self):
        return f"FakeMHTTrack({self.satIDs})"

    def update(self, *args, **kwargs):
        self.updated = True

    def gate(self, arc, return_nwrap=False):
        return self.gate_result

    def addto(self, satid):
        self.added_track = _FakeMHTTrack(
            self.satIDs + [satid], lnprob=self.lnprob + 0.5,
            chi2=self.chi2 + self.added_chi2_delta)
        return self.added_track

    def gaussian_approximation(self, propagator=None):
        self.approximated_with = propagator
        return self


def test_fit_arc_blind_via_track_reset_and_approximate(monkeypatch, capsys):
    data = _minimal_track_table([1, 2, 3], [0.0, 10.0, 20.0])
    constructed = []

    class FakeSequentialTrack(_FakeMHTTrack):
        def __init__(self, satids, data, **kwargs):
            super().__init__(list(satids), lnprob=-1.0, chi2=0.0)
            self.data = data
            self.kwargs = kwargs
            constructed.append(self)

        def gate(self, arc, return_nwrap=False):
            if self.satIDs == [1]:
                return 1.0, np.pi
            return 2.0, 0.01

        def addto(self, satid):
            new_ids = self.satIDs + list(satid)
            return FakeSequentialTrack(new_ids, self.data, **self.kwargs)

    monkeypatch.setattr(ct, 'Track', FakeSequentialTrack)

    with pytest.raises(AssertionError, match='Geometric factor'):
        fit_arc_blind_via_track(data, factor=0.5)

    tracks = fit_arc_blind_via_track(
        data, reset_if_too_uncertain=True, approximate=True,
        verbose=True, factor=1)

    assert [track.satIDs for track in tracks] == [[1], [2], [2, 3]]
    assert tracks[-1].approximated_with is None
    assert constructed[0].kwargs['propagator'] is None
    assert 'resetting' in capsys.readouterr().out


def test_mht_run_orders_tracklets_and_prunes(capsys):
    mht = object.__new__(MHT)
    mht.satids = [1, 2, 3]
    live = _FakeMHTTrack([1])
    dead = _FakeMHTTrack([2])
    dead.dead = True
    mht.track2hyp = {live: [], dead: []}
    mht.nfit = 4
    mht.hypotheses = [Hypothesis([], nsat=10)]
    calls = []
    mht.add_tracklet = lambda satid: calls.append(('add', satid))
    mht.prune = lambda satid, **kwargs: calls.append(('prune', satid, kwargs))

    MHT.run(mht, first=0, last=2, verbose=True, order='backward', pkeep=0.5)

    assert calls == [
        ('add', 2), ('prune', 2, {'pkeep': 0.5}),
        ('add', 1), ('prune', 1, {'pkeep': 0.5}),
    ]
    assert 'Tracklet' in capsys.readouterr().out

    mht = object.__new__(MHT)
    mht.satids = [1, 2]
    mht.track2hyp = {}
    mht.nfit = 0
    mht.hypotheses = [Hypothesis([], nsat=10)]
    calls = []
    mht.add_tracklet = lambda satid: calls.append(('add', satid))
    mht.prune = lambda satid, **kwargs: calls.append(('prune', satid, kwargs))
    MHT.run(mht)
    assert calls == [('add', 1), ('prune', 1, {}), ('add', 2), ('prune', 2, {})]


def test_mht_add_tracklet_skip_gate_and_refit_branches(monkeypatch, capsys):
    data = _minimal_track_table([9], [10.0])
    same_time = _FakeMHTTrack([1], lnprob=-1.0)
    same_time.times = np.array([data['time'].gps[0]])
    already_dead = _FakeMHTTrack([2], lnprob=-2.0)
    already_dead.dead = True
    out_of_gate = _FakeMHTTrack([3], lnprob=-3.0, gate_result=(100.0, 0.01))
    poor_refit = _FakeMHTTrack([4], lnprob=-4.0, gate_result=(0.0, 0.01),
                               added_chi2_delta=100.0)
    hypothesis = Hypothesis([same_time, already_dead, out_of_gate, poor_refit],
                            nsat=10)
    truth = {1: 'X', 2: 'Y', 3: 'A', 4: 'B', 9: 'A'}
    mht = MHT(data, nsat=10, truth=truth, hypotheses=[hypothesis])

    singleton = _FakeMHTTrack([9], lnprob=-0.25)
    monkeypatch.setattr(ct, 'Track', lambda satids, data, **kwargs: singleton)

    mht.add_tracklet(9)

    assert same_time.updated is False
    assert already_dead.updated is False
    assert out_of_gate.updated is True
    assert poor_refit.updated is True
    assert poor_refit.added_track not in mht.track2hyp
    assert singleton in mht.track2hyp
    assert mht.nfit == 1
    output = capsys.readouterr().out
    assert 'warning, excluding real track by gate' in output


def test_mht_initial_hypothesis_and_add_tracklet_debug_paths(monkeypatch, capsys):
    data = _minimal_track_table([1, 2], [0.0, 10.0])
    mht = MHT(data, nsat=10)
    assert len(mht.hypotheses) == 1

    mht.add_tracklet(-1)
    assert 'skipping tracklet -1' in capsys.readouterr().out

    track = _FakeMHTTrack([1, 2], lnprob=-1.0, gate_result=(1.0, 0.01))
    hypothesis = Hypothesis([track], nsat=10)
    mht = MHT(data, nsat=10, truth={1: 'A', 2: 'A'}, hypotheses=[hypothesis])
    monkeypatch.setattr(ct, 'Track', lambda satids, data, **kwargs: _FakeMHTTrack(satids, lnprob=-0.25))
    mht.add_tracklet(2)
    assert track.added_track in mht.track2hyp

    class BadHypothesis:
        tracks = []
        lnprob = 0.0

        def ntracklet(self):
            return 0

    monkeypatch.setattr(ct.Hypothesis, 'addto', staticmethod(lambda *args, **kwargs: BadHypothesis()))
    monkeypatch.setattr(ct.pdb, 'set_trace', lambda: None)
    mht = MHT(data[0:1], nsat=10, hypotheses=[Hypothesis([], nsat=10)])
    mht.add_tracklet(1)


def test_mht_prune_noop_and_empty_keep_error():
    short_track = _FakeMHTTrack([1, 2], lnprob=0.0)
    short_hyp = Hypothesis([short_track], nsat=20)
    short_hyp.lnprob = 0.0
    mht = object.__new__(MHT)
    mht.hypotheses = [short_hyp]
    mht.track2hyp = {short_track: [short_hyp]}
    np.testing.assert_array_equal(mht.prune_tracks(2, nconfirm=5), [True])

    mht.prune_tracks = lambda satid, nconfirm=6: np.array([False])
    mht.prune_stale_hypotheses = lambda newdead: np.array([True])
    mht._newly_dead_tracks = []
    mht.truth = None
    with pytest.raises(ValueError, match='should not be possible'):
        mht.prune(2, nconfirm=1)


def test_mht_prune_tracks_debug_and_truth_branches(monkeypatch, capsys):
    track_a = _FakeMHTTrack([1], lnprob=0.0)
    track_b = _FakeMHTTrack([2], lnprob=-1.0)
    hyp_a = Hypothesis([track_a], nsat=20)
    hyp_b = Hypothesis([track_b], nsat=20)
    hyp_a.lnprob = 0.0
    hyp_b.lnprob = -1.0

    mht = object.__new__(MHT)
    mht.hypotheses = [hyp_a, hyp_b]
    mht.track2hyp = {track_a: [hyp_a], track_b: [hyp_b]}
    monkeypatch.setattr(ct.pdb, 'set_trace', lambda: None)
    with pytest.raises(IndexError):
        mht.prune_tracks(99, nconfirm=0)

    truth_track_a = _FakeMHTTrack([1], lnprob=0.0)
    truth_track_b = _FakeMHTTrack([2], lnprob=0.0)
    mixed_track = _FakeMHTTrack([1, 2], lnprob=-2.0)
    truth_hyp = Hypothesis([truth_track_a, truth_track_b], nsat=20)
    mixed_hyp = Hypothesis([mixed_track], nsat=20)
    truth_hyp.lnprob = 0.0
    mixed_hyp.lnprob = -2.0
    mht.hypotheses = [truth_hyp, mixed_hyp]
    mht.track2hyp = {
        truth_track_a: [truth_hyp],
        truth_track_b: [truth_hyp],
        mixed_track: [mixed_hyp],
    }
    mht._newly_dead_tracks = []
    mht.truth = {1: 'A', 2: 'B'}
    mht.prune_tracks = lambda satid, nconfirm=6: np.array([True, True])
    mht.prune_stale_hypotheses = lambda newdead: np.array([True, True])
    mht.prune(2, keeponlytrue=True, nconfirm=1)
    assert mht.hypotheses == [truth_hyp]
    assert 'truth: dlnprob' in capsys.readouterr().out

    mht.hypotheses = [truth_hyp, mixed_hyp]
    mht.track2hyp = {
        truth_track_a: [truth_hyp],
        truth_track_b: [truth_hyp],
        mixed_track: [mixed_hyp],
    }
    mht.prune_tracks = lambda satid, nconfirm=6: np.array([False, True])
    mht.prune_stale_hypotheses = lambda newdead: np.array([True, True])
    mht.prune(2, keeponlytrue=False, nconfirm=1)
    assert 'warning: true solution is no longer included.' in capsys.readouterr().out


def test_mht_add_tracklet_updates_gated_tracks_and_singletons(monkeypatch):
    data = _minimal_track_table([1, 2], [0.0, 10.0])
    existing = _FakeMHTTrack([1], lnprob=-1.0)
    hypothesis = Hypothesis([existing], nsat=10)
    mht = MHT(data, nsat=10, hypotheses=[hypothesis], approximate=True)
    created = []

    def fake_track(satids, data, **kwargs):
        track = _FakeMHTTrack(satids, lnprob=-0.25)
        created.append((track, kwargs.get('propagator')))
        return track

    monkeypatch.setattr(ct, 'Track', fake_track)

    mht.add_tracklet(2)

    assert existing.updated is True
    assert existing.added_track in mht.track2hyp
    assert existing.added_track.approximated_with is mht.propagator
    assert created[0][1] is None
    assert created[0][0] in mht.track2hyp
    assert len(mht.hypotheses) == 2
    assert mht.nfit == 1


def test_mht_add_tracklet_marks_tracks_dead_when_wrap_uncertain(monkeypatch):
    data = _minimal_track_table([1, 2], [0.0, 10.0])
    dying = _FakeMHTTrack([1], lnprob=-1.0, gate_result=(1.0, np.pi))
    hypothesis = Hypothesis([dying], nsat=10)
    mht = MHT(data, nsat=10, hypotheses=[hypothesis])

    monkeypatch.setattr(
        ct, 'Track',
        lambda satids, data, **kwargs: _FakeMHTTrack(satids, lnprob=-0.25))

    mht.add_tracklet(2)

    assert dying.dead is True
    assert mht._newly_dead_tracks == [dying]
    assert mht.nfit == 0


def test_mht_prune_tracks_and_stale_hypotheses():
    best_track = _FakeMHTTrack([1, 2, 3, 4, 5], lnprob=0.0)
    partial_track = _FakeMHTTrack([1, 2, 5], lnprob=-1.0)
    best_hyp = Hypothesis([best_track], nsat=20)
    partial_hyp = Hypothesis([partial_track], nsat=20)
    best_hyp.lnprob = 0.0
    partial_hyp.lnprob = -1.0

    mht = object.__new__(MHT)
    mht.hypotheses = [best_hyp, partial_hyp]
    mht.track2hyp = {best_track: [best_hyp], partial_track: [partial_hyp]}

    keep = mht.prune_tracks(5, nconfirm=2)
    np.testing.assert_array_equal(keep, [True, False])

    live = _FakeMHTTrack([10], lnprob=0.0)
    dead_low = _FakeMHTTrack([11], lnprob=-10.0)
    dead_high = _FakeMHTTrack([12], lnprob=-1.0)
    dead_low.dead = True
    dead_high.dead = True
    low_hyp = Hypothesis([live, dead_low], nsat=20)
    high_hyp = Hypothesis([live, dead_high], nsat=20)
    low_hyp.lnprob = -10.0
    high_hyp.lnprob = -1.0
    mht.hypotheses = [low_hyp, high_hyp]
    mht.track2hyp = {dead_low: [low_hyp], dead_high: [high_hyp]}

    keep = mht.prune_stale_hypotheses([dead_low, dead_high])
    np.testing.assert_array_equal(keep, [False, True])

    np.testing.assert_array_equal(
        mht.prune_stale_hypotheses([]), [True, True])


def test_mht_prune_and_consistency_checks(monkeypatch):
    tracks = [_FakeMHTTrack([i], lnprob=-float(i)) for i in range(3)]
    hypotheses = [Hypothesis([track], nsat=20) for track in tracks]
    for i, hypothesis in enumerate(hypotheses):
        hypothesis.lnprob = -float(i)

    mht = object.__new__(MHT)
    mht.hypotheses = hypotheses
    mht.track2hyp = {track: [hypothesis]
                     for track, hypothesis in zip(tracks, hypotheses)}
    mht._newly_dead_tracks = []
    mht.truth = None
    mht.prune_tracks = lambda satid, nconfirm=6: np.array([True, True, False])
    mht.prune_stale_hypotheses = lambda newdead: np.array([True, False, True])

    mht.prune(0, nkeepmax=3, pkeep=1e-9, nconfirm=2)
    assert mht.hypotheses == [hypotheses[0]]
    assert list(mht.track2hyp) == [tracks[0]]

    assert MHT.flag_inconsistency({tracks[0]: [hypotheses[0]]},
                                  [hypotheses[0]]) == 0
    monkeypatch.setattr(ct.pdb, 'set_trace', lambda: None)
    assert MHT.flag_inconsistency({tracks[0]: []}, [hypotheses[0]]) == 10
    assert MHT.flag_inconsistency({tracks[0]: [Hypothesis([], nsat=20)]}, []) == 1
    mismatch_hypotheses = [
        Hypothesis([tracks[0]], nsat=20),
        Hypothesis([tracks[1], tracks[2]], nsat=20),
    ]
    assert MHT.flag_inconsistency(
        {
            tracks[0]: [mismatch_hypotheses[0]],
            tracks[1]: [mismatch_hypotheses[1]],
            tracks[2]: [mismatch_hypotheses[1]],
        },
        mismatch_hypotheses,
    ) == 4


def test_iterate_mht_trims_long_tracks_and_runs_new_mht(monkeypatch):
    data = _minimal_track_table([1, 2, 3, 4, 5, 6],
                                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    class IterTrack(_FakeMHTTrack):
        def __init__(self, satids, data, **kwargs):
            super().__init__(satids, lnprob=-0.5, chi2=8.0)
            self.data = data
            self.param = kwargs.get('guess', np.arange(7.0))
            self.priors = kwargs.get('priors')
            self.mode = kwargs.get('mode', 'rv')
            self.propagator = kwargs.get('propagator')
            self.orbitattr = kwargs.get('orbitattr')

    long_track = IterTrack([1, 2, 3, 4], data)
    short_track = IterTrack([5], data)
    dead_track = IterTrack([6, 7, 8, 9], data)
    dead_track.dead = True
    low_hyp = Hypothesis([short_track], nsat=50)
    best_hyp = Hypothesis([long_track, short_track, dead_track], nsat=50)
    low_hyp.lnprob = -10.0
    best_hyp.lnprob = 1.0

    class FakeOldMHT:
        hypotheses = [low_hyp, best_hyp]
        nsat = 50
        propagator = object()
        mode = 'rv'
        approximate = True
        priors = ['prior']
        truth = {1: 'A'}
        orbitattr = ['cr']

    created = {}

    class FakeNewMHT:
        def __init__(self, data_arg, **kwargs):
            created['data'] = data_arg
            created['kwargs'] = kwargs
            self.run_kwargs = None

        def run(self, **kwargs):
            self.run_kwargs = kwargs

    monkeypatch.setattr(ct, 'Track', IterTrack)
    monkeypatch.setattr(ct, 'MHT', FakeNewMHT)

    newmht = iterate_mht(data, FakeOldMHT(), nminlength=2, trimends=1,
                         first=2, last=5)

    initial_track = created['kwargs']['hypotheses'][0].tracks[0]
    assert initial_track.satIDs == [2, 3]
    assert initial_track.approximated_with is None
    np.testing.assert_array_equal(
        created['kwargs']['fitonly'],
        np.array([True, False, False, True, True, True]),
    )
    assert created['kwargs']['approximate'] is True
    assert newmht.run_kwargs == {'first': 2, 'last': 5}
