import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import Longitude, Latitude
from astropy.table import QTable
from types import SimpleNamespace
import pytest

import ssapy
from ssapy.utils import cluster_emcee_walkers, norm
from ssapy.constants import RGEO, VGEO
from .ssapy_test_helpers import timer, checkAngle


@timer
def test_prior():
    rprior = ssapy.rvsampler.RPrior(1e7, 1e5)
    # Just make a dummy orbit
    class Dummy:
        pass
    orbit = Dummy()
    orbit.r = np.array([0, 0, 1e7])  # right on target
    np.testing.assert_allclose(rprior(orbit), 0.0, rtol=0, atol=1e-12)
    # Now move one sigma away
    orbit.r = np.array([0, 0, 1.01e7])
    np.testing.assert_allclose(rprior(orbit), -0.5, rtol=0, atol=1e-12)
    # 100-sigma
    orbit.r = np.array([0, 0, 2e7])
    np.testing.assert_allclose(rprior(orbit), -0.5*100**2, rtol=0, atol=1e-12)

    # Basically the same for priors on velocity, semimajor axis, eccentricity...
    vprior = ssapy.rvsampler.VPrior(1e3, 1e1)
    orbit.v = np.array([0, -1e3, 0])  # right on target
    np.testing.assert_allclose(vprior(orbit), 0.0, rtol=0, atol=1e-12)
    orbit.v = np.array([-1.01e3, 0, 0])
    np.testing.assert_allclose(vprior(orbit), -0.5, rtol=0, atol=1e-12)
    orbit.v = np.array([0, 0, 2e3])
    np.testing.assert_allclose(vprior(orbit), -0.5*100**2, rtol=0, atol=1e-12)

    aprior = ssapy.rvsampler.APrior(1e7, 1e5)
    orbit.a = 1e7
    np.testing.assert_allclose(aprior(orbit), 0.0, rtol=0, atol=1e-12)
    orbit.a = 1.01e7
    np.testing.assert_allclose(aprior(orbit), -0.5, rtol=0, atol=1e-12)
    orbit.a = 2e7
    np.testing.assert_allclose(aprior(orbit), -0.5*100**2, rtol=0, atol=1e-12)

    eprior = ssapy.rvsampler.EPrior(0.7, 0.01)
    orbit.e = 0.7
    np.testing.assert_allclose(eprior(orbit), 0.0, rtol=0, atol=1e-12)
    orbit.e = 0.71
    np.testing.assert_allclose(eprior(orbit), -0.5, rtol=0, atol=1e-12)
    orbit.e = 0.8
    np.testing.assert_allclose(eprior(orbit), -0.5*10**2, rtol=0, atol=1e-12)

    # Exercise chi-output and object-property priors used by optimizers.
    orbit.r = np.array([0, 0, 1.01e7])
    orbit.v = np.array([-1.01e3, 0, 0])
    orbit.a = 1.01e7
    orbit.e = 0.71
    orbit.ex = 0.2
    orbit.ey = -0.4
    orbit.propkw = {'area': 10.0}
    np.testing.assert_allclose(rprior(orbit, chi=True), 1.0)
    np.testing.assert_allclose(vprior(orbit, chi=True), 1.0)
    np.testing.assert_allclose(aprior(orbit, chi=True), 1.0)
    np.testing.assert_allclose(eprior(orbit, chi=True), 1.0)
    eqprior = ssapy.rvsampler.EquinoctialExEyPrior(sigma=0.2)
    np.testing.assert_allclose(eqprior(orbit, chi=True), [1.0, -2.0])
    np.testing.assert_allclose(eqprior(orbit), [-0.5, -2.0])
    logarea = ssapy.rvsampler.Log10AreaPrior(mean=0.0, sigma=0.5)
    np.testing.assert_allclose(logarea(orbit, chi=True), 2.0)
    np.testing.assert_allclose(logarea(orbit), -2.0)
    area = ssapy.rvsampler.AreaPrior(mean=6.0, sigma=2.0)
    np.testing.assert_allclose(area(orbit, chi=True), 2.0)
    np.testing.assert_allclose(area(orbit), -2.0)


def test_lightweight_sampler_helpers_and_translators():
    np.random.seed(7)
    center = np.array([1.0, 2.0, 3.0])
    spread = np.array([0.1, 0.2, 0.3])
    samples = ssapy.rvsampler.sample_ball(center, spread, nSample=4)
    assert samples.shape == (4, 3)
    with pytest.raises(AssertionError):
        ssapy.rvsampler.sample_ball([1.0, 2.0], [1.0], nSample=1)

    direct_samples = np.arange(18.0).reshape(3, 6)
    direct = ssapy.DirectInitializer(direct_samples, replace=False)
    assert direct(nSample=5).shape == (5, 6)
    direct_replace = ssapy.DirectInitializer(direct_samples, replace=True)
    assert direct_replace(nSample=5).shape == (5, 6)

    proposal = ssapy.MVNormalProposal(np.eye(6) * 0.01)
    assert proposal.propose(np.zeros(6)).shape == (6,)
    rv_proposal = ssapy.RVSigmaProposal(2.0, 3.0)
    np.testing.assert_allclose(np.diag(rv_proposal.cov), [4.0, 4.0, 4.0, 9.0, 9.0, 9.0])

    translator = ssapy.rvsampler.ParamOrbitTranslator(
        np.array([1.0, 2.0, 3.0, 4.0]), Time("J2000"),
        fixed=np.array([True, False, True, False]), orbitattr=['log10area', 'mass']
    )
    np.testing.assert_allclose(translator.fullparam(np.zeros(4)), [1.0, 0.0, 3.0, 0.0])
    np.testing.assert_allclose(translator.optimizeparam(np.arange(4.0)), [0.0, 2.0])
    assert translator.get_propkw_from_fullparam([0, 0, 0, 0, 0, 0, 2.0, 5.0]) == {'area': 100.0, 'mass': 5.0}

    class DummyOrbit:
        propkw = {'area': 100.0, 'mass': 5.0}

    np.testing.assert_allclose(translator.get_propkw_from_orbit(DummyOrbit()), [2.0, 5.0])
    assert translator.input_param_translation([1, 2]) == [1, 2]
    assert translator.output_covar_translation(np.eye(2)).shape == (2, 2)
    with pytest.raises(NotImplementedError):
        translator.param_to_orbit()
    with pytest.raises(NotImplementedError):
        translator.orbit_to_param()

    rv_translator = ssapy.rvsampler.ParamOrbitRV(
        np.zeros(7), Time("J2000"), orbitattr=['log10area']
    )
    orbit = rv_translator.param_to_orbit(np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0]))
    np.testing.assert_allclose(orbit.r, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(orbit.v, [0.1, 0.2, 0.3])
    assert orbit.propkw['area'] == 10.0
    rv_param = rv_translator.orbit_to_param(orbit)
    np.testing.assert_allclose(rv_param, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0])

    eq_translator = ssapy.rvsampler.ParamOrbitEquinoctial(
        np.array([4.2e7, 0.01, 0.02, 0.03, 0.04, 0.5]), Time("J2000")
    )
    translated = eq_translator.input_param_translation(np.array([4.2e7, 0.01, 0.02, 0.03, 0.04, 0.5]))
    np.testing.assert_allclose(translated[0], 4.2)
    covar = eq_translator.output_covar_translation(np.eye(6))
    assert covar[0, 0] == 1e14
    eq_orbit = eq_translator.param_to_orbit(np.array([4.2, 0.01, 0.02, 1.5, 0.0, 0.5]))
    assert eq_orbit.e < 1.0
    eq_param = eq_translator.orbit_to_param(eq_orbit)
    assert len(eq_param) == 6

    base_orbit = ssapy.Orbit(np.array([7000e3, 0.0, 0.0]), np.array([0.0, 7500.0, 0.0]), Time("J2000"))
    init_obs_pos = np.zeros(3)
    init_obs_vel = np.zeros(3)
    angle_seed = ssapy.compute.rvObsToRaDecRate(base_orbit.r, base_orbit.v, init_obs_pos, init_obs_vel)
    angle_translator = ssapy.rvsampler.ParamOrbitAngle(
        np.array(angle_seed), base_orbit.t, init_obs_pos, init_obs_vel
    )
    angle_orbit = angle_translator.param_to_orbit(np.array(angle_seed))
    np.testing.assert_allclose(angle_orbit.r, base_orbit.r, rtol=0, atol=1e-6)
    np.testing.assert_allclose(angle_orbit.v, base_orbit.v, rtol=0, atol=1e-6)
    angle_param = angle_translator.orbit_to_param(angle_orbit)
    np.testing.assert_allclose(angle_param, angle_seed, rtol=0, atol=1e-12)


def test_mh_sampler_acceptance_and_rejection_paths(monkeypatch):
    monkeypatch.setattr(np.random, 'uniform', lambda *args, **kwargs: 0.99)
    initial = np.zeros((2, 6))
    proposals = [np.ones(6), -np.ones(6), np.ones(6) * 2, -np.ones(6) * 2]

    class Initializer:
        def __call__(self, n):
            assert n == 2
            return initial.copy()

    class Proposer:
        def __init__(self):
            self.index = 0

        def propose(self, p):
            out = proposals[self.index].copy()
            self.index += 1
            return out

    def probfn(p):
        score = p[0]
        return score, score / 10.0

    sampler = ssapy.MHSampler(probfn, Initializer(), Proposer(), nChain=2)
    sampler._step()
    assert sampler.nStep == 2
    assert sampler.nAccept == 1
    assert sampler.acceptanceRatio == 0.5

    chain, lnprob, lnprior = sampler.sample(nBurn=0, nStep=1)
    assert chain.shape == (1, 2, 6)
    assert lnprob.shape == (1, 2)
    assert lnprior.shape == (1, 2)
    assert sampler.nStep == 2


def test_damper_helpers():
    chi = np.array([-2.0, 0.0, 2.0])
    damped = ssapy.rvsampler.damper(chi, damp=2.0)
    assert damped[0] < 0
    assert damped[1] == 0
    assert damped[2] > 0
    np.testing.assert_allclose(
        ssapy.rvsampler.damper_deriv(chi, damp=2.0, derivnum=1),
        (1 + np.abs(chi) / 2.0) ** (-0.5),
    )
    np.testing.assert_allclose(
        ssapy.rvsampler.damper_deriv(chi, damp=2.0, derivnum=2),
        -0.25 * np.sign(chi) * (1 + np.abs(chi) / 2.0) ** (-1.5),
    )


def _circular_guess_arc():
    time = Time("J2000")
    orbit = ssapy.Orbit.fromKeplerianElements(
        RGEO,
        1e-6,
        1e-5,
        0.1,
        0.2,
        0.3,
        time,
    )
    observer = ssapy.EarthObserver(lon=22.0, lat=33.0, elevation=300.0)
    times = time + np.array([0.0, 10.0]) * u.s
    ra, dec, _ = ssapy.radec(orbit, times, observer=observer)
    r_station, v_station = observer.getRV(times)

    arc = QTable()
    arc['ra'] = Longitude(ra * u.rad)
    arc['dec'] = Latitude(dec * u.rad)
    arc['time'] = times
    arc['rStation_GCRF'] = r_station * u.m
    arc['vStation_GCRF'] = v_station * u.m / u.s
    return orbit, arc


def test_circular_guess_recovers_synthetic_circular_orbit():
    orbit, arc = _circular_guess_arc()

    state, epoch = ssapy.circular_guess(arc)
    r_expected, v_expected = ssapy.rv(orbit, epoch)

    np.testing.assert_allclose(state[:3], r_expected, rtol=0, atol=20.0)
    np.testing.assert_allclose(state[3:], v_expected, rtol=0, atol=2e-3)


def test_circular_guess_uses_proper_motion_columns():
    orbit, arc = _circular_guess_arc()
    dt = (arc['time'][1] - arc['time'][0]).to(u.s).value
    pmra = (((arc['ra'][1] - arc['ra'][0]).to(u.rad).value + np.pi) % (2 * np.pi) - np.pi)
    pmra *= np.cos(arc['dec'][0].to(u.rad).value) / dt
    pmdec = (arc['dec'][1] - arc['dec'][0]).to(u.rad).value / dt

    arcpm = QTable()
    arcpm['ra'] = [arc['ra'][0]]
    arcpm['dec'] = [arc['dec'][0]]
    arcpm['time'] = Time([arc['time'][0]])
    arcpm['rStation_GCRF'] = [arc['rStation_GCRF'][0]]
    arcpm['vStation_GCRF'] = [arc['vStation_GCRF'][0]]
    arcpm['pmra'] = [pmra] * u.rad / u.s
    arcpm['pmdec'] = [pmdec] * u.rad / u.s

    state, epoch = ssapy.circular_guess(arcpm)
    r_expected, v_expected = ssapy.rv(orbit, epoch)

    np.testing.assert_allclose(state[:3], r_expected, rtol=0, atol=20.0)
    np.testing.assert_allclose(state[3:], v_expected, rtol=0, atol=2e-3)


def test_circular_guess_edge_guards(monkeypatch):
    arc = QTable()
    arc['ra'] = Longitude([0.0, 0.0] * u.rad)
    arc['dec'] = Latitude([0.0, 0.0] * u.rad)
    arc['time'] = Time([0.0, 0.0], format='gps')
    arc['rStation_GCRF'] = np.zeros((2, 3)) * u.m
    arc['vStation_GCRF'] = np.zeros((2, 3)) * u.m / u.s

    with np.errstate(divide='ignore', invalid='ignore'):
        state, epoch = ssapy.circular_guess(arc)
    np.testing.assert_allclose(state[:3], [200000.0, 0.0, 0.0])
    np.testing.assert_allclose(state[3:], [0.0, 0.0, 0.0])
    assert epoch.gps == pytest.approx(0.0)

    _, real_arc = _circular_guess_arc()

    import scipy.optimize

    monkeypatch.setattr(
        scipy.optimize,
        'root_scalar',
        lambda fn, bracket: SimpleNamespace(root=300000.0),
    )

    def reflected_leastsq(fn, x0, args=(), full_output=False):
        fn(args[0] + 1.0, *args)
        return np.array([500000.0]), None, None, '', 1

    monkeypatch.setattr(scipy.optimize, 'leastsq', reflected_leastsq)
    state, _ = ssapy.circular_guess(real_arc)
    assert np.all(np.isfinite(state))

    monkeypatch.setattr(
        scipy.optimize,
        'root_scalar',
        lambda fn, bracket: SimpleNamespace(root=1.0e7),
    )
    monkeypatch.setattr(
        scipy.optimize,
        'leastsq',
        lambda fn, x0, args=(), full_output=False:
            (np.array([5.0e6]), None, None, '', 1),
    )
    monkeypatch.setattr(
        ssapy.rvsampler,
        'radecRateObsToRV',
        lambda *args, **kwargs: (np.full(3, np.nan), np.zeros(3)),
    )
    with pytest.raises(RuntimeError, match='non-finite state vector'):
        ssapy.circular_guess(real_arc)


class _LinearOrbitProbability:
    epoch = Time("J2000")

    def chi(self, orbit):
        if hasattr(orbit, 'r'):
            return [np.hstack([orbit.r / 1e7, orbit.v / 1e4])]
        return [np.asarray(orbit, dtype=float)]


def test_least_squares_optimizer_translates_fit_options(monkeypatch):
    import scipy.optimize

    prob = _LinearOrbitProbability()
    init = np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
    optimizer = ssapy.rvsampler.LeastSquaresOptimizer(
        prob, init, ssapy.rvsampler.ParamOrbitRV)
    np.testing.assert_allclose(
        optimizer.resid(init), [0.7, 0.0, 0.0, 0.0, 0.75, 0.0])

    captured = {}

    def fake_least_squares(resid, par0, **kwargs):
        captured['par0'] = par0.copy()
        captured['kwargs'] = kwargs
        return SimpleNamespace(
            x=np.array([7000e3, 0.0, 0.0, 0.0, 20000.0, 0.0]),
            jac=np.eye(6),
            fun=resid(par0),
            success=True,
        )

    monkeypatch.setattr(scipy.optimize, 'least_squares', fake_least_squares)

    fit = optimizer.optimize(maxfev=12)

    np.testing.assert_allclose(captured['par0'], init)
    assert captured['kwargs']['max_nfev'] == 12
    assert 'maxfev' not in captured['kwargs']
    assert optimizer.result.success is False
    assert optimizer.result.hithyperbolicorbit is True
    assert optimizer.result.covar.shape == (6, 6)
    np.testing.assert_allclose(optimizer.result.residual,
                               [0.7, 0.0, 0.0, 0.0, 0.75, 0.0])
    assert norm(fit[3:]) < 20000.0


def test_lm_optimizer_jacobian_and_hyperbolic_guard(monkeypatch):
    import lmfit

    captured = []

    def fake_minimize(resid, params, Dfun=None, **kwargs):
        captured.append(Dfun)
        return SimpleNamespace(params=params, success=True)

    monkeypatch.setattr(lmfit, 'minimize', fake_minimize)

    with pytest.warns(UserWarning, match='deprecated'):
        optimizer = ssapy.rvsampler.LMOptimizer(
            _LinearOrbitProbability(),
            np.array([7000e3, 0.0, 0.0, 0.0, 20000.0, 0.0]),
        )
    fit = optimizer.optimize()

    assert captured[-1].__self__ is optimizer
    assert captured[-1].__func__ is optimizer._jac.__func__
    assert optimizer.result.success is False
    assert optimizer.result.hithyperbolicorbit is True
    assert optimizer.result.covar.shape == (6, 6)
    assert norm(fit[3:]) < 20000.0

    with pytest.warns(UserWarning, match='deprecated'):
        no_jac_optimizer = ssapy.rvsampler.LMOptimizer(
            _LinearOrbitProbability(),
            np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0]),
        )
    no_jac_optimizer.optimize(usejac=False)
    assert captured[-1] is None
    assert not hasattr(no_jac_optimizer.result, 'covar')


def test_lm_angular_optimizer_smoke_path(monkeypatch):
    import lmfit

    monkeypatch.setattr(
        lmfit, 'minimize',
        lambda resid, params, Dfun=None, **kwargs: SimpleNamespace(
            params=params, success=True))

    orbit = ssapy.Orbit(
        np.array([7000e3, 0.0, 0.0]),
        np.array([0.0, 7500.0, 0.0]),
        Time("J2000"),
    )
    obs_pos = np.zeros(3)
    obs_vel = np.zeros(3)
    init_guess = ssapy.compute.rvObsToRaDecRate(
        orbit.r, orbit.v, obs_pos, obs_vel)

    with pytest.warns(UserWarning, match='deprecated'):
        optimizer = ssapy.rvsampler.LMOptimizerAngular(
            _LinearOrbitProbability(), init_guess, obs_pos, obs_vel)

    fit = optimizer.optimize(usejac=False)
    assert len(fit) == 6
    assert not hasattr(optimizer.result, 'covar')


def test_lm_angular_optimizer_jacobian_and_hyperbolic_guard(monkeypatch):
    import lmfit
    import warnings

    monkeypatch.setattr(
        lmfit, 'minimize',
        lambda resid, params, Dfun=None, **kwargs: SimpleNamespace(
            params=params, success=True))

    obs_pos = np.zeros(3)
    obs_vel = np.zeros(3)
    init_guess = np.array([0.0, 0.0, 7000e3, 0.0, 0.0, 20000.0])

    with pytest.warns(UserWarning, match='deprecated'):
        optimizer = ssapy.rvsampler.LMOptimizerAngular(
            _LinearOrbitProbability(), init_guess, obs_pos, obs_vel)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='invalid value encountered in sqrt',
            category=RuntimeWarning,
        )
        assert optimizer._getOrbit(init_guess).r.shape == (3,)
        assert optimizer._resid(init_guess).shape == (6,)
        assert optimizer._jac(init_guess).shape == (6, 6)

        fit = optimizer.optimize(usejac=True)

    assert len(fit) == 6
    assert optimizer.result.success is False
    assert optimizer.result.hithyperbolicorbit is True
    assert optimizer.result.covar.shape == (6, 6)


def test_legacy_element_optimizers_with_mocked_lmfit(monkeypatch):
    import lmfit

    captured = []

    def fake_minimize(resid, params, Dfun=None, **kwargs):
        captured.append(Dfun)
        return SimpleNamespace(params=params, success=True)

    monkeypatch.setattr(lmfit, 'minimize', fake_minimize)

    prob = _LinearOrbitProbability()
    initel = np.array([4.2e7, 0.01, 0.02, 0.03, 0.04, 0.5])
    sgp4_optimizer = ssapy.rvsampler.SGP4LMOptimizer(prob, initel)
    kep = ssapy.rvsampler.eq2kep(initel)
    np.testing.assert_allclose(kep[:2], [4.2e7, np.hypot(0.03, 0.04)])
    assert sgp4_optimizer._getOrbit(initel).r.shape == (3,)
    assert sgp4_optimizer._getOrbit(sgp4_optimizer.p).r.shape == (3,)
    sgp4_optimizer._getOrbit = lambda p: np.array(
        [p[name].value for name in sgp4_optimizer.fieldnames]
        if isinstance(p, dict) else p,
        dtype=float,
    )
    assert sgp4_optimizer._jac(np.arange(1.0, 7.0)).shape == (6, 6)
    sgp4_fit = sgp4_optimizer.optimize()
    np.testing.assert_allclose(sgp4_fit, initel)
    assert captured[-1].__self__ is sgp4_optimizer
    assert captured[-1].__func__ is sgp4_optimizer._jac.__func__

    with pytest.warns(UserWarning, match='deprecated'):
        eq_optimizer = ssapy.rvsampler.EquinoctialLMOptimizer(
            prob, np.array([4.2e7, 0.01, 0.02, 1.0, 1.0, 0.5]))
    assert eq_optimizer._getOrbit(
        np.array([4.2, 0.01, 0.02, 1.0, 1.0, 0.5])).e < 1.0
    assert eq_optimizer._getOrbit(eq_optimizer.p).e < 1.0
    eq_optimizer._getOrbit = lambda p: np.array(
        [p[name].value for name in eq_optimizer.fieldnames]
        if isinstance(p, dict) else p,
        dtype=float,
    )
    eq_fit = eq_optimizer.optimize()
    assert eq_fit[0] == pytest.approx(4.2e7)
    assert eq_optimizer.result.covar.shape == (6, 6)
    assert captured[-1].__self__ is eq_optimizer
    assert captured[-1].__func__ is eq_optimizer._jac.__func__


@timer
def test_initializer():
    np.random.seed(5)

    # Construct a nearly GEOstationary orbit and an observer.  Should be able to very nearly recover
    # the true position and velocity at epoch

    a = RGEO
    e = 0.000001  # nearly circular
    i = 0.00001  # nearly equatorial
    pa = 1.02  # doesn't really matter since equatorial
    raan = 2.3  # same
    trueAnomaly = 0.1  # doesn't matter
    t = Time("J2000")
    orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, t)

    lon = 22.0
    lat = 33.0
    elevation = 300.0
    observer = ssapy.EarthObserver(lon, lat, elevation)

    times = t + np.linspace(0, 0.1, 100) * u.h

    ra, dec, _ = ssapy.radec(orbit, times, observer=observer)

    arc = QTable()
    arc['ra'] = ra*u.rad
    arc['dec'] = dec*u.rad
    arc['time'] = times

    rStation, vStation = observer.getRV(times)

    arc['rStation_GCRF'] = rStation*u.m
    arc['vStation_GCRF'] = vStation*u.m/u.s

    initializer = ssapy.GEOProjectionInitializer(arc, rSigma=RGEO*1e-6, vSigma=VGEO*1e-6)
    samples = initializer(nSample=100)

    assert samples.shape == (100, 6)

    rMean = np.mean(samples[:,0:3], axis=0)
    vMean = np.mean(samples[:,3:6], axis=0)
    np.testing.assert_allclose(rMean, orbit.r, rtol=0, atol=RGEO*1e-3)
    np.testing.assert_allclose(vMean, orbit.v, rtol=0, atol=VGEO*1e-3)

    # Check we get reasonable results with the other initializers too.
    initializer = ssapy.GaussianRVInitializer(rMean, vMean, rSigma=RGEO*0.1, vSigma=VGEO*0.1)
    samples = initializer(nSample=100)
    np.testing.assert_allclose(np.mean(samples[:,0:3], axis=0), rMean, rtol=0, atol=RGEO*0.03)
    np.testing.assert_allclose(np.mean(samples[:,3:6], axis=0), vMean, rtol=0, atol=VGEO*0.03)

    initializer = ssapy.DirectInitializer(samples, replace=False)
    np.testing.assert_equal(samples, initializer(nSample=100))
    np.testing.assert_equal(np.tile(samples, 2), initializer(nSample=200))

    for n in np.random.randint(1, 101, size=10):
        out = initializer(nSample=n)
        assert np.unique(out, axis=0).shape == (n, 6)
        assert np.all([o in samples for o in out])

    initializer = ssapy.DirectInitializer(samples, replace=True)
    alwaysUnique = True
    for n in np.random.randint(1, 101, size=10):
        out = initializer(nSample=n)
        assert np.all([o in samples for o in out])
        alwaysUnique &= np.unique(out, axis=0).shape == (n, 6)
    # While it's possible to get unique elements when sampling with replacement, we probably won't
    # always.  So check that here, even though there's a small chance this could fail.
    assert not alwaysUnique

    # Try the DistanceProjectionInitializer; if I use the true range, then this should be pretty
    # accurate.
    range = norm(orbit.r - rStation[0])
    initializer = ssapy.DistanceProjectionInitializer(arc, range, rSigma=1e-3, vSigma=1e-6)
    np.testing.assert_allclose(initializer(1)[0][0:3], orbit.r, rtol=0, atol=1e3)
    np.testing.assert_allclose(initializer(1)[0][3:6], orbit.v, rtol=0, atol=1)


@timer
def test_likelihood():
    np.random.seed(21)

    time1 = Time(2458316., format='jd')
    num_obs_per_track = 6

    # Setup observer location
    lon = 100.0
    lat = 33.0
    elevation = 1300.0
    earthObserver = ssapy.EarthObserver(lon, lat, elevation)

    # Create orbit for target
    orbit = ssapy.Orbit.fromKeplerianElements(RGEO*0.98, 0.01, 0.001, 0.1, 1.2, 1.03, time1)

    # Build observation times
    earthTimes = time1 + np.linspace(0, 4, num_obs_per_track)*u.h
    earthRaDec = ssapy.radec(orbit, earthTimes, observer=earthObserver)
    earthRStation, earthVStation = earthObserver.getRV(earthTimes)

    arc = QTable()
    arc['ra'] = Longitude(earthRaDec[0]*u.rad)
    arc['dec'] = Latitude(earthRaDec[1]*u.rad)
    arc['rStation_GCRF'] = earthRStation*u.m
    arc['vStation_GCRF'] = earthVStation*u.m/u.s
    arc['time'] = Time(earthTimes)
    arc['sigma'] = 1.0 * np.ones(num_obs_per_track, dtype=float)*u.arcsec

    # Check that the likelihood is the same when the input orbit epoch is different
    prob = ssapy.RVProbability(arc, time1)

    r1, v1 = ssapy.rv(orbit, time1)
    time2 = time1 + 6.0 * u.h
    r2, v2 = ssapy.rv(orbit, time2)
    # print("r0:",orbit.r,"v0:",orbit.v)
    # print("r1:",r1,"v1:",v1)
    # print("r2:",r2,"v2:",v2)

    orbit1 = ssapy.Orbit(r1, v1, time1)
    orbit2 = ssapy.Orbit(r2, v2, time2)
    # print("orbit0:", orbit.keplerianElements)
    # print("orbit1:", orbit1.keplerianElements)
    # print("orbit2:", orbit2.keplerianElements)

    lnL0 = prob.lnlike(orbit)
    lnL1 = prob.lnlike(orbit1)
    lnL2 = prob.lnlike(orbit2)
    # print("log likelihoods:", lnL0, lnL1, lnL2)
    np.testing.assert_allclose(lnL0, lnL2, rtol=0, atol=1e-14)
    np.testing.assert_allclose(lnL1, lnL2, rtol=0, atol=1e-14)


def test_rv_probability_meanpm_priors_and_bad_orbits(monkeypatch):
    time = Time("J2000")
    orbit = ssapy.Orbit(
        np.array([7000e3, 0.0, 0.0]),
        np.array([0.0, 7500.0, 0.0]),
        time,
    )
    times = time + np.array([0.0, 10.0]) * u.s
    obs_pos = np.zeros((2, 3))
    obs_vel = np.zeros((2, 3))
    ra, dec, _, pmra, pmdec, _ = ssapy.radec(
        orbit, times, obsPos=obs_pos, obsVel=obs_vel, rate=True)

    arc = QTable()
    arc['ra'] = Longitude(ra * u.rad)
    arc['dec'] = Latitude(dec * u.rad)
    arc['rStation_GCRF'] = obs_pos * u.m
    arc['vStation_GCRF'] = obs_vel * u.m / u.s
    arc['time'] = Time(times)
    arc['sigma'] = np.ones(2) * u.arcsec
    arc['dra'] = np.ones(2) * 1e-6 * u.rad
    arc['ddec'] = np.ones(2) * 1e-6 * u.rad
    arc['pmra'] = pmra * u.rad / u.s
    arc['pmdec'] = pmdec * u.rad / u.s
    arc['dpmra'] = np.ones(2) * 1e-9 * u.rad / u.s
    arc['dpmdec'] = np.ones(2) * 1e-9 * u.rad / u.s

    prob_meanpm = ssapy.RVProbability(arc, time, priors=[], meanpm=True)
    chi, chiprior = prob_meanpm.chi(orbit)
    np.testing.assert_allclose(chi, np.zeros(8), atol=1e-6)
    np.testing.assert_allclose(chiprior, [0.0])
    assert prob_meanpm.lnprior(orbit) == pytest.approx(0.0)

    prob = ssapy.RVProbability(arc, time, priors=[], damp=2.0)
    chi, chiprior = prob.chi(orbit)
    np.testing.assert_allclose(chi, np.zeros(4), atol=1e-6)
    np.testing.assert_allclose(chiprior, [0.0])

    def fake_radec(*args, **kwargs):
        return prob.ra + prob._raSigma, prob.dec + prob._decSigma, np.ones_like(prob.ra)

    monkeypatch.setattr(ssapy.rvsampler, 'radec', fake_radec)
    fast = ssapy.Orbit(
        np.array([7000e3, 0.0, 0.0]),
        np.array([0.0, 1.0e6, 0.0]),
        time,
    )
    small = ssapy.Orbit(
        np.array([0.5, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        time,
    )
    assert np.max(np.abs(prob.chi(fast)[0])) > 1e4
    assert np.max(np.abs(prob.chi(small)[0])) > 1e4

    with monkeypatch.context() as m:
        m.setattr(ssapy.rvsampler, 'Orbit', lambda *args, **kwargs: (_ for _ in ()).throw(ValueError()))
        lnprob, lnprior = prob.lnprob(np.zeros(6))
    assert lnprob == -np.inf
    assert lnprior == -np.inf

    prob.chi = lambda orbit: (np.array([np.inf]), np.array([2.0]))
    lnprob, lnprior = prob.lnprob(np.hstack([orbit.r, orbit.v]))
    assert lnprob == -np.inf
    assert lnprior == pytest.approx(-2.0)


def test_emcee_sampler_rejects_old_versions(monkeypatch):
    import emcee

    real_version = emcee.__version__
    initializer = lambda n: np.zeros((n, 6))
    probfn = lambda p: (0.0, 0.0)

    monkeypatch.setattr(emcee, '__version__', '2.2.1')
    with pytest.raises(ValueError, match='emcee version'):
        ssapy.EmceeSampler(probfn, initializer, nWalker=12)

    monkeypatch.setattr(emcee, '__version__', real_version)
    sampler = ssapy.EmceeSampler(probfn, initializer, nWalker=12)
    monkeypatch.setattr(emcee, '__version__', '2.2.1')
    with pytest.raises(ValueError, match='emcee version'):
        sampler.sample(nBurn=1, nStep=1)


@timer
def test_emcee_sampler():
    np.random.seed(57)

    # Let's make 2 observers; one on Earth and one in LEO
    lon = 100.0
    lat = 33.0
    elevation = 1300.0
    earthObserver = ssapy.EarthObserver(lon, lat, elevation)

    a = RGEO/7  # leo-ish
    e = 0.01
    i = np.pi/2-0.02 # Nearly polar orbit
    raan = 1.2
    pa = 2.0
    trueAnomaly = 2.03
    leoOrbit = ssapy.Orbit.fromKeplerianElements(a, e, i, raan, pa, trueAnomaly, Time("J2000"))
    leoObserver = ssapy.OrbitalObserver(leoOrbit)

    if __name__ == '__main__':
        ntest = 10
    else:
        ntest = 1
    for _ in range(ntest):
        r = np.random.uniform(RGEO*0.9, RGEO*1.1)
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        r = np.array([x, y, z])
        # Pick a velocity near VGEO
        v = np.random.uniform(VGEO*0.9, VGEO*1.1)
        # Pick a random direction
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        vx = v * np.cos(theta) * np.sin(phi)
        vy = v * np.sin(theta) * np.sin(phi)
        vz = v * np.cos(phi)
        v = np.array([vx, vy, vz])
        orbit = ssapy.Orbit(r, v, Time("J2000"))

        leoTimes = Time("J2000") + np.linspace(0, 10, 5)*u.h
        earthTimes = Time("J2000") + np.linspace(0, 10, 5)*u.h
        leoRaDec = ssapy.radec(orbit, leoTimes, observer=leoObserver)
        earthRaDec = ssapy.radec(orbit, earthTimes, observer=earthObserver)
        leoRStation, leoVStation = leoObserver.getRV(leoTimes)
        earthRStation, earthVStation = earthObserver.getRV(earthTimes)

        arc = QTable()
        arc['ra'] = Longitude(np.hstack([leoRaDec[0], earthRaDec[0]])*u.rad)
        arc['dec'] = Latitude(np.hstack([leoRaDec[1], earthRaDec[1]])*u.rad)
        arc['rStation_GCRF'] = np.vstack([leoRStation, earthRStation])*u.m
        arc['vStation_GCRF'] = np.vstack([leoVStation, earthVStation])*u.m/u.s
        arc['time'] = Time(np.hstack([leoTimes.mjd, earthTimes.mjd]), format='mjd')
        arc['sigma'] = np.ones(10, dtype=float)*u.arcsec

        # initialize somewhere near the truth, but perturbed a bit and with large uncertainties
        initializer = ssapy.GaussianRVInitializer(r*1.2, v*0.7, rSigma=0.1*RGEO, vSigma=0.1*VGEO)
        # Use R and A priors
        priors = [ssapy.rvsampler.RPrior(RGEO, RGEO*0.2), ssapy.rvsampler.APrior(RGEO, RGEO*0.2)]

        prob = ssapy.RVProbability(arc, Time("J2000"), priors=priors)
        sampler = ssapy.EmceeSampler(prob, initializer, nWalker=32)
        chain, lnprob, lnprior = sampler.sample(nBurn=350, nStep=100)
        chain, lnprob, lnprior = cluster_emcee_walkers(chain, lnprob, lnprior)
        rmean = np.mean(chain[...,0:3], axis=(0,1))
        vmean = np.mean(chain[...,3:6], axis=(0,1))

        np.testing.assert_allclose(rmean, r, rtol=0, atol=RGEO*0.01)
        np.testing.assert_allclose(vmean, v, rtol=0, atol=VGEO*0.01)
        print("r difference wrt GEO")
        print((rmean-r)/RGEO)
        print("v difference wrt GEO")
        print((vmean-v)/VGEO)


@timer
def test_mh_sampler():
    np.random.seed(577)

    # Let's make 2 observers; one on Earth and one in LEO
    lon = 100.0
    lat = 33.0
    elevation = 1300.0
    earthObserver = ssapy.EarthObserver(lon, lat, elevation)

    a = RGEO/7  # leo-ish
    e = 0.01
    i = np.pi/2-0.02 # Nearly polar orbit
    raan = 1.2
    pa = 2.0
    trueAnomaly = 2.03
    leoOrbit = ssapy.Orbit.fromKeplerianElements(a, e, i, raan, pa, trueAnomaly, Time("J2000"))
    leoObserver = ssapy.OrbitalObserver(leoOrbit)

    if __name__ == '__main__':
        ntest = 5
    else:
        ntest = 1
    for _ in range(ntest):
        r = np.random.uniform(RGEO*0.9, RGEO*1.1)
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        r = np.array([x, y, z])
        # Pick a velocity near VGEO
        v = np.random.uniform(VGEO*0.9, VGEO*1.1)
        # Pick a random direction
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        vx = v * np.cos(theta) * np.sin(phi)
        vy = v * np.sin(theta) * np.sin(phi)
        vz = v * np.cos(phi)
        v = np.array([vx, vy, vz])
        orbit = ssapy.Orbit(r, v, Time("J2000"))

        leoTimes = Time("J2000") + np.linspace(0, 10, 5)*u.h
        earthTimes = Time("J2000") + np.linspace(0, 10, 5)*u.h
        leoRaDec = ssapy.radec(orbit, leoTimes, observer=leoObserver)
        earthRaDec = ssapy.radec(orbit, earthTimes, observer=earthObserver)
        leoRStation, leoVStation = leoObserver.getRV(leoTimes)
        earthRStation, earthVStation = earthObserver.getRV(earthTimes)

        arc = QTable()
        arc['ra'] = Longitude(np.hstack([leoRaDec[0], earthRaDec[0]])*u.rad)
        arc['dec'] = Latitude(np.hstack([leoRaDec[1], earthRaDec[1]])*u.rad)
        arc['rStation_GCRF'] = np.vstack([leoRStation, earthRStation])*u.m
        arc['vStation_GCRF'] = np.vstack([leoVStation, earthVStation])*u.m/u.s
        arc['time'] = Time(np.hstack([leoTimes.mjd, earthTimes.mjd]), format='mjd')
        arc['sigma'] = np.ones(10, dtype=float)*u.arcsec

        # Initialize somewhere in the general neighborhood, but biased.
        # Probably the optimizer will remove the bias.
        initializer = ssapy.GaussianRVInitializer(
            r+0.3*RGEO, v+0.3*VGEO, rSigma=0.1*RGEO, vSigma=0.1*VGEO)
        # Use R and A priors
        priors = [ssapy.rvsampler.RPrior(RGEO, RGEO*0.2), ssapy.rvsampler.APrior(RGEO, RGEO*0.2)]

        prob = ssapy.RVProbability(arc, Time("J2000"), priors=priors)

        # fit = ssapy.LMOptimizer(prob, initializer(1)[0]).optimize()
        optimizer = ssapy.rvsampler.LeastSquaresOptimizer(
            prob, initializer(1)[0], ssapy.rvsampler.ParamOrbitRV)
        fit = optimizer.optimize()
        # reinitialize with the fit results
        initializer = ssapy.GaussianRVInitializer(
            fit[0:3], fit[3:6], rSigma=0.01*RGEO, vSigma=0.01*VGEO)

        # Need a proposal distribution
        # Start with a simple RVSigmaProposal, which draws r and v isotropically around current
        # value
        sampler = ssapy.MHSampler(prob, initializer, ssapy.RVSigmaProposal(RGEO*3e-5, VGEO*3e-5), nChain=4)
        chain, lnprob, lnprior = sampler.sample(nBurn=250, nStep=250)
        # Just check that things are generally improving over time...
        np.testing.assert_array_less(np.mean(lnprob[:10]), np.mean(lnprob[-10:]))
        print("Acceptance ratio: ", sampler.acceptanceRatio)

        # Now iterate while updating covariance of proposals
        for _ in range(1):
            cov = np.cov(chain.reshape(-1, 6), rowvar=False)
            cov += np.diag([1e5, 1e5, 1e5, 1e2, 1e2, 1e2])
            sampler.proposal = ssapy.MVNormalProposal(0.1*cov)
            chain, lnprob, lnprior = sampler.sample(nBurn=250, nStep=250)
            np.testing.assert_array_less(np.mean(lnprob[:10]), np.mean(lnprob[-10:]))
            print("Acceptance ratio: ", sampler.acceptanceRatio)

        rmean = np.mean(chain[...,0:3], axis=(0,1))
        vmean = np.mean(chain[...,3:6], axis=(0,1))

        print("r difference wrt GEO")
        print((rmean-r)/RGEO)
        print("v difference wrt GEO")
        print((vmean-v)/VGEO)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--prof", action='store_true')
    parser.add_argument("--prof_out", type=str, default=None)
    parser.add_argument("--prof_png", type=str, default=None)
    args = parser.parse_args()

    if args.prof:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

    test_prior()
    test_initializer()
    test_likelihood()
    test_emcee_sampler()
    test_mh_sampler()

    if args.prof:
        import pstats
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('cumtime')
        ps.print_stats(30)
        ps = pstats.Stats(pr).sort_stats('tottime')
        ps.print_stats(30)
        if args.prof_out:
            pr.dump_stats(args.prof_out)
            if args.prof_png:
                import subprocess
                cmd = "gprof2dot -f pstats {} -n1 -e1 | dot -Tpng -o {}".format(args.prof_out, args.prof_png)
                subprocess.run(cmd, shell=True)
