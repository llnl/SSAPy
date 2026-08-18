"""
Provides classes and functions for orbital tracking, hypothesis testing, angle wrapping, 
and fitting satellite observations to orbital models.
"""


import random
from functools import partial
import pdb
from astropy import units as u
from astropy.time import Time
from ssapy.constants import EARTH_MU
from ssapy.utils import norm, normSq, normed
from ssapy import rvsampler
import numpy as np
import ssapy
from .utils import lazy_property
from .constants import RGEO

keppropagator = ssapy.propagator.KeplerianPropagator()

priorvolume = dict(rv=(2*40*10**6)**3*(2*7000)**3*1.0,
                   # rough guess of total volume of prior for satellite
                   # parameter space (2*RGEO)**3*(2*v_LEO)**3 / m^3 / (m/s)^3

                   equinoctial=2*40*10**6*(4*np.pi)*np.pi*2*np.pi,
                   # equivalent rough guess in equinoctial space
                   # out to 2*RGEO in semimajor axis, 4pi for the
                   # plane of the orbit, pi for ex and ey, 2*pi for lambda.

                   angle=4*np.pi*np.pi*0.01**2*2*40*10**6*7000,
                   # very rough guess for angular space
                   # 4 pi for the position, 0.01 rad/s for the proper motions
                   # 2*RGEO for the distance, 7000 km/s for the range in
                   # allowed los velocities.  Vaguely the right ballpark?

                   )

__all__ = [
    "CircVelocityPrior",
    "GaussPrior",
    "Hypothesis",
    "MHT",
    "Track",
    "TrackBase",
    "TrackGauss",
    "VolumeDistancePrior",
    "ZeroRadialVelocityPrior",
    "combinatoric_lnprior",
    "data_for_satellite",
    "fit_arc",
    "fit_arc_blind",
    "fit_arc_blind_via_track",
    "fit_arc_with_gaussian_prior",
    "iterate_mht",
    "make_optimizer",
    "make_param_guess",
    "orbit_to_param",
    "param_to_orbit",
    "radeczn",
    "summarize_tracklet",
    "summarize_tracklets",
    "time_ordered_satIDs",
    "wrap_angle_difference",
]

# dead tracks: want to prune hypotheses that differ only by dead tracks.
# every time a track dies, do the N^2 search for equality among all of the
# hypotheses it includes?

class CircVelocityPrior:
    """Gaussian prior that log(GM/r/v^2) = 0.

    Parameters
    ----------
    sigma : float
        Prior standard deviation on log(GM/r/v^2), dimensionless.

    Attributes
    ----------
    sigma

    Methods
    -------
    __call__(orbit)
        Returns log prior probability of given orbit.
    """
    def __init__(self, sigma=np.log(2)/3.5):
        self.sigma = sigma

    def __call__(self, orbit, distance, chi=False):
        """Return log prior probability of given orbit.

        Parameters
        ----------
        orbit : Orbit
            The given orbit.
        chi : bool
            If True, return chi corresponding to log probability, rather than
            log probability.

        Returns
        -------
        logprior : float
            The log of the prior probability for given orbit.
        """
        loggmorv2 = np.log(EARTH_MU/norm(orbit.r)/normSq(orbit.v))
        chi0 = loggmorv2/self.sigma
        if chi:
            res = chi0
        else:
            res = -0.5*chi0**2
        return [res]


class ZeroRadialVelocityPrior:
    """Gaussian prior that ``v_R = 0``.

    Parameters
    ----------
    sigma : float
        Prior standard deviation on ``np.dot(v, r) / |v| / |r|``,
        dimensionless.

    Attributes
    ----------
    sigma

    Methods
    -------
    __call__(orbit)
        Returns log prior probability of given orbit.
    """
    def __init__(self, sigma=0.25):
        self.sigma = sigma

    def __call__(self, orbit, distance, chi=False):
        """Return log prior probability of given orbit.

        Parameters
        ----------
        orbit : Orbit
            The given orbit.
        chi : bool
            If True, return chi corresponding to log probability, rather than
            log probability.

        Returns
        -------
        logprior : float
            The log of the prior probability for given orbit.
        """
        vdotr = np.dot(normed(orbit.r), normed(orbit.v))
        chi0 = vdotr/self.sigma
        if chi:
            res = chi0
        else:
            res = -0.5*chi0**2
        return [res]


class GaussPrior:
    """Gaussian prior on parameters.

    Parameters
    ----------
    mu : array_like (6)
        Mean of prior
    cinvcholfac : array_like (6, 6)
        Inverse Cholesky factor for covariance matrix,
        chi = cinvcholfac*(param-mu)
    translator : function(orbit) -> param
        function that translates from orbits to parameters

    Attributes
    ----------
    mu, cinvcholfac, tranlator

    Methods
    -------
    __call__(orbit)
        Returns log prior probability of given orbit.
    """
    def __init__(self, mu, cinvcholfac, translator):
        self.mu = mu
        self.cinvcholfac = cinvcholfac
        self.translator = translator

    def __call__(self, orbit, distance=None, chi=False):
        """Return log prior probability of given orbit.

        Parameters
        ----------
        orbit : Orbit
            The given orbit.
        chi : bool
            If True, return chi corresponding to log probability, rather than
            log probability.

        Returns
        -------
        logprior : float
            The log of the prior probability for given orbit.
        """
        param = self.translator(orbit)
        chi0 = self.cinvcholfac.dot(param - self.mu)
        if chi:
            res = chi0
        else:
            res = -0.5*chi0**2
        return res


class VolumeDistancePrior:
    """Prior on range like r^2*exp(-r/scale), preferring larger distances where
    there is more volume.

    Methods
    -------
    __call__(orbit)
        Returns log prior probability of given orbit.
    """

    def __init__(self, scale=RGEO):
        self.scale = RGEO

    def __call__(self, orbit, distance, chi=False):
        """Return log prior probability of given orbit.

        Parameters
        ----------
        orbit : Orbit
            The given orbit.
        distance : float
            Distance between object and observer (m)
        chi : bool
            If True, return chi corresponding to log probability, rather than
            log probability.

        Returns
        -------
        logprior : float
            The log of the prior probability for given orbit.
        """
        distance = distance / self.scale
        lnp = 2*np.log(distance) - distance - (2*np.log(2) - 2 + 1e-9)
        # - np.log(2*self.scale) --- true normalization factor.
        # 2*np.log(2) - 2 --- fake normalization factor that takes
        # max(lnp) = 0, so more chi^2 like.
        if chi:
            res = np.sqrt(-2*lnp)
        else:
            res = lnp
        return res



def make_param_guess(rvguess, arc, mode='rv', orbitattr=None):
    """
    Generates an initial parameter guess for orbital estimation based on the provided mode, observation arc, and optional orbital attributes.

    Parameters:
    -----------
    rvguess : list or array-like
        Initial guess for the state vector (position and velocity). 
        Typically contains six elements: `[x, y, z, vx, vy, vz]`.
    arc : structured array
        Observation arc containing time and station data. Must include:
        - `'time'`: Observation times (assumed to have a `.gps` attribute).
        - `'rStation_GCRF'` and `'vStation_GCRF'`: Position and velocity of the station in the GCRF frame.
        - `'pmra'` (optional): Proper motion in right ascension (used to determine if proper motion data is available).
    mode : str, optional
        The mode for parameter generation. Must be one of:
        - `'rv'`: Radial velocity mode (default).
        - `'equinoctial'`: Equinoctial orbital elements mode.
        - `'angle'`: Angular observation mode.
    orbitattr : list of str, optional
        Additional orbital attributes to include in the parameter guess. Supported attributes include:
        - `'mass'`: Mass of the object.
        - `'area'`: Cross-sectional area.
        - `'cr'`: Radiation pressure coefficient.
        - `'cd'`: Drag coefficient.
        - `'log10area'`: Logarithm of the cross-sectional area.

    Returns:
    --------
    list
        A list of guessed parameters based on the specified mode:
        - For `'rv'`: `[rvguess, extraparam, epoch]`.
        - For `'equinoctial'`: `[equinoctialElements, extraparam, epoch]`.
        - For `'angle'`: `[radecrate, extraparam, epoch, initObsPos, initObsVel]`.

    Raises:
    -------
    ValueError
        If an unrecognized mode is provided.

    Notes:
    ------
    - The `epoch` is determined based on the observation times in the `arc`. If proper motion data (`'pmra'`) is available, the epoch is set to the first observation time. Otherwise, it is calculated as the midpoint between the first two observation times.
    - For `'equinoctial'` mode, the initial state vector (`rvguess`) is converted into equinoctial orbital elements using the `Orbit` class from `ssapy`.
    - For `'angle'` mode, the initial observation position and velocity are computed based on the station's GCRF data. If proper motion data is available, only the first observation is used; otherwise, the average of the first two observations is used.
    """
      
    s = np.argsort(arc['time'])
    usepm = 'pmra' in arc.dtype.names
    if usepm:
        epoch = arc['time'][s[0]].gps
    else:
        dt = arc['time'][s[1]]-arc['time'][s[0]]
        epoch = (arc['time'][s[0]]+dt/2).gps

    if orbitattr is not None:
        orbitpropguessdict = dict(mass=1, area=0, cr=1, cd=1, log10area=-1)
        extraparam = [orbitpropguessdict[x] for x in orbitattr]
    else:
        extraparam = []

    if mode == 'rv':
        return list(rvguess) + extraparam + [epoch]
    elif mode == 'equinoctial':
        orbInit = ssapy.orbit.Orbit(r=rvguess[:3], v=rvguess[3:6], t=epoch)
        return list(orbInit.equinoctialElements) + extraparam + [epoch]
    elif mode == 'angle':
        obs0 = arc[s[0]]
        if not usepm:
            obs1 = arc[s[1]]
            initObsPos = 0.5*(obs0['rStation_GCRF'].to(u.m).value +
                              obs1['rStation_GCRF'].to(u.m).value)
            initObsVel = 0.5*(obs0['vStation_GCRF'].to(u.m/u.s).value +
                              obs1['vStation_GCRF'].to(u.m/u.s).value)
        else:
            initObsPos = obs0['rStation_GCRF'].to(u.m).value
            initObsVel = obs0['vStation_GCRF'].to(u.m/u.s).value
        radecrate = ssapy.compute.rvObsToRaDecRate(rvguess[:3], rvguess[3:],
                                                 initObsPos, initObsVel)
        return (list(radecrate) + extraparam + [epoch] +
                list(initObsPos) + list(initObsVel))
    else:
        raise ValueError('unrecognized mode')

def make_optimizer(mode, param, lsq=False):
    """
    Creates and returns an optimizer object or function for orbital parameter estimation based on the specified mode and configuration.

    Parameters:
    -----------
    mode : str
        The optimization mode to use. Must be one of:
        - `'rv'`: Radial velocity-based optimization.
        - `'angle'`: Angular-based optimization.
        - `'equinoctial'`: Equinoctial orbital parameter optimization.
    param : list or array-like
        Initial parameters for the optimizer. 
        - For `'angle'` mode, the last six elements are expected to represent the initial position (last 6 to -3) and velocity (last 3).
    lsq : bool, optional
        If `True`, the optimizer will use a least-squares approach. Default is `False`.

    Returns:
    --------
    out : callable
        A callable optimizer object or function configured for the specified mode and parameters.

    Raises:
    -------
    ValueError
        If an unknown mode is provided.

    Notes:
    ------
    - When `lsq` is `True`, the function wraps the optimizer in a least-squares optimizer (`LeastSquaresOptimizer`) and uses the appropriate translator class based on the mode.
    - For `'angle'` mode, the function extracts the initial position and velocity from the `param` argument and passes them to the optimizer.
    - The function relies on the `rvsampler` module for optimizer classes and partial function creation.
    """

    if lsq:
        tdict = dict(rv=rvsampler.ParamOrbitRV,
                     angle=rvsampler.ParamOrbitAngle,
                     equinoctial=rvsampler.ParamOrbitEquinoctial)
    else:
        tdict = dict(rv=rvsampler.LMOptimizer,
                     angle=rvsampler.LMOptimizerAngular,
                     equinoctial=rvsampler.EquinoctialLMOptimizer)
    if mode not in tdict:
        raise ValueError('unknown mode %s!' % mode)
    out = tdict[mode]
    if mode == 'angle':
        initObsPos = param[-6:-3]
        initObsVel = param[-3:]
        out = partial(out, initObsPos=initObsPos, initObsVel=initObsVel)
    if lsq:
        out = partial(rvsampler.LeastSquaresOptimizer, translatorcls=out)
    return out


def fit_arc_blind(arc, verbose=False, mode='rv', priors=None, propagator=None,
                  damp=-1, orbitattr=None, optimizerkw={}, lsq=True,
                  factor=2, **kw):
    """Fit an orbit to an arc.

    The arc can be arbitrarily long.  Orbits are initially fit to the initial
    two observations and extended using a geomtrically increasing time baseline.
    If no new observations are inside of a given baseline, the next obs is added.
    The fit is least-squares, Levenberg-Marquardt, given specified priors and
    propagator. If the factor is set to 1, the orbit is built out iteratively
    by satID.

    Parameters
    ----------
    arc : array_like (n)
        numpy ndarray containing several fields necessary for describing
        observations.  rStation_GCRF (m), vStation_GCRF (m),
        time (astropy.time.Time or gps seconds), ra, dec,
        pmra, pmdec are all required fields.
    verbose : bool
        True for more verbose output
    mode : str
        Mode for fitting, must be one of 'rv', 'equinoctial', or 'angle'.
        In the
        first case, the parameters defining the orbit are taken to be
        the position and velocity of the object.  In the second case,
        the parameters are taken to be the equinoctial elements.  In the
        third case, the parameters are taken to be the angular position
        and proper motion of the object, together with its line-of-sight
        range and velocity.
    priors : list of objects implementing ssa Prior interface
        any priors to be applied in fitting
    propagator : ssapy.propagator.Propagator instance
        propagator to use in fitting
    damp : float
        damp chi residuals according to pseudo-Huber loss function.  Default
        of -1 means do not damp
    orbitattr : list(str)
        names of additional orbit propagation keywords to guess
    optimizerkw : dict
        any extra keywords to pass to the optimizer
    factor : float
        factor for geometrically increasing time baseline (default 2)
    kwargs : dict
        any extra keywords to pass to optimizer.optimize()
    """
    assert factor >= 1, "Geometric factor must be greater than or equal to 1"

    if len(arc) == 0:
        raise ValueError('len(arc) must be > 0')

    if priors is None:
        priors = [rvsampler.EquinoctialExEyPrior(0.3)]
    satids_ordered, times_ordered = time_ordered_satIDs(arc, with_time=True)
    paramInit, epoch = rvsampler.circular_guess(
        arc[np.isin(arc['satID'], satids_ordered[:2])])
    meanpm = 'pmra' in arc.dtype.names
    paramInit = make_param_guess(paramInit, arc,  mode=mode,
                                 orbitattr=orbitattr)
    nparam = 6 if orbitattr is None else 6 + len(orbitattr)
    if epoch.gps != paramInit[nparam]:
        raise ValueError('inconsistent epoch and guess?')

    optimizerclass = make_optimizer(mode, paramInit, lsq=lsq)

    if propagator is None:
        propagator = keppropagator

    groupind = [0]
    while groupind[-1] < len(times_ordered) - 1:
        dt = times_ordered[groupind[-1]] - times_ordered[0]
        newgroupind = np.max(np.flatnonzero(
                times_ordered <= times_ordered[0] + factor*dt))
        if newgroupind <= groupind[-1]:
            newgroupind = groupind[-1] + 1
        groupind.append(newgroupind)

    for ind in groupind:
        arc0 = arc[np.isin(arc['satID'], satids_ordered[:ind+1])]
        prob = rvsampler.RVProbability(
            arc0, epoch, priors=priors,
            propagator=propagator, meanpm=meanpm, damp=damp)
        optimizer = optimizerclass(prob, paramInit[:nparam],
                                   orbitattr=orbitattr, **optimizerkw)
        state = optimizer.optimize(**kw)
        paramInit = [x for x in state] + [x for x in paramInit[nparam:]]
        if getattr(optimizer.result, 'hithyperbolicorbit', False):
            print('hyperbolic orbit in blind')

    chi2 = np.sum(optimizer.result.residual**2)
    if getattr(optimizer.result, 'hithyperbolicorbit', False):
        print('hyperbolic orbit')
        chi2 += 1e9
    return (chi2, np.concatenate([state, paramInit[nparam:]]), optimizer.result)


def fit_arc(arc, guess,
            verbose=False, propagator=keppropagator,
            mode='rv', priors=None, damp=-1, optimizerkw={},
            lsq=True, orbitattr=None, **kw):
    """Fit an orbit to an arc.

    See documentation for fit_arc_blind.  fit_arc implements the same
    interface, but takes a additional guess & epoch arguments.  These specify
    the initial guess in appropriate units given the chosen mode.  epoch
    specifies the time at which this initial guess applies.

    Parameters
    ----------
    arc : array_like (n)
        numpy ndarray containing several fields necessary for describing
        observations.  rStation_GCRF (m), vStation_GCRF (m),
        time (astropy.time.Time or gps seconds), ra, dec,
        pmra, pmdec are all required fields.
    guess : array_like (n_par)
        parameters for initial guess
        guess[nparam] should be the epoch in GPS
        guess[nparam:] should be extra parameters not optimized over
    verbose : bool
        True for more verbose output
    propagator : ssapy.propagator.Propagator instance
        propagator to use in fitting
    mode : str
        Mode for fitting, must be one of 'rv', 'equinoctial', or 'angle'.
        In the
        first case, the parameters defining the orbit are taken to be
        the position and velocity of the object.  In the second case,
        the parameters are taken to be the equinoctial elements.  In the
        third case, the parameters are taken to be the angular position
        and proper motion of the object, together with its line-of-sight
        range and velocity.
    priors : list of objects implementing ssa Prior interface
        any priors to be applied in fitting
    damp : float
        Damping used in pseudo-Huber loss function.  Default of -1 means no
        damping.
    orbitattr : list(str)
        Names of orbit attributes used for propagation (mass, area, ...)
    optimizerkw : dict
        any extra keywords to pass to the optimizer
    kwargs : dict
        any extra keywords to pass to optimizer.optimize()
    """
    if priors is None:
        priors = [rvsampler.EquinoctialExEyPrior(0.3)]
    paramInit = guess
    meanpm = 'pmra' in arc.dtype.names
    nparam = 6 + (0 if orbitattr is None else len(orbitattr))
    epoch = Time(paramInit[nparam], format='gps')
    prob = rvsampler.RVProbability(arc, epoch, priors=priors,
                                   propagator=propagator, meanpm=meanpm,
                                   damp=damp)
    optimizerclass = make_optimizer(mode, paramInit, lsq=lsq)
    optimizer = optimizerclass(prob, paramInit[:nparam], orbitattr=orbitattr,
                               **optimizerkw)
    state = optimizer.optimize(**kw)
    chi2 = np.sum(optimizer.result.residual**2)
    if getattr(optimizer.result, 'hithyperbolicorbit', False):
        chi2 += 1e9
    return (chi2, np.concatenate([state, paramInit[nparam:]]),
            optimizer.result)


def fit_arc_with_gaussian_prior(arc, mu, cinvcholfac, verbose=False,
                                propagator=keppropagator, mode='rv',
                                lsq=True,
                                optimizerkw={}, orbitattr=None,
                                **kw):
    """Fit an orbit to an arc.

    See documentation for fit_arc_blind.  fit_arc_with_gaussian_prior
    implements the same interface, but takes an additional mu, cinvcholfac,
    and epoch arguments.  These specify a gaussian prior on the parameters.
    The mean and covariance of the Gaussian are given by mu and cinvcholfac,
    the inverse Cholesky matrix corresponding to the covariance matrix.  The
    initial guess is taken to be the mean of the Gaussian prior.  Presently,
    any other priors are assumed to have been folded into the Gaussian.
    epoch specifies the time at which Gaussian prior parameters are
    applicable.  For greatest speed, the arc should contain a single
    observation and the epoch should be equal to the time of that
    observation; then no propagations are needed in this function.

    Parameters
    ----------
    arc : array_like (n)
        numpy ndarray containing several fields necessary for describing
        observations.  rStation_GCRF (m), vStation_GCRF (m),
        time (astropy.time.Time or gps seconds), ra, dec,
        pmra, pmdec are all required fields.
    mu : array_like (n_par)
        parameters for initial guess
    cinvcholfac : array_like (n_par, n_par)
        inverse cholesky matrix for covariance matrix
    verbose : bool
        True for more verbose output
    propagator : ssapy.propagator.Propagator instance
        propagator to use in fitting
    mode : str
        Mode for fitting, must be one of 'rv', 'equinoctial', or 'angle'.
        In the
        first case, the parameters defining the orbit are taken to be
        the position and velocity of the object.  In the second case,
        the parameters are taken to be the equinoctial elements.  In the
        third case, the parameters are taken to be the angular position
        and proper motion of the object, together with its line-of-sight
        range and velocity.
    orbitattr : list(str)
        list of strings of names of additional propagation attributes to
        fit (mass, area, cr, cd, ...)
    optimizerkw : dict
        any extra keywords to pass to the optimizer
    kwargs : dict
        any extra keywords to pass to optimizer.optimize()
    """
    nparam = 6 + (0 if orbitattr is None else len(orbitattr))
    translator = partial(orbit_to_param, mode=mode, orbitattr=orbitattr, fitonly=True)
    if mode == 'angle':
        translator = partial(translator, rStation=mu[-6:-3], vStation=mu[-3:])
    priors = [GaussPrior(mu[:nparam], cinvcholfac, translator=translator)]

    paramInit = mu
    meanpm = 'pmra' in arc.dtype.names
    epoch = Time(paramInit[nparam], format='gps')
    prob = rvsampler.RVProbability(arc, epoch, priors=priors,
                                   propagator=propagator, meanpm=meanpm)
    optimizerclass = make_optimizer(mode, paramInit, lsq=lsq)
    optimizer = optimizerclass(prob, paramInit[:nparam], orbitattr=orbitattr,
                               **optimizerkw)
    state = optimizer.optimize(**kw)
    chi2 = np.sum(optimizer.result.residual**2)
    if getattr(optimizer.result, 'hithyperbolicorbit', False):
        chi2 += 1e9
    return (chi2, np.concatenate([state, paramInit[nparam:]]),
            optimizer.result)


def data_for_satellite(data, satIDList):
    """Extract data for specific satID.

    This helper routine takes a set of observations containing a column
    'satID', and returns the subset of those observations for which
    'satID' is in the list satIDList.

    Parameters
    ----------
    data : array_like (n), including column 'satID'
        data from which to select matching rows
    satIDList : list
        list of satIDs to select from data
    """
    m = np.zeros(len(data), dtype='bool')
    for satID in satIDList:
        if satID == -1:
            continue
        m0 = data['satID'] == satID
        if np.sum(m0) == 0:
            raise ValueError('satID %d not found!' % satID)
        m |= m0
    return data[m]


def wrap_angle_difference(dl, wrap, center=0.5):
    """
    Wraps an angle difference to a specified range.

    This function adjusts a given angle difference (`dl`) to lie within a range defined by `wrap` and centered around `center * wrap`.

    Parameters:
    -----------
    dl : float or array-like
        The angle difference to be wrapped.
    wrap : float
        The wrap range. Typically represents the full range of the angle (e.g., `2 * pi` for radians or `360` for degrees).
    center : float, optional
        The center of the wrapping range, expressed as a fraction of `wrap`. Default is `0.5` (centered around `wrap / 2`).

    Returns:
    --------
    float or array-like
        The wrapped angle difference, adjusted to lie within the range `[center * wrap - wrap, center * wrap]`.
    """

    return ((dl + center*wrap) % wrap)-center*wrap


def radeczn(orbit, arc, **kw):
    """Compute right ascension, declination, range, rates, and anomaly wraps.

    Parameters
    ----------
    orbit : object or list of objects
        Orbital object(s) containing orbital parameters. Must include
        ``meanMotion`` and ``t``. May include ``r`` to determine scalar or
        array-like orbit handling.
    arc : structured array
        Observation arc containing ``time``, ``rStation_GCRF``, and
        ``vStation_GCRF`` fields. May include ``satID`` to determine scalar or
        array-like arc handling.
    kwargs : dict, optional
        Additional keyword arguments passed to :func:`ssapy.compute.radec`.

    Returns
    -------
    tuple
        Tuple containing ``rr``, ``dd``, ``zz``, ``pmrr``, ``pmdd``, ``dzzdt``,
        and ``nwrap`` arrays.
    """
    
    rr, dd, zz, pmrr, pmdd, dzzdt = ssapy.compute.radec(
        orbit, arc['time'],
        obsPos=arc['rStation_GCRF'].to(u.m).value,
        obsVel=arc['vStation_GCRF'].to(u.m/u.s).value,
        rate=True, **kw)
    if not hasattr(orbit, 'meanMotion'):
        nwrap = [o.meanMotion*(arc['time'].gps-o.t) for o in orbit]
    else:
        scalarorbit = (orbit.r.ndim == 1)
        try:
            _ = len(arc['satID'])
            scalararc = False
        except TypeError:
            scalararc = True
        meanMotion = orbit.meanMotion
        tt = arc['time'].gps
        to = orbit.t
        if not scalararc and not scalarorbit:
            meanMotion = meanMotion.reshape(-1, 1)
            tt = tt.reshape(1, -1)
            to = orbit.t.reshape(-1, 1)
        nwrap = meanMotion*(tt-to)
    return rr, dd, zz, pmrr, pmdd, dzzdt, nwrap


def param_to_orbit(param, mode='rv', orbitattr=None):
    """Convert parameters to Orbits.

    Note: Orbits are converted so that they are all bound orbits to
    avoid challenges with hyperbolic orbit propagation.

    Parameters
    ----------
    param : array_like (n, n_param)
        parameters for orbits
    mode : str
        parameter type, one of 'rv', 'angle', 'orbit'
    """
    param = np.atleast_1d(param)
    if len(param.shape) == 1:
        param = param[None, :]
        squeeze = True
    else:
        squeeze = False
    nparam = 6
    if orbitattr is not None:
        nparam += len(orbitattr)
    else:
        orbitattr = []
    tt = Time(param[:, nparam], format='gps')

    propkw = dict()
    for name, val in zip(orbitattr, param[:, 6:nparam].T):
        if name.startswith('log10'):
            name = name[5:]
            val = 10.**val
        propkw[name] = val

    if mode == 'rv':
        orbit = ssapy.orbit.Orbit(r=param[:, :3], v=param[:, 3:6], t=tt,
                                propkw=propkw)
    elif mode == 'equinoctial':
        aa = np.clip(param[:, 0], 1, np.inf)
        orbit = ssapy.orbit.Orbit.fromEquinoctialElements(
            aa, param[:, 1], param[:, 2],
            param[:, 3], param[:, 4], param[:, 5],
            t=tt, propkw=propkw)
    elif mode == 'angle':
        rv = ssapy.compute.radecRateObsToRV(
            param[:, 0], param[:, 1], param[:, 2],
            param[:, 3], param[:, 4], param[:, 5],
            param[:, -6:-3], param[:, -3:])
        orbit = ssapy.orbit.Orbit(r=rv[0], v=rv[1], t=tt, propkw=propkw)
    else:
        raise ValueError('unknown mode')

    vmax = np.sqrt(2*EARTH_MU/norm(orbit.r))
    m = norm(orbit.v) > vmax
    if np.any(m):
        vnew = orbit.v.copy()
        vnew[m, :] = vnew[m, :]*vmax[m, None]/norm(vnew[m, :])[:, None]*0.999
        orbit = ssapy.Orbit(r=orbit.r, v=vnew, t=orbit.t, propkw=propkw)

    if squeeze:
        orbit = orbit[0]


    return orbit


def orbit_to_param(orbit, mode='rv', orbitattr=None,
                   rStation=None, vStation=None, fitonly=False):
    """Convert Orbits to parameters.

    Note: Orbits are converted so that they are all bound orbits to
    avoid challenges with hyperbolic orbit propagation.

    Parameters
    ----------
    param : array_like (n, n_param)
        parameters for orbits
    mode : str
        parameter type, one of 'rv', 'angle', 'orbit'
    """
    if orbit.r.ndim == 1:
        squeeze = True
        orbit = ssapy.Orbit(
            orbit.r[None, :], orbit.v[None, :], np.atleast_1d(orbit.t),
            propkw={k: np.atleast_1d(v) for k, v in orbit.propkw.items()})
    else:
        squeeze = False

    vmax = np.sqrt(2*EARTH_MU/norm(orbit.r))
    m = norm(orbit.v) > vmax
    if np.any(m):
        vnew = orbit.v.copy()
        vnew[m, :] = vnew[m, :]*vmax[m, None]/norm(vnew[m, :])[:, None]*0.999
        orbit = ssapy.Orbit(r=orbit.r, v=vnew, t=orbit.t, propkw=orbit.propkw)

    tt = orbit.t[:, None]
    if orbitattr is None:
        orbitattr = list()
    orbitparam = []
    for n in orbitattr:
        if n.startswith('log10'):
            val = np.log10(orbit.propkw[n[5:]])
        else:
            val = orbit.propkw[n]
        orbitparam.append(val[:, None])

    if mode == 'rv':
        param = np.hstack([orbit.r, orbit.v] + orbitparam + [tt])
    elif mode == 'equinoctial':
        eqel = [e[:, None] for e in orbit.equinoctialElements]
        param = np.hstack([*eqel] + orbitparam + [tt])
    elif mode == 'angle':
        rdrate = ssapy.compute.rvObsToRaDecRate(
            orbit.r, orbit.v, rStation, vStation)
        rdrate = [r[:, None] for r in rdrate]
        rStation = np.tile(rStation, len(orbit)).reshape(len(orbit), 3)
        vStation = np.tile(vStation, len(orbit)).reshape(len(orbit), 3)
        param = np.hstack([*rdrate] + orbitparam + [tt] +
                          [rStation, vStation])
    else:
        raise ValueError('unknown mode')

    if fitonly:
        nparam = 6 + len(orbitattr)
        param = param[:, :nparam]

    if squeeze:
        param = param[0]

    return param


class TrackBase:
    """
    Set of observations fit together as an orbital track.

    This class represents a set of observations that are being modeled
    as the motion of a single object through space.

    Parameters
    ----------
    satIDs : array_like
        list of detection IDs considered part of this track
    data : array_like
        observations that could be part of this track.  Observations that are
        actually part of this track are specified in satIDs.
        data is must contain a number of fields used in fitting, like
        ra, dec, obsPos, obsVel, and time.
    volume : float
        the volume in 6-D orbital element space in which objects could be
        found, needed to normalize false-positive vs. new satellite odds
        default to a very rough number only applicable for RV parameterization
    propagator : ssa Propagator instance
        the propagator to use when propagating the orbit.  Default to None
        (Keplerian)

    Attributes
    ----------
    satIDs : array_like, int
        the detection IDs that are part of this Track
    data : array_like
        observations that could be part of this track; see constructor
    volume : float
        the prior volume in 6-D parameter where satellites may be observed
    times : array_like, float
        the GPS times at which this object was observed
    mode : str
        the type of parameterization to use to fit the track
        one of 'rv', 'angle', 'equinoctial'
    lnprob : float
        the log probability of this association of observations to an orbit
        contains contributions both from priors and likelihood.
    """
    # new_track_lnpenalty = -13  # is this valuable?!

    def __init__(self, satIDs, data, volume=None, mode='rv',
                 propagator=None, orbitattr=None):
        self.satIDs = list(satIDs)
        self.data = data
        self.propagator = (propagator
                           if propagator is not None
                           else ssapy.propagator.KeplerianPropagator())
        if volume is not None:
            self.volume = volume
        else:
            self.volume = priorvolume[mode]
        self.times = data_for_satellite(data, satIDs)['time'].gps
        self.mode = mode
        self.orbitattr = orbitattr

    @lazy_property
    def lnprob(self):
        lnprob = -np.log(self.volume)  # prior term
        if np.any(~np.isfinite(self.covar)):
            lnprob += 0
            # i.e., we localized to 1 m^3 x 1 (m/s)^3...
            # for ~70 obs, localization is usually ~(100 m)^3*(0.01 m/s)^3
            # (at least formally!)
            # we still need to think more about treating the fact that the
            # errors should be more like ~40" in position
            # maybe translating from 3 position measurements to a
            # position measurement (with 40" uncertainties) and a velocity
            # measurement (with formal uncertainties only?)
        else:
            # SVD seems more robust than direct np.linalg.det
            _, ss, _ = np.linalg.svd(2*np.pi*self.covar)
            determinant = np.prod(ss)
            if determinant <= 0:
                determinant = 1
                if self.chi2 < 1e8:
                    print('Weird determinant!', self.satIDs)
                    # pdb.set_trace()
            lnprob += 0.5*np.log(determinant)
        # rarely, the fit screws up, and gets something with a huge covariance
        # in all cases I've encountered, due to hitting the hyperbolic orbit
        # boundary.  We adjust the covariance volume here to be the same as
        # the prior volume, and add a huge number to the chi2 in
        # fit_arc.
        if lnprob > 0:
            lnprob = -np.log(self.volume)
            if self.chi2 < 1e8:
                print('warning, huge covariance', self.satIDs)
        # normalizing factor
        # if len(self.satIDs) == 1:
        #     self.initial_lnprob = lnprob
        # lnprob -= self.initial_lnprob
        # lnprob += TrackBase.new_track_lnpenalty
        lnprob += -self.chi2/2
        return lnprob

    def predict(self, arc0, return_sigma=False, return_nwrap=True):
        """Predict the location and uncertainty of this object at future times.

        Parameters
        ----------
        arc0 : array_like, structure array
            must contain fields:
            time: future times (GPS) (float)
            rStation_GCRF: GCRF position of observer at future time (3, float)
            vStation_GCRF: GCRF velocity of observer at future time (3, float)
        return_sigma : bool
            if True, also return the sigma points used to compute the
            future covariance
        return_nwrap : bool
            if True, also return the number of orbital wrappings of the
            sigma points.

        Returns
        -------
        mean, covar, fsigma
        mean : array_like (4, n)
            ra, dec, dra/dt*cos(dec), ddec/dt at future times (rad, rad/s)
        covar : array_like (4, 4, n)
            covariance of mean parameters at future times
        fsigma : array_like (4, 13, n)
            13 sigma points used to compute covar
            fsigma[:, 0, :] is identical to mean.
        """
        nparam = 6
        if self.orbitattr is not None:
            nparam += len(self.orbitattr)
        fixed_dimensions = np.arange(len(self.param)) >= nparam
        fsigma = ssapy.utils.sigma_points(
            partial(self.propagaterdz, arc0=arc0, return_nwrap=return_nwrap),
            self.param, self.covar, fixed_dimensions=fixed_dimensions)
        mean = fsigma[:, 0, ...]
        dmean = fsigma[:, 1:, ...]-mean[:, None, ...]
        dmean[0, ...] = ((dmean[0, ...]+np.pi) % (2*np.pi))-np.pi
        if len(dmean.shape) == 2:
            covar = np.cov(dmean, ddof=0)
        else:
            covar = np.array([np.cov(dmean[:, :, i], ddof=0)
                              for i in range(dmean.shape[-1])])
        # next step here would be to do some tangent
        # plane projections (stereoscopic?) so that we don't have
        # issues at the pole or with wraps in ra.
        res = mean, covar
        if return_sigma:
            res = res + (fsigma,)
        return res

    def gate(self, arc, return_nwrap=False):
        """Compute chi^2 that this track could be associated with arc.

        Parameters
        ----------
        arc : array_like, structure array
            must contain fields:
            time: future time (GPS) (float)
            rStation_GCRF: GCRF position of observer at future time (3, float)
            vStation_GCRF: GCRF velocity of observer at future time (3, float)
            ra: observed ra (float)
            pmra: observed proper motion in ra (float)
            dec: observed dec (float)
            pmdec: observed proper motion in dec (float)
            dra: uncertainty in ra (float)
            ddec: uncertainty in dec (float)
            dpmra: uncertainty in proper motion in ra (float)
            dpmdec uncertainty in proper motion in dec (float)

        return_nwrap : bool
            if True, also return the standard deviation of the number of
            orbits the object could have made at the time of observation.

        Returns
        -------
        chi2, nwrapsig
        chi2 : float
            The chi-squared implied by associating this observation with this
            track.
        nwrapsig : float
            if return_nwrap, the standard deviation of the number of orbits the
            object could have made at the time of observation
        """
        arc0 = arc[0]  # just take the first tracklet at the moment...
        mean, covar = self.predict(arc0, return_nwrap=True)
        mean = mean[:-1]  # drop nwrap
        covar, nwrapcovar = covar[:-1, :-1], covar[-1, -1]
        cd = np.cos(arc0['dec'].to(u.rad).value)
        covar[:, 0] *= cd
        covar[0, :] *= cd
        covarobs = np.diag([arc0['dra'].to(u.rad).value**2,
                            arc0['ddec'].to(u.rad).value**2,
                            arc0['dpmra'].to(u.rad/u.s).value**2,
                            arc0['dpmdec'].to(u.rad/u.s).value**2])
        covar = covar + covarobs
        area = np.linalg.det(2*np.pi*covar[0:2, 0:2])**0.5
        if (area <= 0) or (area > np.pi/8):
            covar = np.diag([np.inf, np.inf, np.inf, np.inf])
        try:
            cinv = np.linalg.inv(covar)
        except np.linalg.LinAlgError:
            cinv = 1e-30
        measvec = np.array([arc0['ra'].to(u.rad).value,
                            arc0['dec'].to(u.rad).value,
                            arc0['pmra'].to(u.rad/u.s).value,
                            arc0['pmdec'].to(u.rad/u.s).value])
        dvec = measvec - mean
        dvec[0] = (((dvec[0] + np.pi) % (2*np.pi)) - np.pi)*cd
        chi2 = np.dot(dvec, cinv).dot(dvec)
        # we should choose a space other than ra/dec, where things
        # are coming closer to following lines, and so are more Gaussian.
        # note however: satellites are not following great circles,
        # so even in gnomonic projection about central point, our
        # points won't quite follow a straight line.
        # but it should at least be much better than whatever we're doing now!
        # or stereoscopic, with NP centered at the observation?  Great circles
        # through projection point are lines.
        # simpler: line up track orbital plane with projection equator.
        res = chi2
        if return_nwrap:
            res = (res, np.sqrt(nwrapcovar))
        return res

    def propagaterdz(self, param, arc0=None, return_nwrap=False):
        """Propagate param to times and locations specified in arc0.

        Parameters
        ----------
        param : array_like
            parameters corresponding to self.mode for orbit to propagate
        arc0 : arary_like, must contain observation fields
            array containing fields specifying times and observer locations
            to which track fit should be propagated.
        return_nwrap : bool
            if True, additionally return nwrap

        Returns
        -------
        array([rr, pmrr, dd, pmdd, nwrap])
        rr : ra at future times, radians
        dd : dec at future times, radians
        pmrr : proper motion in ra at future times, rad / s
        pmdd : proper motion in dec at future times, rad / s
        nwrap : number of orbits completed.  Only provided if return_nwrap
            is True.
        """
        orb = param_to_orbit(param, mode=self.mode, orbitattr=self.orbitattr)
        rr, dd, zz, pmrr, pmdd, dzzdt, nwrap = radeczn(
            orb, arc0, propagator=self.propagator)
        res = [rr, dd, pmrr, pmdd]
        if return_nwrap:
            res = res + [nwrap]
        return np.array(res)

    def update(self, time, rStation=None, vStation=None):
        """Shift this track to a new time.

        Only implemented for numerical propagators where time-shifting can provide a big
        speed up.
        """
        return

    def __repr__(self):
        return (('Track, chi2: %.2e, lnprob: %.2e, tracklets:' %
                 (self.chi2, self.lnprob)) + str(self.satIDs))


class Track(TrackBase):
    """Set of observations to be fit as an object moving through space.

    Subclasses TrackBase, which could have other implementations
    (for example, a subclass that doesn't fit the whole track simultaneously,
    but instead approximates past information with a single Gaussian
    prior on the parameters).

    Parameters
    ----------
    satIDs : see TrackBase
    data : see TrackBase
    guess : array_like, float
        guess of initial parameters fitting orbit
    initial_lnprob : float
        adjust probability of this track by initial_lnprob (default: 0)
    priors : rvsampler.Prior instance
        list of priors to use when fitting this track
    mode : str
        fitting mode to use.  One of 'rv', 'angle', or 'equinoctial'
        parameters have different meanings in these modes
    propagator : Propagator instance
        ssa Propagator instance to use for propagation.  Default None (Keplerian)
    """
    def __init__(self, satIDs, data, guess=None, initial_lnprob=0,
                 priors=None, mode='rv', propagator=None, orbitattr=None):
        super().__init__(satIDs, data, mode=mode, propagator=propagator,
                         orbitattr=orbitattr)
        self.initial_lnprob = initial_lnprob
        if guess is not None:
            chi2, param, res = fit_arc(
                data_for_satellite(data, self.satIDs), guess,
                maxfev=100, mode=mode,
                priors=priors, propagator=self.propagator,
                orbitattr=orbitattr)
        else:
            chi2, param, res = fit_arc_blind(
                data_for_satellite(data, self.satIDs), maxfev=100, mode=mode,
                priors=priors, propagator=self.propagator,
                orbitattr=orbitattr)
        self.chi2 = chi2
        self.param = param
        nparam = 6 if orbitattr is None else 6 + len(orbitattr)
        self.covar = getattr(res, 'covar', np.full((nparam, nparam), np.inf))
        self.priors = priors
        self.success = getattr(res, 'success', False)

    def gaussian_approximation(self, propagator=None):
        """Get Gaussian approximation of this track.

        Returns
        -------
        TrackGauss, giving Gaussian approximation of this track.
        """
        # determinant = np.sqrt(np.linalg.det(2*np.pi*self.covar))
        # this is supposed to be a volume
        # when it's small enough, or when it's "close enough" to actually being
        # Gaussian, we want to make a Gaussian approximation.
        # but we really want, e.g., a fractional uncertainty in a.
        # I don't really know how to do this.
        # let's just say that we need at least four observations.
        # can do some tests with real tracks.
        if len(self.satIDs) >= 4:
            prop = propagator if propagator is not None else self.propagator
            if prop != self.propagator:
                # refitting with a better propagator
                t = Track(self.satIDs, self.data, guess=self.param,
                          initial_lnprob=self.initial_lnprob,
                          priors=self.priors, mode=self.mode,
                          propagator=prop, orbitattr=self.orbitattr)
                self.param = t.param
                self.covar = t.covar
                self.chi2 = t.chi2
            return TrackGauss(self.satIDs, self.data, self.param, self.covar,
                              self.chi2, mode=self.mode, propagator=prop,
                              orbitattr=self.orbitattr)
        else:
            # don't bother approximating if the track is too short.
            return self
        # maybe we could just always approximate the track?
        # since the initial track is "on epoch", i.e., not propagated,
        # it's going to be some ~cone in xyz, vxvyvz.
        # that's not _great_ to approximate as a Gaussian, but maybe not
        # horrible?

    def addto(self, satid):
        """Add a detection to this track.

        Parameters
        ----------
        satID : int or list(int)
            the detection ID of the observation to add to this track

        Returns
        -------
        A new Track, including the additional observation.
        """
        newsatidlist = satid if isinstance(satid, list) else [satid]
        return Track(self.satIDs+newsatidlist, self.data, guess=self.param,
                     initial_lnprob=self.initial_lnprob,
                     priors=self.priors, mode=self.mode,
                     propagator=self.propagator, orbitattr=self.orbitattr)


class TrackGauss(TrackBase):
    """Set of observations to be fit as an object moving through space.

    Subclasses TrackBase.  This implementation does not fit the entire
    track simultaneously, but instead approximates past tracklets with
    a single Gaussian prior on the parameters.  This information is then
    updated with each new observation in a Kalman-filter approach.

    Parameters
    ----------
    satIDs : see TrackBase
    data : see TrackBase
    param : array_like, float
        mean of Gaussian
    covar : array_like, float
        covariance of Gaussian
    chi2 : float
        chi2 of fit at center of Gaussian
    mode : str
        fitting mode to use.  One of 'rv', 'angle', or 'equinoctial'
        parameters have different meanings in these modes
    propagator : Propagator instance
        ssa Propagator instance to use for propagation.  Default None
        (Keplerian)
    priors : list of Priors instances
        priors used in initially fitting this orbit.  Their influence
        should now be completely recorded in param + covar, but they are
        recorded here for reference.
    """
    def __init__(self, satIDs, data, param, covar, chi2,
                 mode='rv', propagator=None, sigma_points=None,
                 priors=None, orbitattr=None):
        super().__init__(satIDs, data, mode=mode, propagator=propagator,
                         orbitattr=orbitattr)
        self.param = param
        self.covar = covar
        self.chi2 = chi2
        if np.any(~np.isfinite(covar)):
            finitecovar = np.diag(np.ones(covar.shape[0], dtype='f4'))
            self.chi2 += 1e9
        else:
            finitecovar = self.covar
        self.cinvcholfac = np.linalg.inv(np.linalg.cholesky(finitecovar))
        nparam = 6 if orbitattr is None else 6 + len(orbitattr)
        if sigma_points is None:
            fixed = np.arange(len(param)) >= nparam
            sigma_points = ssapy.utils.sigma_points(None, param, finitecovar,
                                                  fixed_dimensions=fixed)
        self.sigma_points = sigma_points
        self.priors = priors

    def gaussian_approximation(self, propagator):
        """Get Gaussian approximation of this track.

        Returns
        -------
        self.  No op; this track already is a Gaussian approximation.
        """
        if propagator != self.propagator:
            raise NotImplementedError
        return self


    def update(self, t, rStation=None, vStation=None):
        """Move this track to a different epoch.

        Does not bother updating if t is different from the current
        track time by less than one microsecond.


        Parameters
        ----------
        t : astropy.time.Time
           epoch for new TrackGauss
        rStation : array_like (3)
           position of station at new time (m, GCRF); required if
           mode == 'angle'
        vStation : array_like (3)
           velocity of station at new time (m, GCRF); required if
           mode == 'angle'
        """

        nparam = 6 if self.orbitattr is None else 6 + len(self.orbitattr)
        oldepoch = Time(self.param[nparam], format='gps')
        if np.abs((t-oldepoch).to(u.s).value) < 1e-6:
            # don't bother propagating to a new time for times less than one
            # microsecond.
            return
        orb = param_to_orbit(self.sigma_points, mode=self.mode,
                             orbitattr=self.orbitattr)
        neworb = orb.at(t, propagator=self.propagator)
        if isinstance(rStation, u.Quantity):
            rStation = rStation.to(u.m).value
            vStation = vStation.to(u.m/u.s).value
        new_sigma = orbit_to_param(neworb, mode=self.mode,
                                   rStation=rStation, vStation=vStation,
                                   orbitattr=self.orbitattr)
        new_param = new_sigma[0]
        new_covar = np.cov((new_sigma[1:, :nparam]-new_param[:nparam]).T,
                           ddof=0)
        self.param = new_param
        self.covar = new_covar
        self.cinvcholfac = np.linalg.inv(np.linalg.cholesky(self.covar))
        self.sigma_points = new_sigma

    def at(self, t, rStation=None, vStation=None):
        """Get Gaussian approximation of this track at different epoch.

        This just copies the track and then updates it to be at the new time.


        Parameters
        ----------
        t : astropy.time.Time
           epoch for new TrackGauss
        rStation : array_like (3)
           position of station at new time (m, GCRF); required if
           mode == 'angle'
        vStation : array_like (3)
           velocity of station at new time (m, GCRF); required if
           mode == 'angle'

        Returns
        -------
        Gaussian-approximated track (TrackGauss) at new time.
        """

        newtrack = TrackGauss(self.satIDs, self.data, self.param,
                              self.covar, self.chi2, mode=self.mode,
                              propagator=self.propagator,
                              sigma_points=self.sigma_points,
                              orbitattr=self.orbitattr)
        newtrack.update(t, rStation=rStation, vStation=vStation)
        return newtrack

    def addto(self, satid):
        """Add a detection to this track.

        This also shifts the track's epoch to the epoch of the added
        observation.

        Parameters
        ----------
        satID : int
            the detection ID of the observation to add to this track

        Returns
        -------
        A new TrackGauss, including the additional observation.
        """
        newsatidlist = satid if isinstance(satid, list) else [satid]
        arc = data_for_satellite(self.data, newsatidlist)
        nparam = 6 if self.orbitattr is None else 6 + len(self.orbitattr)
        mu = self.param[:nparam]
        oldepoch = Time(self.param[nparam], format='gps')
        ind = np.flatnonzero(arc['satID'] == newsatidlist[0])
        lastind = ind[np.argmax(arc['time'][ind].gps)]
        newepoch = arc['time'][lastind]
        if newepoch.gps != oldepoch.gps:
            newtrack = self.at(newepoch,
                               rStation=arc['rStation_GCRF'][lastind],
                               vStation=arc['vStation_GCRF'][lastind])
            param = newtrack.param
            cinvcholfac = newtrack.cinvcholfac
        else:
            param = self.param
            cinvcholfac = self.cinvcholfac
        chi2new, paramnew, resnew = fit_arc_with_gaussian_prior(
            arc, param, cinvcholfac, mode=self.mode,
            propagator=self.propagator, orbitattr=self.orbitattr)
        covarnew = getattr(resnew, 'covar', np.full((nparam, nparam), np.inf))
        if np.any(~np.isfinite(covarnew)):
            covarnew = np.full((nparam, nparam), np.inf)
        return TrackGauss(self.satIDs+newsatidlist, self.data,
                          paramnew, covarnew,
                          self.chi2+chi2new, mode=self.mode,
                          propagator=self.propagator,
                          orbitattr=self.orbitattr)

def fit_arc_blind_via_track(arc, propagator=None, verbose=False,
                            reset_if_too_uncertain=False, mode='rv',
                            priors=None, order='forward',
                            approximate=False, orbitattr=None, factor=1):
    """Fit a series of detections using the Track interface.

    Parameters
    ----------
    arc : ndarry with many fields
        The observations to fit.  Must contain many fields (ra, dec, ...)
    propagator : Propagator instance
        Propagator to use for orbit propagation.  Default None (Keplerian).
    verbose : bool
        print additional information during fitting if True
    reset_if_too_uncertain : bool
        if the uncertainty in the number of wraps grows to be greater than
        pi, start track fitting over, ignoring old tracklets.
    factor : float
        factor for geometrically increasing time baseline (default 1)
    order : str
        order to process observations; default 'forward'.  'backward'
        processes observations in reverse chronological order.

    Returns
    -------
    tracklist
    A list of Tracks, giving the fits to the track as each observation
    in added sequentially in time to the fit.
    """
    assert factor >= 1, "Geometric factor must be greater than or equal to 1"

    satids_ordered, times_ordered = time_ordered_satIDs(arc, with_time=True,
                                                        order=order)
    tracklist = []
    prop = None if approximate else propagator
    track0 = Track([satids_ordered[0]], arc, propagator=prop, mode=mode,
                   priors=priors, orbitattr=orbitattr)
    tracklist.append(track0)

    resetnwrap = np.pi / 3

    groupind = [0]
    while groupind[-1] < len(times_ordered) - 1:
        dt = times_ordered[groupind[-1]] - times_ordered[0]
        newgroupind = np.max(np.flatnonzero(
                times_ordered <= times_ordered[0] + factor * dt))
        if newgroupind <= groupind[-1]:
            newgroupind = groupind[-1] + 1
        groupind.append(newgroupind)

    for i in range(1, len(groupind)):
        arc0 = arc[np.isin(arc['satID'], satids_ordered[groupind[i]])]
        chi2, nwrapsig = tracklist[-1].gate(arc0[0:1], return_nwrap=True)
        reset = ((reset_if_too_uncertain and (nwrapsig > resetnwrap)) or
                 (tracklist[-1].chi2 >= 1e9))
        if verbose:
            if reset:
                resetstr = 'resetting'
            else:
                resetstr = ''
            dt = (times_ordered[groupind[i]] - times_ordered[groupind[i-1]])
            print(f'{chi2:5.1f} {nwrapsig:8.4f} dt:{dt/60/60/24:5.2f} {resetstr}')
        if reset:
            track = Track(arc0['satID'], arc, propagator=prop,
                          mode=mode, priors=priors, orbitattr=orbitattr)
        else:
            newsatidlist = satids_ordered[groupind[i-1]+1:groupind[i]+1]
            track = tracklist[-1].addto(newsatidlist)
            if approximate:
                track = track.gaussian_approximation(propagator)
        tracklist.append(track)

    return tracklist


def combinatoric_lnprior(nsat, ntrack, ndet):
    """Prior probability of assigning detections to tracks of satellites.

    This function intends to give the prior probability that ndet detections
    of nsat total satellites correspond to ntrack satellites.

    Parameters
    ----------
    nsat : int
        number of satellites that could possibly be observed
    ntrack : int
        number of tracks in current solution
    ndet : int
        number of detections of these satellites
    """
    # I believe the desired factor is:
    # 1/N_sat^(N_detections - 1) * (N_sat - 1)! / (N_sat - N_objects)!
    from scipy import special
    lnprior = -(ndet - 1)*np.log(nsat)
    lnprior += special.gammaln(nsat-1)-special.gammaln(nsat-ntrack)
    # may have to think about an approximation to N_sat!/(N_sat - N_objects)!
    # that doesn't rely on as much cancellation as the above...
    return lnprior


def time_ordered_satIDs(data, with_time=False, order='forward'):
    """Give satIDs in data in order of observation.
    Optionally includes first times for each ID.

    Parameters
    ----------
    data : array_like
        the data to consider
    with_time : bool
        Option to also output list of corresponding gps times
    order : 'forward' or 'backward'

    Returns
    -------
    list of satIDs in data, ordered by observation time

    list of corresponding gps times of first obs per ID
    """
    s = np.argsort(data['time'].gps)
    if order == 'backward':
        s = s[::-1]
    usedtracklet = {}
    out = []
    times = []
    for satid in data['satID'][s]:
        if usedtracklet.get(satid, False):
            continue
        usedtracklet[satid] = True
        out.append(satid)

        if with_time:
            times.append(data[data['satID'] == satid]['time'].gps[0])

    res = out
    if with_time:
        res = (res,) + (times,)
    return res


class Hypothesis:
    """Assignment of detections to multiple tracks.

    This class represents an assignment of detections to multiple tracks.  The
    goal of Multiple Hypothesis Tracking to to find the Hypothesis (track
    assignment) that has the highest likelihood.

    Parameters
    ----------
    tracks : list of Tracks
        The tracks in the hypothesis.  Each track must correspond to
        different observations, and all observations under consideration
        must be accounted for

    Attributes
    ----------
    tracks : list
        the tracks in this hypothesis
    lnprob : float
        the log probability of this hypothesis
    nsat : int
        the total number of satellites in the sky that could be observed
    """
    def __init__(self, tracks, nsat=1000):
        self.tracks = tracks
        self.lnprob = np.sum([t.lnprob for t in tracks])
        ndet = self.ntracklet()
        self.lnprob += combinatoric_lnprior(nsat, len(tracks), ndet)
        # not clear this has value.
        self.nsat = nsat

    def __repr__(self):
        return self.summarize()

    def summarize(self, verbose=False):
        """Summarize the Hypothesis as a string.

        Parameters
        ----------
        verbose : bool
            Include information about each track in the hypothesis.

        Returns
        -------
        A string summarizing the hypothesis.
        """
        out = 'Hypothesis with %d tracks:' % len(self.tracks)
        out += ', lnprob: %7.4e' % self.lnprob
        if verbose:
            out += '\n'
            for track in self.tracks:
                out += str(track)+'\n'
            out += '\n'
        return out


    def difference(self, hypothesis):
        """Print the difference between two hypotheses.

        Often two hypotheses are very similar.  This makes it easier to
        inspect only the differences between them.

        Parameters
        ----------
        hypothesis : Hypothesis
            hypothesis to which to compare this hypothesis
        """
        matchinother = np.zeros(len(hypothesis.tracks), dtype='bool')
        matchinsame = np.zeros(len(self.tracks), dtype='bool')
        for i, t in enumerate(self.tracks):
            try:
                ind = hypothesis.tracks.index(t)
                matchinsame[i] = True
                matchinother[ind] = True
            except ValueError:
                continue
        print('dlnprob: %5.3f, %d tracks in common' %
              (self.lnprob-hypothesis.lnprob, sum(matchinsame)))
        print('Tracks only in 1:')
        print(self.summarize(verbose=False))
        for i in np.flatnonzero(~matchinsame):
            print(self.tracks[i])
        print('Tracks only in 2:')
        print(hypothesis.summarize(verbose=False))
        for i in np.flatnonzero(~matchinother):
            print(hypothesis.tracks[i])

    def ntracklet(self):
        """Number of observations in tracks in the hypothesis."""
        n = 0
        for t in self.tracks:
            n += len(t.satIDs)
        return n

    @staticmethod
    def addto(hypothesis, track, oldtrack=None, **kw):
        """Add a new track to a Hypothesis.

        Parameters
        ----------
        hypothesis : Hypothesis
            hypothesis to which to add
        track : Track
            track to add to hypothesis
        oldtrack : Track
            track to remove from hypothesis, if the new track updates the
            old track.

        Returns
        -------
        new Hypothesis, with track added, replacing oldtrack if oldtrack is
        not None.
        """
        newtracks = [t for t in hypothesis.tracks]
        if oldtrack is not None and oldtrack != []:
            oldtrackind = hypothesis.tracks.index(oldtrack)
            newtracks[oldtrackind] = track
        else:
            newtracks.append(track)
        return Hypothesis(newtracks, nsat=hypothesis.nsat, **kw)


class MHT:
    """Multiple Hypothesis Tracking of many hypotheses explaining data.

    This class manages the assessment of which hypothesis are most likely
    to explain data, building a set of tracks observation by observation.

    Initialize the MHT with the data, do mht.run(), and inspect the most
    likely hypotheses using [h.lnprob for h in mht.hypotheses].

    Parameters
    ----------
    data : array_like[N]
        data to model.  Must contain angles of observations, observer
        locations and velocities, times, detection IDs, etc.
    nsat : int
        total number of satellites in the sky.  Affects the overall
        prior and the preference for new tracks vs. additional assignments
        to existing tracks.
    truth : dict
        when true assignments are known, this argument can provide debug
        information about when the tracker has lost the true assignment
    hypotheses : list of Hypothesis
        initialize the MHT with this existing list of Hypotheses.
        Default None.
    propagator : instance of ssa Propagator
        propagator to use.  Default to None (Keplerian)
    fitonly : ndarray[N] bool
        entries in data to fit
    """
    def __init__(self, data, nsat=1000, truth=None, hypotheses=None,
                 propagator=None, mode='rv', fitonly=None,
                 approximate=False, orbitattr=None, priors=None):
        self.data = data.copy()
        if fitonly is None:
            fitonly = np.ones(len(data), dtype='bool')
        self.satids = time_ordered_satIDs(data[fitonly])
        self.nsat = nsat
        self.truth = truth
        self.track2hyp = {}
        self.mode = mode
        self.nfit = 0
        self.orbitattr = orbitattr
        self.priors = priors
        self.propagator = (propagator
                           if propagator is not None
                           else ssapy.propagator.KeplerianPropagator())
        self._newly_dead_tracks = []
        self.approximate = approximate
        if hypotheses is None:
            self.hypotheses = [Hypothesis([], nsat=self.nsat)]
        else:
            for hyp in hypotheses:
                for t in hyp.tracks:
                    if t not in self.track2hyp:
                        self.track2hyp[t] = []
                    self.track2hyp[t].append(hyp)
            self.hypotheses = hypotheses

    def run(self, first=None, last=None, verbose=False, order='forward',
            **kw):
        """Run the MHT analysis.

        Parameters
        ----------
        first : int
            First tracklet to consider
        last : int
            Last tracklet to consider
        verbose : bool
            print extra progress information while running
        kwargs : dict
            extra keyword arguments to pass to MHT.prune.
        """
        tsatid = self.satids
        if first is None:
            first = 0
        if last is None:
            last = len(tsatid)
        tsatid = tsatid[first:last]
        if order == 'backward':
            tsatid = tsatid[::-1]
        for i, satid in enumerate(tsatid):
            if verbose:
                ndead = sum([getattr(t, 'dead', False)
                             for t in self.track2hyp])
                print(('Tracklet %5d of %5d, %5d live tracks '
                       '(%5d dead), %4d fit, %5d hypotheses') %
                      (first+i, len(self.satids), len(self.track2hyp)-ndead,
                       ndead, self.nfit, len(self.hypotheses)))
            self.add_tracklet(satid)
            self.prune(satid, **kw)

    def add_tracklet(self, satid, **kw):
        """Add an observation from data to the MHT analysis.

        Parameters
        ----------
        satid : int
            the ID of the observation to add.
        """
        newhypotheses = []
        tracklist = list(self.track2hyp.keys())
        newtrack2hyp = {t:[] for t in tracklist}
        self.nfit = 0
        tooshorttogate = 0
        self._newly_dead_tracks = []
        arc = data_for_satellite(self.data, [satid])
        if (len(arc) < 1) and (
            (('dpmra' not in arc.dtype.names) or
             (np.isfinite(arc['dpmra'][0])))):
            print('Just one observation, skipping tracklet %d' % satid)
            # used to match these, but not generate new tracks for these.
            # this is problematic since then hypotheses aren't directly
            # comparable to one another; they don't explain the same
            # numbers of tracks
            return

        for track in tracklist:
            if arc['time'].gps[0] in track.times:
                continue
            if getattr(track, 'dead', False):
                continue
            track.update(arc['time'][0],
                         rStation=arc['rStation_GCRF'][0],
                         vStation=arc['vStation_GCRF'][0])
            gatechi2, nwrapsig = track.gate(arc, return_nwrap=True)
            outofgatechi2 = 8**2
            if self.truth is not None:
                goodtrack = np.all([self.truth[s] == self.truth[satid]
                                    for s in track.satIDs])
                if goodtrack and gatechi2 > outofgatechi2:
                    print('warning, excluding real track by gate', gatechi2)
                    # pdb.set_trace()
                elif goodtrack:
                    # print('real gate chi2: %5.2f' % gatechi2)
                    pass
            if gatechi2 > outofgatechi2:
                # if it's hard to add this tracklet to this track, don't
                # bother.
                continue
            if nwrapsig > np.pi/3:
                # at this point, there's real danger of fitting the wrong
                # number of orbits to this point.  Don't do that; kill the
                # track instead, and hope we get better sampling of it later.
                # we could do better explicitly trying multiple sets of orbits
                # there's currently an assumption that a track is uniquely
                # defined by a set of satID; this would require opening another
                # dimension, or, better, playing some games with chi2 and covar
                # in the track, letting those try to track the multimodal
                # posteriors.
                track.dead = True
                self._newly_dead_tracks.append(track)
                continue
            if gatechi2 < 1e-9:
                tooshorttogate += 1
                # pdb.set_trace()
            if gatechi2 == 0.:
                # pdb.set_trace()
                pass
            newtrack = track.addto(satid)
            self.nfit += 1
            if newtrack.chi2 - track.chi2 > outofgatechi2:
                continue
            if self.approximate:
                newtrack = newtrack.gaussian_approximation(self.propagator)
            if (gatechi2 > 1e-9) & (len(track.satIDs) >= 2):
                # print(track.satIDs, '%5.1f %5.1f %5.1f %5.1f %5.1f' % (gatechi2, track.chi2, track.lnprob(),
                #       newtrack.chi2, newtrack.lnprob()))
                # if (newtrack.chi2 > 1e9) & (len(track.satIDs) >= 3):
                #     pdb.set_trace()
                pass
            newtrack2hyp[newtrack] = []
            hypaffected = self.track2hyp[track]
            for hypothesis in hypaffected:
                newhypothesis = Hypothesis.addto(hypothesis, newtrack, track,
                                                 **kw)
                newhypotheses.append(newhypothesis)
                for t0 in newhypothesis.tracks:
                    newtrack2hyp[t0].append(newhypothesis)

        prop = None if self.approximate else self.propagator
        newtrack = Track([satid], self.data, mode=self.mode,
                         propagator=prop, orbitattr=self.orbitattr,
                         priors=self.priors)
        for h in self.hypotheses:
            newhypothesis = Hypothesis.addto(h, newtrack, **kw)
            if newhypothesis.ntracklet() != h.ntracklet()+1:
                pdb.set_trace()
            newhypotheses.append(newhypothesis)
            for t in newhypothesis.tracks:
                if t not in newtrack2hyp:
                    newtrack2hyp[t] = []
                newtrack2hyp[t].append(newhypothesis)
        # if self.flag_inconsistency(newtrack2hyp, newhypotheses):
        #     pdb.set_trace()
        self.hypotheses = newhypotheses
        self.track2hyp = newtrack2hyp

    def prune_tracks(self, satid, nconfirm=6):
        """Prune tracks that are now nearly identical from the MHT analysis.

        As the MHT adds observations to tracks, eventually some tracks
        have a number of confirming observations.  For these long tracks,
        we really only need to keep different Tracks and Hypotheses for cases
        where there are recent disagreements with how those tracks continue;
        if hypotheses agree that the track continues in the same way,
        we don't need to continue tracking the past differences.  So
        we prune out cases of former disagreement.

        This tries to identify such overlapping tracks and keep only one of
        them.

        Parameters
        ----------
        satid : int
            the satID of the most-recently added detection to the MHT
        nconfirm : int
            if two tracks are identical in their last nconfirm observations
            delete the tracks not in the most likely hypothesis.

        Returns
        -------
        boolean array indicating hypotheses to keep
        """
        lnprob = np.array([hyp.lnprob for hyp in self.hypotheses])
        s = np.argsort(-lnprob)  # best first
        besthyp = self.hypotheses[s[0]]
        lengthenedtrack = np.flatnonzero([satid in track.satIDs
                                          for track in besthyp.tracks])
        if len(lengthenedtrack) != 1:
            pdb.set_trace()
        lengthenedtrack = besthyp.tracks[lengthenedtrack[0]]
        nsatid = len(lengthenedtrack.satIDs)
        keephyp = np.ones(len(self.hypotheses), dtype='bool')
        if nsatid <= nconfirm+1:
            # no op; no sufficiently long tracks for this to be meaningful.
            return keephyp
        keeptogether = lengthenedtrack.satIDs[:nsatid-nconfirm]
        # relies on ordering of self.track2hyp being fixed...
        deltrack = []
        ndel = 0
        for i, track in enumerate(self.track2hyp):
            ntogether = 0
            for sid in keeptogether:
                ntogether += (sid in track.satIDs)
            if (ntogether != 0) and (ntogether != len(keeptogether)):
                deltrack.append(track)
        for track in deltrack:
            for hyp in self.track2hyp[track]:
                hyp.deleteme = True
                ndel += 1
        keephyp = ~np.array([getattr(hyp, 'deleteme', False)
                             for hyp in self.hypotheses])
        if ndel > 0:
            print('%d hypotheses pruned with differences more than %d frames ago.' %
                  (ndel, nconfirm))
            # import pdb
            # pdb.set_trace()
        return keephyp

    def prune_stale_hypotheses(self, newdeadtracks):
        """Prune hypotheses that are different from better hypotheses only in
        dead tracks.

        Note: implentation relies on identical tracks always having the same
        id.  The MHT code tries to guarantee this---otherwise it must do
        extra work matching identical tracks to new detections, etc.  But
        using id(Track) feels horrible to me.

        Parameters
        ----------
        newdeadtracks : list of Track
            list of tracks that died since the last pruning.

        Returns
        -------
        boolean mask indicating hypotheses to keep
        """
        if len(newdeadtracks) == 0:
            return np.ones(len(self.hypotheses), dtype='bool')
        hyps = list(set(sum([self.track2hyp[t] for t in newdeadtracks], [])))
        ntrack = [len(h.tracks) for h in hyps]
        maxntrack = max(ntrack)
        idarr = np.zeros((maxntrack, len(hyps)), dtype=np.intp) - 1
        for i, hyp in enumerate(hyps):
            idarr[:ntrack[i], i] = [
                id(t) if not getattr(t, 'dead', False) else -1
                for t in hyp.tracks]
        idarr = np.sort(idarr, axis=0)
        lnprob = [h.lnprob for h in hyps]
        s = np.lexsort([lnprob] + [idarr[i] for i in range(len(idarr))])
        todelete = 0
        for i in range(1, len(hyps)):
            if np.all(idarr[:, s[i]] == idarr[:, s[i-1]]):
                hyps[s[i-1]].deleteme = True
                todelete += 1
        keephyp = ~np.array([getattr(hyp, 'deleteme', False)
                             for hyp in self.hypotheses])
        if todelete > 0:
            print('%d hypotheses pruned, differing only by dead tracks.' %
                  todelete)
        return keephyp


    def prune(self, satid, nkeepmax=10000, pkeep=1e-9,
              keeponlytrue=False, nconfirm=6):
        """Prune unlikely hypotheses from the MHT.

        Parameters
        ----------
        satid : int
            the detection ID most recently added to the MHT
        nkeepmax : int
            keep no more than nkeepmax hypotheses
        pkeep : float
            keep no hypotheses more than pkeep times less likely than the
            most likely hypothesis
        keeponlytrue : bool
            keep only the single hypothesis known to be true for speed
            comparison purposes.
        nconfirm : int
            only keep one variant of tracks that agree in the last nconfirm
            detections
        """
        lnprob = np.array([hyp.lnprob for hyp in self.hypotheses])
        s = np.argsort(-lnprob)  # best first
        keep = np.zeros(len(self.hypotheses), dtype='bool')
        bestprob = lnprob[s[0]]
        # keep ones with high enough probability relative to best
        keep[lnprob > bestprob + np.log(pkeep)] = 1
        # don't keep more than nkeepmax
        keep[s[nkeepmax:]] = 0
        if nconfirm > 0:
            keep = keep & self.prune_tracks(satid,
                                            nconfirm=nconfirm)
        keep = keep & self.prune_stale_hypotheses(self._newly_dead_tracks)
        if np.sum(keep) == 0:
            raise ValueError('should not be possible!')

        # keep at least nkeepmin
        # keep[s[:nkeepmin]] = 1
        if self.truth is not None:
            tracksat = [[[self.truth[s] for s in t.satIDs] for t in h.tracks]
                        for h in self.hypotheses]
            nomixup = [[all(s == t[0] for s in t) for t in h] for h in tracksat]
            nomixup = np.array([all(h) for h in nomixup])
            split = np.array(
                [len(set(t[0] for t in h)) != len(h) for h in tracksat])
            truthind = np.argmax(~split & nomixup)
            truthisalive = nomixup & ~split & keep
            if np.any(nomixup & ~split):
                print('truth: dlnprob: %5.3f, nhyp: %d' %
                      (bestprob-lnprob[truthind],
                       np.sum(lnprob > lnprob[truthind])))
            if np.sum(truthisalive) != 1:
                print('warning: true solution is no longer included.')
            if keeponlytrue:
                keep = keep & nomixup & ~split

        # print(len(self.hypotheses), np.sum(keep))
        self.hypotheses = [self.hypotheses[i] for i in np.flatnonzero(keep)]
        live = {h: True for h in self.hypotheses}
        for track in self.track2hyp:
            self.track2hyp[track] = (
                [hyp0 for hyp0 in self.track2hyp[track]
                 if live.get(hyp0, False)])
        for track in list(self.track2hyp):
            if len(self.track2hyp[track]) == 0:
                del self.track2hyp[track]
        # if self.flag_inconsistency(self.track2hyp, self.hypotheses):
        #     pdb.set_trace()

    @staticmethod
    def flag_inconsistency(track2hyp, hyp):
        """Debug method checking to see if there are any inconsistencies
        between the tracks tracked by the MHT and the hypotheses tracked
        by the MHT.  Should only be needed for debugging.

        Every hypothesis in track2hyp[track] should include the Track track.
        Every hypothesis in hyp should be in the list of track2hyp[track] for
        each of its tracks.
        Every track should be in a hypothesis.

        Parameters
        ----------
        track2hyp : dict[Track -> list(Hypothesis)]
        hyp: list(Hypothesis)

        Returns
        -------
        True if everything is consistent.
        """
        consistent = 0
        for t in track2hyp:
            for h in track2hyp[t]:
                if t not in h.tracks:
                    consistent |= 1
            if len(track2hyp[t]) == 0:
                consistent |= 8
        for h in hyp:
            for t in h.tracks:
                if h not in track2hyp[t]:
                    consistent |= 2
        ntracklet = np.array([sum([len(t.satIDs) for t in h.tracks], 0) for h in hyp])
        if len(ntracklet) > 0 and np.any(ntracklet[0] != ntracklet):
            consistent |= 4
        if consistent != 0:
            pdb.set_trace()
        return consistent


def summarize_tracklets(data, posuncfloor=None, pmuncfloor=None):
    """Add fields to default data set to accommodate MHT fitting.

    Parameters
    ----------
    data : array_like
        data to add fields to
    posuncfloor : float or None
        add a positional uncertainty floor to the ra/dec uncertainties (deg)
    pmuncfloor : float or None
        add a proper motion uncertainty floor to the ra/dec proper motion
        uncertainties (deg/s)

    Returns
    -------
    array parallel to data with additional fields dra, ddec,
    pmra, pmdec, dpmra, dpmdec
    """
    datao = data.copy()
    data = data.copy()
    s = np.lexsort([data['time'], data['satID']])
    usat, first = np.unique(data['satID'][s], return_index=True)
    last = np.concatenate([first[1:], [len(data)]])
    newfields = [('dra', u.deg, None),
                 ('ddec', u.deg, None),
                 ('pmra', u.deg/u.s, None),
                 ('pmdec', u.deg/u.s, None),
                 ('dpmra', u.deg/u.s, None),
                 ('dpmdec', u.deg/u.s, None),
                 ('t_baseline', u.s, None)]
    for field, unit, _ in newfields:
        dtype = 'f8' if unit is not None else 'object'
        data[field] = np.zeros(len(data), dtype=dtype)*unit
    for f, l in zip(first, last):
        meanpos, dmeanpos, pm, dpm = summarize_tracklet(data[s[f:l]])
        r0 = np.mean(data['rStation_GCRF'][s[f:l], :], axis=0)
        dr0 = data['rStation_GCRF'][s[l-1]]-data['rStation_GCRF'][s[f]]
        dt0 = data['time'][s[l-1]]-data['time'][s[f]]
        v0 = (dr0 / dt0).to(u.m/u.s)
        data['rStation_GCRF'][s[f:l]] = r0
        if np.any(~np.isfinite(data['vStation_GCRF'][s[f:l]])):
            data['vStation_GCRF'][s[f:l]] = v0
            # this isn't quite the same as taking the mean velocity over
            # the observations.  But on Aero-B it makes a maximum difference
            # of less than one part in a million, which seems negligible
            # for _velocity_ measurements.  Note that for _positions_ this
            # would be significant!
        else:
            data['vStation_GCRF'][s[f:l]] = (
                np.mean(data['vStation_GCRF'][s[f:l], :], axis=0)[None, :])
        data['time'][s[f:l]] = Time(
            np.mean(data['time'][s[f:l]].gps), format='gps')
        data['ra'][s[f:l]] = meanpos[0]
        data['dec'][s[f:l]] = meanpos[1]
        data['dra'][s[f:l]] = dmeanpos[0]
        data['ddec'][s[f:l]] = dmeanpos[1]
        data['pmra'][s[f:l]] = pm[0]
        data['pmdec'][s[f:l]] = pm[1]
        data['dpmra'][s[f:l]] = dpm[0]
        data['dpmdec'][s[f:l]] = dpm[1]
        data['t_baseline'][s[f:]]= dt0.to(u.s)
    if posuncfloor is not None:
        data['dra'] = np.sqrt(posuncfloor**2+data['dra']**2)
        data['ddec'] = np.sqrt(posuncfloor**2+data['ddec']**2)
    if pmuncfloor is not None:
        data['dpmra'] = np.sqrt(pmuncfloor**2+data['dpmra']**2)
        data['dpmdec'] = np.sqrt(pmuncfloor**2+data['dpmdec']**2)
    data = data[s[first]]
    s = np.argsort(data['time'].gps)
    return data[s]


def summarize_tracklet(arc):
    """Compute mean ra/dec, uncertainty in mean, proper motion, uncertainty
    for a short arc.

    Parameters
    ----------
    arc : the short arc of detections

    Returns
    -------
    [(ra, dec), (dra, ddec), (pmra, pmdec), (dpmra, dpmdec)]
    ra, dec: the mean position in right ascension and declination
    dra, ddec: the uncertainty in the mean RA and declination
    pmra, pmdec: the proper motion in right ascension and declination
    dpmra, dpmdec: the uncertainty in the proper motion in RA and declination
    """
    unit = ssapy.utils.lb_to_unit(arc['ra'].to(u.rad).value,
                               arc['dec'].to(u.rad).value)
    meanunit = np.mean(unit, axis=0)
    # defines tangent plane
    pole = np.array([0, 0, 1])
    raunit = normed(np.cross(pole, meanunit))
    decunit = normed(np.cross(meanunit, raunit))
    ratp = np.einsum('ij,j', unit, raunit)
    dectp = np.einsum('ij,j', unit, decunit)
    if len(unit) <= 1:
        return ((arc['ra'], arc['dec']),
                (arc['sigma'], arc['sigma']),
                (0., 0.),
                (np.inf, np.inf))
    meantime = np.mean(arc['time'].gps)
    dt = arc['time'].gps-meantime
    dpos = arc['sigma'].to(u.rad).value
    rap, racovar = np.polyfit(dt, ratp, 1, w=1./dpos, cov='unscaled')
    rasig = np.sqrt(np.diag(racovar))
    decp, deccovar = np.polyfit(dt, dectp, 1, w=1./dpos, cov='unscaled')
    decsig = np.sqrt(np.diag(deccovar))
    meanra, meandec = ssapy.utils.unit_to_lb(meanunit[None, :])
    rap[1] += meanra[0]
    decp[1] += meandec[0]
    out = [(rap[1]*u.rad, decp[1]*u.rad),
           (rasig[1]*u.rad, decsig[1]*u.rad),
           (rap[0]*u.rad/u.s, decp[0]*u.rad/u.s),
           (rasig[0]*u.rad/u.s, decsig[0]*u.rad/u.s)]
    return out


# need to write code that ~iterates MHT to try to bring in detections that weren't
# matched.

def iterate_mht(data, oldmht, nminlength=20, trimends=2, **kw):
    """
    Iterates and refines the Multiple Hypothesis Tracking (MHT) process by updating tracks and generating new hypotheses.

    Parameters:
    -----------
    data : dict
        A dictionary containing satellite data. Must include a key `'satID'` with satellite IDs.
    oldmht : MHT
        The previous MHT object containing hypotheses and tracking information.
    nminlength : int, optional
        Minimum length of satellite tracks to be included in the new hypotheses. Tracks shorter than this length are excluded. Default is 20.
    trimends : int, optional
        Number of observations to trim from both ends of each track. This helps refine the tracks by removing edge data. Default is 2.
    kwargs : dict
        Additional keyword arguments passed to the `MHT.run()` method.

    Returns:
    --------
    newmht : MHT
        A new MHT object with updated hypotheses and refined tracks.

    Notes:
    ------
    - The function identifies the best hypothesis from the `oldmht` object based on the highest log probability (`lnprob`).
    - Tracks are filtered based on their length (`nminlength`) and whether they are marked as "dead."
    - If `trimends` is greater than 0, the ends of each track are trimmed, and new track objects are created.
    - The function creates an initial hypothesis using the refined tracks and generates a new MHT object.
    - Satellite IDs used in the updated tracks are excluded from the fitting process for the new MHT object.
    - The `newmht.run()` method is executed with the provided keyword arguments.
    """

    bestind = np.argmax([h.lnprob for h in oldmht.hypotheses])
    besthyp = oldmht.hypotheses[bestind]
    tracks = [t for t in besthyp.tracks
              if not getattr(t, 'dead', False) and (len(t.satIDs) > nminlength)]
    # want to cut off the ends of the tracks
    if trimends > 0:
        newtracks = []
        for i, t in enumerate(tracks):
            newt = Track(t.satIDs[trimends:-trimends], t.data,
                         guess=t.param, priors=t.priors, mode=t.mode,
                         propagator=t.propagator, orbitattr=t.orbitattr)
            print(i, len(tracks), newt.chi2/4/len(newt.satIDs))
            if oldmht.approximate:
                newt = newt.gaussian_approximation()
            newtracks.append(newt)
        tracks = newtracks
    initialhyp = Hypothesis(tracks, nsat=besthyp.nsat)
    usedsatid = set(sum([t.satIDs for t in tracks], []))
    keepdata = np.array([s not in usedsatid for s in data['satID']])
    newmht = MHT(data, hypotheses=[initialhyp], nsat=oldmht.nsat,
                 propagator=oldmht.propagator, mode=oldmht.mode,
                 fitonly=keepdata, approximate=oldmht.approximate,
                 priors=oldmht.priors, truth=oldmht.truth,
                 orbitattr=oldmht.orbitattr)
    newmht.run(**kw)
    return newmht
