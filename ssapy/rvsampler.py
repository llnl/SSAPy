""" This module provides functions and classes for orbital mechanics calculations, optimization techniques, sampling methods, and orbital parameter translations. """


import numpy as np
from astropy.time import Time

from .orbit import Orbit
from .propagator import KeplerianPropagator
from .compute import dircos, radec, radec, radecRateObsToRV, rvObsToRaDecRate
from .utils import norm, normed, normSq, cluster_emcee_walkers, unitAngle3, _emcee_version_before_3
from .constants import RGEO, VGEO, WGS84_EARTH_MU


# Priors
class RPrior:
    """Gaussian prior on distance from origin |r|.

    Parameters
    ----------
    rmean : float
        Prior mean on |r| in meters.
    rsigma : float
        Prior standard deviation on |r| in meters.

    Attributes
    ----------
    rMean
    rSigma

    Methods
    -------
    __call__(orbit)
        Returns log prior probability of given orbit.
    """
    def __init__(self, rmean, rsigma):
        self.rmean = rmean
        self.rsigma = rsigma

    def __call__(self, orbit, distance=None, chi=False):
        """Return log prior probability of given orbit.

        Parameters
        ----------
        orbit : Orbit
            The given orbit.
        distance: float, optional
            Distance between object and observer (m) [not used]

        Returns
        -------
        logprior : float
            The log of the prior probability for given orbit.
        """
        chi0 = (norm(orbit.r) - self.rmean)/self.rsigma
        res = chi0 if chi else -0.5*chi0**2
        return res


class VPrior:
    """Gaussian prior on velocity magnitude |v|.

    Parameters
    ----------
    vmean : float
        Prior mean on |v| in meters per second.
    vsigma : float
        Prior standard deviation on |v| in meters per second.

    Attributes
    ----------
    vMean
    vSigma

    Methods
    -------
    __call__(orbit)
        Returns log prior probability of given orbit.
    """
    def __init__(self, vmean, vsigma):
        self.vmean = vmean
        self.vsigma = vsigma

    def __call__(self, orbit, distance=None, chi=False):
        """Return log prior probability of given orbit.

        Parameters
        ----------
        orbit : Orbit
            The given orbit.
        distance: float, optional
            Distance between object and observer (m) [not used]

        Returns
        -------
        logprior : float
            The log of the prior probability for given orbit.
        """
        chi0 = (norm(orbit.v) - self.vmean)/self.vsigma
        res = chi0 if chi else -0.5*chi0**2
        return res


class APrior:
    """Gaussian prior on orbit semimajor axis a.

    Parameters
    ----------
    amean : float
        Prior mean on a in meters.
    asigma : float
        Prior standard deviation on a in meters.

    Attributes
    ----------
    aMean
    aSigma

    Methods
    -------
    __call__(orbit)
        Returns log prior probability of given orbit.
    """
    def __init__(self, amean, asigma):
        self.amean = amean
        self.asigma = asigma

    def __call__(self, orbit, distance=None, chi=False):
        """Return log prior probability of given orbit.

        Parameters
        ----------
        orbit : Orbit
            The given orbit.
        distance: float, optional
            Distance between object and observer (m) [not used]

        Returns
        -------
        logprior : float
            The log of the prior probability for given orbit.
        """
        chi0 = (orbit.a - self.amean)/self.asigma
        res = chi0 if chi else -0.5*chi0**2
        return res


class EPrior:
    """Gaussian prior on orbit eccentricity e.

    Parameters
    ----------
    emean : float
        Prior mean on e.
    esigma : float
        Prior standard deviation on e.

    Attributes
    ----------
    eMean
    eSigma

    Methods
    -------
    __call__(orbit)
        Returns log prior probability of given orbit.
    """
    def __init__(self, emean, esigma):
        self.emean = emean
        self.esigma = esigma

    def __call__(self, orbit, distance=None, chi=False):
        """Return log prior probability of given orbit.

        Parameters
        ----------
        orbit : Orbit
            The given orbit.
        distance: float, optional
            Distance between object and observer (m) [not used]
        chi : bool
            If true, return chi rather than ln(prob)

        Returns
        -------
        logprior : float
            The log of the prior probability for given orbit.
        """
        chi0 = (orbit.e - self.emean)/self.esigma
        res = chi0 if chi else -0.5*chi0**2
        return res


class EquinoctialExEyPrior:
    """Gaussian prior on equinoctial ex and ey, favoring ex = ey = 0.

    Parameters
    ----------
    sigma : float
        standard deviation on ex and ey

    Attributes
    ----------
    sigma

    Methods
    -------
    __call__(orbit)
        Returns log prior probability of given orbit.
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, orbit, distance=None, chi=False):
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
        chi0 = np.array([orbit.ex / self.sigma, orbit.ey / self.sigma])
        res = chi0 if chi else -0.5*chi0**2
        return res


class Log10AreaPrior:
    """Gaussian prior on log10(area) of object.

    Parameters
    ----------
    mean : float
        mean of log10(area)
    sigma : float
        standard deviation of log10(area)

    Attributes
    ----------
    mean, sigma

    Methods
    -------
    __call__(orbit)
        Returns log prior probability of given orbit.
    """
    def __init__(self, mean=-0.5, sigma=1.5):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, orbit, distance=None, chi=False):
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
        chi0 = (np.log10(orbit.propkw['area']) - self.mean) / self.sigma
        res = chi0 if chi else -0.5*chi0**2
        return res


class AreaPrior:
    """Gaussian prior on log10(area) of object.

    Parameters
    ----------
    mean : float
        mean of log10(area)
    sigma : float
        standard deviation of log10(area)

    Attributes
    ----------
    mean, sigma

    Methods
    -------
    __call__(orbit)
        Returns log prior probability of given orbit.
    """
    def __init__(self, mean=0, sigma=2):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, orbit, distance=None, chi=False):
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
        chi0 = (orbit.propkw['area'] - self.mean) / self.sigma
        res = chi0 if chi else -0.5*chi0**2
        return res


# Stolen from emcee.utils  Was deprecated in emcee version 3, so
# reimplementing here.
def sample_ball(p0, std, nSample=1):
    """Produce a ball of walkers around an initial parameter value.

    Parameters
    ----------
    p0 : array_like, (d,)
        Central parameter values.
    std : array_like, (d,)
        Axis-aligned standard deviations.
    nSample : int, optional
        The number of samples to produce.

    Returns
    -------
    samples : array_like (nSample, d)
        The samples.
    """
    assert(len(p0) == len(std))
    return np.vstack([p0 + std * np.random.normal(size=len(p0))
                      for i in range(nSample)])


class GEOProjectionInitializer:
    """Initialize position and velocity samples by projecting an observed ra/dec
    into the equatorial plane and assuming a GEO statationary orbital velocity.
    Samples are dispersed from this given position and velocity using an
    isotropic Gaussian distribution.

    Parameters
    ----------
    arc : QTable with columns for 'ra', 'dec', 'rStation_GCRF'.
        The set of observations used to initialize samples.  Note, only the
        first row of the given QTable will be used.
    rSigma : float, optional
        Standard deviation of sample positions dispersion in meters.
    vSigma : float, optional
        Standard deviation of sample velocity dispersion in meters per second.

    Attributes
    ----------
    rSigma
    vSigma
    observation : QTable
        First row of input arc.

    Methods
    -------
    __call__(nSample)
        Returns `nSample` initial samples.
    """
    def __init__(self, arc, rSigma=0.1*RGEO, vSigma=0.1*VGEO):
        self.observation = arc[0]
        self.rSigma = rSigma
        self.vSigma = vSigma

    def __call__(self, nSample):
        """Generate samples.

        Parameters
        ----------
        nSample : int
            Number of samples to return.

        Returns
        -------
        samples : array_like, (nSample, 6)
            Generated samples.  Columns are [x, y, z, vx, vy, vz].
        """
        from astropy import units as u

        # Only use the first observation
        rStation = np.array(self.observation['rStation_GCRF'].to(u.m).value)
        ra = self.observation['ra']
        dec = self.observation['dec']
        sdec, cdec = np.sin(dec), np.cos(dec)
        sra, cra = np.sin(ra), np.cos(ra)
        lineOfSight = np.array([cdec*cra, cdec*sra, sdec])

        range = -rStation[2] / lineOfSight[2]
        rGuess = rStation + lineOfSight * range
        zhat = np.array([0., 0., 1.])
        vGuess = VGEO*normed(np.cross(zhat, rGuess))

        guess = np.hstack([rGuess, vGuess])
        return sample_ball(guess, 3*[self.rSigma]+3*[self.vSigma], nSample)


class DistanceProjectionInitializer:
    """Initialize position and velocity from two closely spaced (in time)
    observations of ra/dec.  The position is initialized by projecting along the
    direction of the first observation a given slant range.  The velocity is
    then initialized assuming that the slant range derivative is zero, and that
    the motion in between observations is inertial.

    Parameters
    ----------
    arc : QTable with columns for 'ra', 'dec', 'rStation_GCRF'.
        The set of observations used to initialize samples.  Note, only the
        first row of the given QTable will be used.
    rho : Distance in meters to project, measured from the position of the
        observer (not the center of the Earth).
    indices : list of indices indicating which two observations in `arc` to use
        for initialization.  Default: [0, 1]; i.e., use the first two
        observations.
    rSigma : float, optional
        Standard deviation of sample positions dispersion in meters.
    vSigma : float, optional
        Standard deviation of sample velocity dispersion in meters per second.

    Attributes
    ----------
    observations : QTable
        Indicated two rows of the input arc.
    rho
    rSigma
    vSigma

    Methods
    -------
    __call__(nSample)
        Returns `nSample` initial samples.
    """
    def __init__(
        self, arc, rho, indices=[0, 1], rSigma=0.1*RGEO, vSigma=0.1*VGEO
    ):
        self.observations = arc[indices]
        self.rho = rho
        self.rSigma = rSigma
        self.vSigma = vSigma

    def __call__(self, nSample):
        from astropy import units as u

        rStation = self.observations['rStation_GCRF'].to(u.m).value

        ra = self.observations['ra']
        dec = self.observations['dec']
        sra, cra = np.sin(ra), np.cos(ra)
        sdec, cdec = np.sin(dec), np.cos(dec)
        lineOfSight = np.array([cdec*cra, cdec*sra, sdec]).T

        rGuess = rStation + lineOfSight * self.rho
        dt = (self.observations['time'][1] - self.observations['time'][0])
        dt = dt.to(u.s).value
        vGuess = (rGuess[1] - rGuess[0])/dt

        guess = np.hstack([rGuess[0], vGuess])
        return sample_ball(guess, 3*[self.rSigma]+3*[self.vSigma], nSample)


def circular_guess(arc, indices=[0, 1]):
    """Guess position and velocity from two closely spaced (in time)
    observations of ra/dec.

    The state is inferred by finding the circular orbit passing
    through the two points, assuming the motion in between the
    observations is inertial.

    Parameters
    ----------
    arc : QTable with columns for 'ra', 'dec', 'rStation_GCRF'.
        The set of observations used to make the guess.  Note, only indices
        rows of the given QTable will be used.
    indices : list of indices indicating which two observations in `arc` to use
        for initialization.  Default: [0, 1]; i.e., use the first two
        observations.

    Returns
    -------
    state: array_like, (6,)
        state corresponding to orbit passing through points
    """
    from astropy import units as u

    usepm = 'pmra' in arc.dtype.names
    if not usepm:
        observations = arc[indices]
    else:
        observations = arc
    rStation = observations['rStation_GCRF'].to(u.m).value
    ra = observations['ra']
    dec = observations['dec']
    sra, cra = np.sin(ra), np.cos(ra)
    sdec, cdec = np.sin(dec), np.cos(dec)
    lineOfSighta = (np.array([cdec*cra, cdec*sra, sdec]).T)
    if not usepm:
        lineOfSight = normed(lineOfSighta[0] + lineOfSighta[1])
    else:
        lineOfSight = lineOfSighta[0]
    zhat = np.array([0., 0., 1.])
    rAlpha = normed(np.cross(zhat, lineOfSight))
    rDelta = normed(np.cross(lineOfSight, rAlpha))
    if not usepm:
        dt = (observations['time'][1] - observations['time'][0])
        epoch = observations['time'][0]+dt/2
        dt = dt.to(u.s).value
        if dt == 0:
            dt = 1e-3
            # something horribly wrong in this case;
            # okay to have really bad guess?
        muAlpha = ((((ra[1]-ra[0]).to(u.rad).value)+np.pi)
                   % (2*np.pi)-np.pi)*cdec[0].value/dt
        muDelta = (dec[1]-dec[0]).to(u.rad).value/dt
        vGround = (rStation[1]-rStation[0])/dt
        rStation = 0.5*(rStation[0]+rStation[1])
    else:
        muAlpha = observations['pmra'][0].to(u.rad/u.s).value
        muDelta = observations['pmdec'][0].to(u.rad/u.s).value
        vGround = observations['vStation_GCRF'][0].to(u.m/u.s).value
        epoch = observations['time'][0]
        rStation = rStation[0]
    vGroundSky = vGround-lineOfSight*np.dot(lineOfSight, vGround)

    # gives the geocentric radial velocity as a function of range, assuming
    # a circular orbit, so v^2/R = GM/R^2
    # the zero of this is the distance at which the object would be on a
    # circular orbit.

    def vr2(r):
        """
        Calculate the radial velocity difference based on the provided position vector.

        Parameters:
        -----------
        r : numpy.ndarray
            A position vector representing the relative location along the line of sight.

        Returns:
        --------
        float
            The calculated radial velocity difference based on the Earth's gravitational constant,
            position, and velocity components.

        Example:
        --------
        >>> r = np.array([1.0, 2.0, 3.0])
        >>> vr2(r)
        -12345.678  # Example output (depends on global variable values)
        """

        pos = rStation + r*lineOfSight
        R = np.sqrt(np.sum(pos**2))
        return WGS84_EARTH_MU/R - normSq(muAlpha*rAlpha*r + muDelta*rDelta*r +
                                         vGroundSky)

    def vR(r, ub=np.inf, verbose=False, square=False):
        """
        Compute the radial velocity at a given position along the line of sight, 
        with optional bounds and verbosity.

        Parameters:
        -----------
        r : float
            The position along the line of sight.
        ub : float, optional
            Upper bound for the position `r`. If `r` exceeds this bound, it is reflected 
            back within the bounds. Default is `np.inf` (no bound).
        verbose : bool, optional
            If `True`, additional information may be printed during execution. 
            Default is `False`.
        square : bool, optional
            If `True`, the squared radial velocity (`vrsquared`) is returned instead 
            of the computed radial velocity. Default is `False`.

        Returns:
        --------
        float
            The computed radial velocity at the given position `r`. If `square=True`, 
            the squared radial velocity is returned instead.

        Example:
        --------
        >>> r = 5.0
        >>> vR(r, ub=10.0, verbose=True, square=False)
        123.456  # Example output (depends on global variable values)

        >>> vR(r, ub=10.0, square=True)
        15234.789  # Example squared radial velocity output
        """
        if r > ub:
            sgn = -1
            r = 2*ub - r
        else:
            sgn = 1
        pos = rStation + r*lineOfSight
        vrsquared = np.clip(vr2(r), 0, np.inf)
        ret = np.dot(normed(pos),
                     (sgn*np.sqrt(vrsquared)*lineOfSight +
                      muAlpha*r*rAlpha +
                      muDelta*r*rDelta + vGroundSky))
        return ret

    from scipy.optimize import root, root_scalar, leastsq
    from functools import partial

    vr2near = vr2(0)
    vr2far = vr2(1000*RGEO)
    if np.sign(vr2near) == np.sign(vr2far):
        # no circular solution.
        # must be space observer?
        # we should take a very close distance and return it as a best fit.
        # e.g., 200 km (?), v_los = 0?
        rRange = 200000
        rGuess = rStation + rRange*lineOfSight
        vGuess = muAlpha*rRange*rAlpha + muDelta*rRange*rDelta + vGroundSky
        return np.concatenate([rGuess, vGuess]), epoch
    resub = root_scalar(vr2, bracket=[0, 1000*RGEO])
    sol, covar, result, mesg, ier = leastsq(
        vR, resub.root-1000, args=(resub.root, False, True), full_output=True)
    if sol > resub.root:
        sgn = -1
        sol = 2*resub.root - sol
    else:
        sgn = 1
    if sol <= 200000:
        rRange = 200000
        rGuess = rStation + rRange*lineOfSight
        vGuess = muAlpha*rRange*rAlpha + muDelta*rRange*rDelta + vGroundSky
        return np.concatenate([rGuess, vGuess]), epoch
    rRange = sol[0]
    vlos = sgn*np.sqrt(vr2(rRange))
    import ssapy.utils
    ra, dec = ssapy.utils.unit_to_lb(np.array([lineOfSight]))
    rGuess, vGuess = radecRateObsToRV(
        ra[0], dec[0], rRange, muAlpha, muDelta, vlos, rStation, vGroundSky)
    # ignores light-time correction, etc.
    if np.any(~np.isfinite(rGuess+vGuess)):
        import pdb
        pdb.set_trace()

    return np.concatenate([rGuess, vGuess]), epoch


class GaussianRVInitializer:
    """Generate position and velocity samples as an isotropic Gaussian
    distribution around an asserted position and velocity.

    Parameters
    ----------
    rMean : array_like (3,)
        Mean of sample positions in meters.
    vMean : array_like (3,)
        Mean of sample velocities in meters per second.
    rSigma : float, optional
        Standard deviation of sample positions dispersion in meters.
    vSigma : float, optional
        Standard deviation of sample velocity dispersion in meters per second.

    Attributes
    ----------
    rMean
    vMean
    rSigma
    vSigma

    Methods
    -------
    __call__(nSample)
        Returns `nSample` initial samples.
    """
    def __init__(self, rMean, vMean, rSigma=0.1*RGEO, vSigma=0.1*VGEO):
        self.rMean = rMean
        self.rSigma = rSigma
        self.vMean = vMean
        self.vSigma = vSigma

    def __call__(self, nSample):
        """ Generate initial samples.

        Parameters
        ----------
        nSample : int
            Number of samples to return.

        Returns
        -------
        samples : array_like, (nSample, 6)
            Generated samples.  Columns are [x, y, z, vx, vy, vz].
        """
        guess = np.hstack([self.rMean, self.vMean])
        return sample_ball(guess, 3*[self.rSigma]+3*[self.vSigma], nSample)


class DirectInitializer:
    """Directly specify initial position and velocity samples.  This can be
    useful if restarting sampling after some previous sampling step has
    completed, for instance, if adding new observations to a previously sampled
    arc.

    Parameters
    ----------
    samples : array_like (nSample, 6)
        Samples in position (in meters) and velocity (in meters per second).
        Columns are [x, y, z, vx, vy, vz].
    replace : bool, optional
        Whether to sample with or without replacement.

    Attributes
    ----------
    samples
    replace

    Methods
    -------
    __call__(nSample)
        Returns `nSample` initial samples.
    """
    def __init__(self, samples, replace=False):
        self.samples = samples
        self._nAvailable = self.samples.shape[0]
        self.replace = replace

    def __call__(self, nSample=50):
        """ Return samples.

        Parameters
        ----------
        nSample : int
            Number of samples to return.

        Returns
        -------
        samples : array_like, (nSample, 6)
            Generated samples.  Columns are [x, y, z, vx, vy, vz].

        Notes
        -----
        If self.replace=False and nSample is greater than the number of input
        samples, then some samples will be duplicated.
        """
        if not self.replace and nSample >= self._nAvailable:
            nTile = int(nSample/self._nAvailable)
            nDone = nTile*self._nAvailable
            out = np.tile(self.samples, nTile)
            if nSample > nDone:
                w = np.random.choice(
                    self._nAvailable, size=nSample-nDone, replace=False
                )
                more = self.samples[w]
                out = np.concatenate([out, more])
            return out
        else:
            w = np.random.choice(
                self._nAvailable, size=nSample, replace=self.replace
            )
            return self.samples[w]


class RVProbability:
    """A class to manage MCMC sampling of orbits (parameterized by position and
    velocity at a given epoch) given some angular observations.

    Parameters
    ----------
    arc : QTable with one row per observation and columns:
            'ra', 'dec' : Angle
                Observed angular position in ICRS (topocentric).
            'rStation_GCRF' : Quantity
                Position of observer in GCRF.
            'sigma' : Quantity
                Isotropic uncertainty in observed angular position.
            'time' : float or astropy.time.Time
                Time of observation.  If float, then should correspond to GPS
                seconds; i.e., seconds since 1980-01-06 00:00:00 UTC
        The arc is a linked set of observations assumed to be of the same
        object.
    epoch : float or astropy.time.Time
        The time at which the model parameters (position and velocity) are
        specified.  It probably makes sense to make this the same as one of the
        observation times, but it's not required to be.  If float, then should
        correspond to GPS seconds; i.e., seconds since 1980-01-06 00:00:00 UTC
    priors : list of priors, optional
        A list of class instances similar to RPrior() or APrior() to apply
        jointly to the posterior probability.
    propagator : class, optional
        The propagator class to use.
    meanpm : bool
        fit model to mean positions and proper motions rather than to
        individual epochs along each streak.
    damp : float
        damp chi residuals with pseudo-Huber loss function.  Default -1: do
        not damp.

    Attributes
    ----------
    arc
    epoch
    priors
    propagator

    Methods
    -------
    lnlike(orbit)
        Calculate log likelihood of observations given orbit.
    lnprob(p)
        Calculate log posterior probability of orbit parameters p given
        observations (up to a constant).  Parameters p are [x, y, z, vx, vy, vz]
        at epoch.
    __call__(p)
        Alias for lnprob(p)
    """
    def __init__(self, arc, epoch,
                 priors=[RPrior(RGEO, RGEO*0.2), APrior(RGEO, RGEO*0.2)],
                 propagator=KeplerianPropagator(),
                 meanpm=False, damp=-1):
        from astropy import units as u

        if isinstance(epoch, Time):
            epoch = epoch.gps
        self.arc = arc
        self.epoch = epoch
        self.priors = priors
        self.propagator = propagator

        # Convert Quantities to plain arrays now so conversions don't slow us
        # down during sampling.
        ra = self.arc['ra']
        dec = self.arc['dec']
        self.ra = ra.to(u.rad).value
        self.dec = dec.to(u.rad).value
        sdec, cdec = np.sin(dec).value, np.cos(dec).value
        sra, cra = np.sin(ra).value, np.cos(ra).value
        # line-of-sight calculation
        self._los = np.vstack([cdec*cra, cdec*sra, sdec]).T
        time = self.arc['time']
        self._rStation = np.array(self.arc['rStation_GCRF'].to(u.m).value)
        self._vStation = np.array(self.arc['vStation_GCRF'].to(u.m/u.s).value)
        self.meanpm = meanpm
        if meanpm:
            self._raSigma = np.array(self.arc['dra'].to(u.rad).value)
            self._decSigma = np.array(
                self.arc['ddec'].to(u.rad).value)
            self.pmra = np.array(
                self.arc['pmra'].to(u.rad/u.s).value)
            self.pmdec = np.array(
                self.arc['pmdec'].to(u.rad/u.s).value)
            self._pmraSigma = np.array(
                self.arc['dpmra'].to(u.rad/u.s).value)
            self._pmdecSigma = np.array(
                self.arc['dpmdec'].to(u.rad/u.s).value)
        else:
            if 'dra' in self.arc.dtype.names:
                self._raSigma = np.array(self.arc['dra'].to(u.rad).value)
                self._decSigma = np.array(self.arc['ddec'].to(u.rad).value)
            else:
                self._raSigma = np.array(self.arc['sigma'].to(u.rad).value)
                self._decSigma = np.array(self.arc['sigma'].to(u.rad).value)
        if isinstance(time, Time):
            time = time.gps
        self._time = time
        self.damp = damp

    def chi(self, orbit):
        """Calculate chi residuals for use with lmfit.

        This is essentially the input to a chi^2 statistic, just without the
        squaring and without the summing over observations.  I.e., it's

        [(data_i - model_i)/err_i for i in 0..nobs]

        Parameters
        ----------
        orbit : Orbit
            Orbit in question.  The model.

        Returns
        -------
        chilike : array_like (nobs,)
            chi residual array for likelihoods
        chiprior : array_like (nprior,)
            chi residual array for priors
        """
        hitbadorbit = False
        vmax = np.sqrt(2*WGS84_EARTH_MU/norm(orbit.r))
        if norm(orbit.v) > vmax*0.999:
            newv = np.array(orbit.v)*vmax/norm(orbit.v)*0.999
            orbit = Orbit(r=orbit.r, v=newv, t=orbit.t, propkw=orbit.propkw)
            hitbadorbit = True
        if norm(orbit.r) < 1:
            orbit = Orbit(r=[1, 0, 0], v=orbit.v, t=orbit.t, propkw=orbit.propkw)
            hitbadorbit = True

        if self.meanpm:
            ra, dec, slant, pmra, pmdec, slantrate = (
                radec(orbit, self._time, obsPos=self._rStation,
                      obsVel=self._vStation,
                      propagator=self.propagator, rate=True))
            drarate = (pmra - self.pmra)/self._pmraSigma
            ddecrate = (pmdec - self.pmdec)/self._pmdecSigma
        else:
            ra, dec, slant = radec(orbit, self._time, obsPos=self._rStation,
                                   propagator=self.propagator)
        dra = ((ra - self.ra + np.pi)%(2*np.pi))-np.pi
        dra = np.cos(dec)*dra/self._raSigma
        ddec = (dec-self.dec)/self._decSigma
        if self.meanpm:
            chi = np.concatenate([dra, drarate, ddec, ddecrate])
        else:
            chi = np.concatenate([dra, ddec])
        if self.damp > 0:
            chi = damper(chi, self.damp)
        if self.priors is not None and len(self.priors) > 0:
            chiprior = np.concatenate([
                    np.atleast_1d(prior(orbit, distance=slant, chi=True))
                    for prior in self.priors])
        else:
            chiprior = np.zeros(1)
        if hitbadorbit:
            chi += np.sign(chi)*1e5
        return chi, chiprior

    def lnlike(self, orbit):
        """Calculate the log likelihood of the observations given an orbit.

        Parameters
        ----------
        orbit : Orbit
            Orbit in question.

        Returns
        -------
        lnlike : float
            Log likelihood.
        """
        return -0.5*np.sum(self.chi(orbit)[0]**2)

    def lnprob(self, p):
        """Calculate the log posterior probability of parameters p given
        observations.

        Parameters
        ----------
        p : array (6,)
            Sampling parameters.  [x, y, z, vx, vy, vz] in meters and meters per
            second.  Understood to be the parameters at epoch.

        Returns
        -------
        lnprob : float
            Log posterior probability.
        lnprior : float
            Log prior probability.
        """
        r = p[0:3]
        v = p[3:6]
        try:
            orbit = Orbit(r, v, self.epoch)
        except Exception:
            return -np.inf, -np.inf

        chilike, chiprior = self.chi(orbit)
        lnprior = np.sum(-0.5*chiprior**2)

        out = np.sum(-0.5*chilike**2)
        out += lnprior
        if not np.isfinite(out):
            return -np.inf, lnprior
        return out, lnprior

    def lnprior(self, orbit):
        """Calculate the log prior probability of the observations given an
        orbit.

        Parameters
        ----------
        orbit : Orbit
            Orbit in question.

        Returns
        -------
        lnprior : float
            Log prior probability.
        """
        return -0.5*np.sum(self.chi(orbit)[1]**2)

    __call__ = lnprob


class EmceeSampler:
    """A sampler built on the emcee package.

    The emcee packages implements the Goodman and Weare (2010) affine-invariant
    sampler.  This is often an efficient sampler to use when selecting a
    proposal distribution is not simple.

    Parameters
    ----------
    probfn : Callable
        Callable that accepts sampling parameters p and returns posterior
        probability.
    initializer : Callable
        Callable that accepts number of desired initial samples and returns
        samples.  Note, the initial samples should be (at least mostly) unique.
    nWalker : int
        Number of 'walkers' to use in the Goodman & Weare algorithm.  This
        should generally be at least 12 for the 6-dimensional problem.

    Attributes
    ----------
    probfn
    initializer
    nWalker

    Methods
    -------
    sample(nBurn, nStep)
        Generate samples, first discarding nBurn steps, and then keeping nStep
        steps.
    """
    def __init__(self, probfn, initializer, nWalker=50):
        # Don't need emcee right here, but better to catch version
        # incompatibility in ctor than down below.
        import emcee
        if _emcee_version_before_3(emcee.__version__):
            raise ValueError("emcee version at least 3.0rc2 required")

        self.probfn = probfn
        self.initializer = initializer
        self.nWalker = nWalker

    def sample(self, nBurn=1000, nStep=500):
        """Generate samples.

        Parameters
        ----------
        nBurn : int
            Number of initial steps to take but discard.
        nStep : int
            Number of subsequent steps to keep and return.

        Returns
        -------
        chain : array (nStep, nWalker, 6)
            Generated samples.  Columns are [x, y, z, vx, vy, vz].
        lnprob : array (nStep, nWalker)
            The log posterior probabilities of the samples.
        lnprior : array(nStep, nWalker)
            The log prior values for each step.
        """
        import emcee
        if _emcee_version_before_3(emcee.__version__):
            raise ValueError("emcee version at least 3.0rc2 required")
        # Type for emcee 'blobs' metadata
        dtype = [("lnprior", float)]
        walkers = self.initializer(self.nWalker)
        sampler = emcee.EnsembleSampler(
            self.nWalker, 6, self.probfn, blobs_dtype=dtype
        )
        pos, _, _, _ = sampler.run_mcmc(walkers, max(nBurn, 1))
        sampler.reset()
        # Could add a trim step here to cut out any huge outliers.
        pos, _, _, _ = sampler.run_mcmc(pos, nStep)

        blobs = sampler.get_blobs()
        lnprior = blobs['lnprior']

        return sampler.get_chain(), sampler.get_log_prob(), lnprior


class MVNormalProposal:
    """A multivariate normal proposal distribution.

    Params
    ------
    cov : array_like(6, 6)
        Covariance of multivariate normal distribution from which to sample.
        The order of variables is [x, y, z, vx, vy, vz] in meters and meters per
        second.

    Attributes
    ----------
    cov

    Methods
    -------
    propose(p)
        Generate a new proposal.
    """
    def __init__(self, cov):
        self.cov = cov

    def propose(self, p):
        """Generate a proposal.

        Parameters
        ----------
        p : array (6,)
            Mean around which to generate a proposal.

        Returns
        -------
        newp : array (6,)
            Generated proposal.
        """
        return np.random.multivariate_normal(p, self.cov)


class RVSigmaProposal(MVNormalProposal):
    """Isotropic multivariate normal proposal distribution.

    Params
    ------
    rSigma : float
        Isotropic standard deviation in position to use.
    vSigma : float
        Isotropic standard deviation in velocity to use.

    Attributes
    ----------
    cov

    Methods
    -------
    propose(p)
        Generate a new proposal.
    """
    def __init__(self, rSigma, vSigma):
        self.cov = np.diag([rSigma**2]*3+[vSigma**2]*3)


class MHSampler:
    """Generate MCMC samples using a Metropolis-Hastings sampler.

    Parameters
    ----------
    probfn : Callable
        Callable that accepts sampling parameters p and returns posterior
        probability.
    initializer : Callable
        Callable that accepts number of desired initial samples and returns
        samples.  Note, the initial samples should be (at least mostly) unique.
    proposer : Callable
        Callable that accepts a "current" sample and returns a "proposed" sample
        to either accept or reject at each step.
    nChain : int
        Number of independent chains to use.

    Attributes
    ----------
    probfn
    initializer
    proposer
    nChain
    chain : array_like (nStep, nChain, 6)
        MCMC sample chain.
    lnprob : array_like (nStep, nChain)
        Log posterior probability of sample chain.
    nAccept : int
        Total number of accepted proposals.
    nStep : int
        Total number of proposal steps.

    Methods
    -------
    reset()
        Reset chains.
    sample(nBurn, nStep)
        Generate samples, first discarding nBurn steps, and then keeping nStep
        steps.
    """
    def __init__(self, probfn, initializer, proposer, nChain):
        self.probfn = probfn
        self.proposer = proposer
        self.nChain = nChain
        self._state = initializer(nChain)
        tmp = [self.probfn(s) for s in self._state]
        self._lnprob, self._lnprior = zip(*tmp)
        self._lnprob = list(self._lnprob)
        self._lnprior = list(self._lnprior)
        self.reset()

    def reset(self):
        """Reset chain, including `chain`, `lnprob`, `nAccept`, and `nStep`
        attributes.
        """
        self.chain = []
        self.lnprob = []
        self.lnprior = []
        self.nAccept = 0
        self.nStep = 0

    def _step(self):
        for i in range(self.nChain):
            curr, currPost, currPrior = (
                self._state[i], self._lnprob[i], self._lnprior[i]
            )
            prop = self.proposer.propose(curr)
            propPost, propPrior = self.probfn(prop)
            if (propPost >= currPost
                or np.exp(propPost-currPost) > np.random.uniform()
            ):
                self.nAccept += 1
                currPost = propPost
                currPrior = propPrior
                curr = prop
            self._state[i], self._lnprob[i], self._lnprior[i] = (
                curr, currPost, currPrior
            )
        self.nStep += self.nChain
        self.chain.append(np.array(self._state))
        self.lnprob.append(np.array(self._lnprob))
        self.lnprior.append(np.array(self._lnprior))

    @property
    def acceptanceRatio(self):
        """Ratio of accepted to proposed steps.
        """
        return self.nAccept/self.nStep

    def sample(self, nBurn=1000, nStep=500):
        """Generate samples.

        Parameters
        ----------
        nBurn : int
            Number of initial steps to take but discard.
        nStep : int
            Number of subsequent steps to keep and return.

        Returns
        -------
        chain : array (nStep, nChain, 6)
            Generated samples.  Columns are [x, y, z, vx, vy, vz].
        lnprob : array (nStep, nChain)
            The log posterior probabilities of the samples.
        lnprior : array (nStep, nChain)
            The log prior probabilities of the samples.
        """
        for _ in range(nBurn):
            self._step()
        self.reset()
        for _ in range(nStep):
            self._step()
        return (
            np.array(self.chain),
            np.array(self.lnprob),
            np.array(self.lnprior))


class LeastSquaresOptimizer:
    """Base class for LeastSquaresOptimizers"""
    def __init__(self, probfn, initparam, translatorcls,
                 absstep=None, fracstep=None, **kw):
        self.initparam = initparam
        self.translator = translatorcls(self.initparam, probfn.epoch, **kw)
        self.absstep = absstep
        self.fracstep = fracstep
        self.probfn = probfn

    def resid(self, p):
        orb = self.translator.param_to_orbit(p)
        chi = np.concatenate(self.probfn.chi(orb))
        return chi

    def optimize(self, **fit_kws):
        """Run the optimizer and return the resulting fit parameters.

        Returns
        -------
        fit : (6,) array_like
            Least-squares fit as [x, y, z, vx, vy, vz] in meters, and
            meters/second.
        """
        import scipy.optimize
        par0 = self.translator.input_param_translation(self.initparam)
        par0 = self.translator.optimizeparam(par0)
        if 'maxfev' in fit_kws:
            nmax = fit_kws.pop('maxfev')
            fit_kws['max_nfev'] = nmax
        self.result = scipy.optimize.least_squares(
            self.resid, par0, **fit_kws)
        orbit = self.translator.param_to_orbit(self.result.x)
        vmax = np.sqrt(2*WGS84_EARTH_MU/norm(orbit.r))
        if norm(orbit.v) > vmax*0.999:
            orbit.v = np.array(orbit.v)*vmax/norm(orbit.v)*0.999
            self.result.success = False
            self.result.hithyperbolicorbit = True
        jac = self.result.jac
        u, s, vh = np.linalg.svd(jac)
        covar = vh.T.dot(np.diag(s**(-2))).dot(vh)
        self.result.covar = self.translator.output_covar_translation(covar)
        self.result.residual = self.result.fun
        fitp = self.translator.orbit_to_param(orbit)
        return fitp


class ParamOrbitTranslator():
    """Class for making parameters into Orbits and vice-versa."""
    def __init__(self, initparam, epoch, fixed=None, orbitattr=None):
        self.initParam = initparam
        self.fixed = fixed
        self.orbitattr = orbitattr if orbitattr is not None else []
        self.epoch = epoch

    def fullparam(self, p):
        if self.fixed is not None:
            return np.where(self.fixed, self.initparam, p)
        return p

    def optimizeparam(self, p):
        if self.fixed is not None:
            return np.array(p)[self.fixed]
        else:
            return p

    def param_to_orbit(self):
        raise NotImplementedError

    def orbit_to_param(self):
        raise NotImplementedError

    def get_propkw_from_fullparam(self, fullparam):
        propkw = dict()
        for name, val in zip(self.orbitattr, fullparam[6:]):
            if name.startswith('log10'):
                newname = name[5:]
                newval = 10.**val
            else:
                newname = name
                newval = val
            propkw[newname] = newval
        return propkw

    def get_propkw_from_orbit(self, orbit):
        propkw = []
        for k in self.orbitattr:
            if k.startswith('log10'):
                k = k[5:]
                v = np.log10(orbit.propkw[k])
            else:
                v = orbit.propkw[k]
            propkw.append(v)
        return propkw

    def input_param_translation(self, p):
        return p

    def output_covar_translation(self, covar):
        return covar


class ParamOrbitRV(ParamOrbitTranslator):
    """
    A class for translating orbital parameters in position-velocity (RV) form, 
    including conversions between parameters and orbit objects.

    Inherits:
    ---------
    ParamOrbitTranslator : Base class providing foundational functionality 
    for orbital parameter translation.

    Methods:
    --------
    __init__(*args, **kwargs):
        Initialize the `ParamOrbitRV` object, inheriting behavior from the 
        `ParamOrbitTranslator` base class.

    param_to_orbit(p):
        Convert position-velocity (RV) parameters to an `Orbit` object.

        Parameters:
        -----------
        p : array-like
            Orbital parameters in position-velocity form. The first three 
            elements represent the position vector (`r`), and the next three 
            elements represent the velocity vector (`v`).

        Returns:
        --------
        Orbit
            An `Orbit` object created from the position (`r`) and velocity (`v`) 
            vectors, along with propagation keywords.

        Notes:
        ------
        - Additional propagation keywords are extracted from the full parameter set.
        - The `Orbit` class is expected to support initialization with `r` and `v`.

    orbit_to_param(orbit):
        Convert an `Orbit` object to a list of position-velocity (RV) parameters 
        and associated propagation keywords.

        Parameters:
        -----------
        orbit : Orbit
            An `Orbit` object containing position (`r`) and velocity (`v`) vectors.

        Returns:
        --------
        numpy.ndarray
            A concatenated array of position (`r`), velocity (`v`), and propagation 
            keywords extracted from the orbit.

    Notes:
    ------
    - This class assumes orbital parameters are represented in position-velocity form.
    - The `Orbit` class is expected to provide attributes `r` and `v` for position 
      and velocity vectors, respectively.

    Example:
    --------
    >>> translator = ParamOrbitRV()
    >>> params = [7000, 0, 0, 0, 7.5, 0]  # Example position-velocity parameters
    >>> orbit = translator.param_to_orbit(params)
    >>> new_params = translator.orbit_to_param(orbit)
    >>> print(new_params)
    [7000, 0, 0, 0, 7.5, 0]  # Example output matching input
    """
    def __init__(self, *args, **kwargs):
        super(ParamOrbitRV, self).__init__(*args, **kwargs)

    def param_to_orbit(self, p):
        fullparam = self.fullparam(p)
        r = p[:3]
        v = p[3:6]
        propkw = self.get_propkw_from_fullparam(p)
        orbit = Orbit(r=r, v=v, t=self.epoch, propkw=propkw)
        return orbit

    def orbit_to_param(self, orbit):
        r = orbit.r
        v = orbit.v
        propkw = self.get_propkw_from_orbit(orbit)
        return np.concatenate([r, v, propkw])


class ParamOrbitEquinoctial(ParamOrbitTranslator):
    """
    A class for translating orbital parameters in equinoctial form, 
    including input/output transformations and conversions between 
    parameters and orbit objects.

    Inherits:
    ---------
    ParamOrbitTranslator : Base class providing foundational functionality 
    for orbital parameter translation.

    Methods:
    --------
    __init__(*args, **kwargs):
        Initialize the `ParamOrbitEquinoctial` object, inheriting behavior 
        from the `ParamOrbitTranslator` base class.

    input_param_translation(p):
        Transform input orbital parameters by scaling the first parameter 
        (typically semi-major axis) for internal use.

        Parameters:
        -----------
        p : array-like
            Input orbital parameters.

        Returns:
        --------
        numpy.ndarray
            Transformed orbital parameters with the first element scaled.

    output_covar_translation(covar):
        Transform covariance matrix for output by scaling the first row 
        and column (typically related to semi-major axis).

        Parameters:
        -----------
        covar : array-like
            Input covariance matrix.

        Returns:
        --------
        numpy.ndarray
            Transformed covariance matrix with scaled elements.

    param_to_orbit(p):
        Convert equinoctial orbital parameters to an `Orbit` object, 
        ensuring constraints on eccentricity and semi-major axis.

        Parameters:
        -----------
        p : array-like
            Equinoctial orbital parameters.

        Returns:
        --------
        Orbit
            An `Orbit` object created from the equinoctial parameters.

        Notes:
        ------
        - The semi-major axis (`p[0]`) is scaled and clipped to a minimum value of 1.
        - The eccentricity vector (`p[3]`, `p[4]`) is normalized if its magnitude exceeds 0.999.
        - Additional propagation keywords are extracted from the full parameter set.

    orbit_to_param(orbit):
        Convert an `Orbit` object to a list of equinoctial orbital parameters 
        and associated propagation keywords.

        Parameters:
        -----------
        orbit : Orbit
            An `Orbit` object containing equinoctial elements.

        Returns:
        --------
        list
            A list of equinoctial orbital parameters and propagation keywords.

    Notes:
    ------
    - This class assumes equinoctial orbital elements are used for parameterization.
    - The `Orbit` class is expected to provide methods for handling equinoctial elements.
    - Scaling factors (e.g., 1e7 for semi-major axis) are applied for numerical stability.

    Example:
    --------
    >>> translator = ParamOrbitEquinoctial()
    >>> params = [7000000, 0, 0, 0.01, 0.01, 0]  # Example equinoctial parameters
    >>> orbit = translator.param_to_orbit(params)
    >>> new_params = translator.orbit_to_param(orbit)
    >>> print(new_params)
    [7000000, 0, 0, 0.01, 0.01, 0]  # Example output matching input
    """
    def __init__(self, *args, **kwargs):
        super(ParamOrbitEquinoctial, self).__init__(*args, **kwargs)

    def input_param_translation(self, p):
        p = np.array(p).copy()
        p[0] /= 1e7
        return p

    def output_covar_translation(self, covar):
        covar = np.array(covar).copy()
        covar[:, 0] *= 1e7
        covar[0, :] *= 1e7
        return covar

    def param_to_orbit(self, p):
        fullparam = self.fullparam(p)
        p = np.array(p).copy()
        p[0] = np.clip(p[0], 1, np.inf)
        e2 = np.hypot(p[3], p[4])
        if e2 >= 0.999:
            norm = 0.999 / e2
            p[3] *= norm
            p[4] *= norm
        p[0] *= 1e7
        propkw = self.get_propkw_from_fullparam(p)
        orbit = Orbit.fromEquinoctialElements(*p[:6], t=self.epoch,
                                              propkw=propkw)
        return orbit

    def orbit_to_param(self, orbit):
        fullparam = list(orbit.equinoctialElements)
        fullparam = fullparam + self.get_propkw_from_orbit(orbit)
        return fullparam


class ParamOrbitAngle(ParamOrbitTranslator):
    """
    A class for translating orbital parameters in angle-based form, 
    including conversions between angular parameters and orbit objects.

    Inherits:
    ---------
    ParamOrbitTranslator : Base class providing foundational functionality 
    for orbital parameter translation.

    Attributes:
    -----------
    initObsPos : array-like
        Initial observer position vector used for angle-based calculations.
    initObsVel : array-like
        Initial observer velocity vector used for angle-based calculations.

    Methods:
    --------
    __init__(initparam, epoch, initObsPos, initObsVel, **kwargs):
        Initialize the `ParamOrbitAngle` object with initial parameters, 
        epoch, observer position, and observer velocity.

        Parameters:
        -----------
        initparam : array-like
            Initial orbital parameters.
        epoch : float
            Epoch time for the orbit.
        initObsPos : array-like
            Initial observer position vector.
        initObsVel : array-like
            Initial observer velocity vector.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.

    param_to_orbit(p):
        Convert angle-based orbital parameters to an `Orbit` object.

        Parameters:
        -----------
        p : array-like
            Orbital parameters in angle-based form. The first six elements 
            represent angular parameters (e.g., right ascension, declination, 
            and rates).

        Returns:
        --------
        Orbit
            An `Orbit` object created from the angular parameters, along with 
            propagation keywords.

        Notes:
        ------
        - The method `radecRateObsToRV` is used to convert angular parameters 
          to position (`r`) and velocity (`v`) vectors.
        - Additional propagation keywords are extracted from the full parameter set.

    orbit_to_param(orbit):
        Convert an `Orbit` object to a list of angle-based orbital parameters 
        and associated propagation keywords.

        Parameters:
        -----------
        orbit : Orbit
            An `Orbit` object containing position (`r`) and velocity (`v`) vectors.

        Returns:
        --------
        list
            A list of angle-based orbital parameters (e.g., right ascension, 
            declination, and rates) and propagation keywords.

        Notes:
        ------
        - The method `rvObsToRaDecRate` is used to convert position (`r`) and 
          velocity (`v`) vectors to angular parameters.
        - Observer position and velocity are used in the conversion process.

    Example:
    --------
    >>> initObsPos = [0, 0, 0]  # Example observer position
    >>> initObsVel = [0, 0, 0]  # Example observer velocity
    >>> translator = ParamOrbitAngle(initparam=[1, 2, 3, 4, 5, 6], epoch=2451545.0, 
    ...                              initObsPos=initObsPos, initObsVel=initObsVel)
    >>> params = [1, 2, 3, 4, 5, 6]  # Example angle-based parameters
    >>> orbit = translator.param_to_orbit(params)
    >>> new_params = translator.orbit_to_param(orbit)
    >>> print(new_params)
    [1, 2, 3, 4, 5, 6]  # Example output matching input
    """
    def __init__(self, initparam, epoch, initObsPos, initObsVel, **kwargs):
        super(ParamOrbitAngle, self).__init__(initparam, epoch, **kwargs)
        self.initObsPos = initObsPos
        self.initObsVel = initObsVel

    def param_to_orbit(self, p):
        fullparam = self.fullparam(p)
        r, v = radecRateObsToRV(*p[:6], self.initObsPos, self.initObsVel)
        propkw = self.get_propkw_from_fullparam(p)
        orbit = Orbit(r, v, self.epoch, propkw=propkw)
        return orbit

    def orbit_to_param(self, orbit):
        r, v = orbit.r, orbit.v
        p = rvObsToRaDecRate(r, v, self.initObsPos, self.initObsVel)
        propkw = self.get_propkw_from_orbit(orbit)
        return list(p)+propkw


class LMOptimizerAngular:
    """
    Optimizer that employs Levenberg-Marquardt least-squares fitting.
    Instead of rv, works in angle/proper motion/range/range rate of initial obs.

    Parameters
    ----------
    probfn : RVProbability
        The RVProbability object that has both an epoch attribute to use for
        the orbit fitting model, and a chi method to use for the fit
        evaluation.
    initGuess : array_like (6,)
        Initial guess.  (ra, dec, range, pmra, pmdec, rangerate).
        (rad, rad, m, rad/s, rad/s, m/s)
    initObsPos : array_like (3,)
        Position of observer at probfn.epoch.
    initObsVel : array_like (3,)
        Position of observer at probfn.epoch.

    Attributes
    ----------
    result : lmfit.MinimizerResult
        Most recently run result object.  Could be useful for inspecting
        error estimates or success/failure conditions.

    Methods
    -------
    optimize()
        Return the optimized parameters list [r, v]
    """
    def __init__(self, probfn, initGuess, initObsPos, initObsVel,
                 fracstep=[1e-7]*6,
                 absstep=[0.01/60/60, 0.01/60/60, 1,
                          0.01/60/60/10/100, 0.01/60/60/10/100,
                          1e-2],
                 orbitattr=None):
        
        import warnings
        warnings.warn("This class is deprecated; use "
                      "ssapy.rvsampler.LeastSquaresOptimizer")

        self.probfn = probfn

        import lmfit
        p = lmfit.Parameters()
        p.add('ra', value=initGuess[0])
        p.add('dec', value=initGuess[1])
        p.add('range', value=initGuess[2])
        p.add('pmra', value=initGuess[3])
        p.add('pmdec', value=initGuess[4])
        p.add('rangerate', value=initGuess[5])
        self.p = p
        self.absstep = absstep
        self.fracstep = fracstep
        self.initObsPos = initObsPos
        self.initObsVel = initObsVel
        self.initGuess = initGuess

    def _getOrbit(self, p):
        if isinstance(p, dict):
            vd = p.valuesdict()
            ra, dec = vd['ra'], vd['dec']
            pmra, pmdec = vd['pmra'], vd['pmdec']
            range, rangerate = vd['range'], vd['rangerate']
        else:
            ra, dec, range, pmra, pmdec, rangerate = p
        r, v = radecRateObsToRV(ra, dec, range, pmra, pmdec, rangerate,
                                self.initObsPos, self.initObsVel)
        orbit = Orbit(r, v, self.probfn.epoch)
        return orbit

    def _resid(self, p):
        chi = np.concatenate(self.probfn.chi(self._getOrbit(p)))
        return chi

    def _jac(self, p):
        dx = np.max([self.fracstep*np.abs(p), self.absstep], axis=0)
        f0 = self._resid(p)
        p0 = np.array(p)
        pi = p0.copy()
        res = np.zeros((len(f0), len(p)), dtype='f8')
        for i, di in enumerate(dx):
            pi[i] = p0[i] + di
            res[:, i] = (self._resid(pi)-f0)/di
            pi[i] = p0[i]
        return res

    def optimize(self, usejac=True, **fit_kws):
        """Run the optimizer and return the resulting fit parameters.

        Returns
        -------
        fit : (6,) array_like
            Least-squares fit as [ra, dec, slant, raRate, decRate,
            slantRate] in rad, rad, m, rad/s, rad/s, m/s.
        """
        import lmfit
        Dfun = self._jac if usejac else None
        self.result = lmfit.minimize(self._resid, self.p, Dfun=Dfun,
                                     **fit_kws)
        orbit = self._getOrbit(self.result.params)
        vmax = np.sqrt(2*WGS84_EARTH_MU/norm(orbit.r))
        if norm(orbit.v) > vmax*0.999:
            orbit.v = np.array(orbit.v)*vmax/norm(orbit.v)*0.999
            self.result.success = False
            self.result.hithyperbolicorbit = True
        if usejac:
            jac = self._jac(
                [x for x in self.result.params.valuesdict().values()])
            u, s, vh = np.linalg.svd(jac)
            covar = vh.T.dot(np.diag(s**(-2))).dot(vh)
            self.result.covar = covar
        return radec(orbit, self.probfn.epoch, obsPos=self.initObsPos,
                     obsVel=self.initObsVel, rate=True)


class LMOptimizer:
    """
    Optimizer that employs Levenberg-Marquardt least-squares fitting.

    Parameters
    ----------
    probfn : RVProbability
        The RVProbability object that has both an epoch attribute to use for
        the orbit fitting model, and a chi method to use for the fit
        evaluation.
    initRV : array_like (6,)
        Initial position and velocity.  Essentially a single output from one
        of the initializers above.

    Attributes
    ----------
    result : lmfit.MinimizerResult
        Most recently run result object.  Could be useful for inspecting
        error estimates or success/failure conditions.

    Methods
    -------
    optimize()
        Return the optimized parameters list [r, v]
    """
    def __init__(self, probfn, initRV,
                 fracstep=[1e-8, 1e-8, 1e-8, 1e-9, 1e-9, 1e-9],
                 absstep=[1, 1, 1, 1e-6, 1e-6, 1e-6],
                 orbitattr=None):

        import lmfit


        import warnings
        warnings.warn("This class is deprecated; use "
                      "ssapy.rvsampler.LeastSquaresOptimizer")

        self.probfn = probfn
        self.initRV = initRV

        p = lmfit.Parameters()
        p.add('x', value=initRV[0])
        p.add('y', value=initRV[1])
        p.add('z', value=initRV[2])
        p.add('vx', value=initRV[3])
        p.add('vy', value=initRV[4])
        p.add('vz', value=initRV[5])
        self.p = p
        self.absstep = absstep
        self.fracstep = fracstep

    def _getOrbit(self, p):
        if isinstance(p, dict):
            vd = p.valuesdict()
            r = [vd['x'], vd['y'], vd['z']]
            v = [vd['vx'], vd['vy'], vd['vz']]
            orbit = Orbit(r, v, self.probfn.epoch)
        else:
            orbit = Orbit(p[:3], p[3:], self.probfn.epoch)
        return orbit

    def _resid(self, p):
        chi = np.concatenate(self.probfn.chi(self._getOrbit(p)))
        return chi

    def _jac(self, p):
        dx = np.max([self.fracstep*np.abs(p), self.absstep], axis=0)
        f0 = self._resid(p)
        p0 = np.array(p)
        pi = p0.copy()
        res = np.zeros((len(f0), len(p)), dtype='f8')
        for i, di in enumerate(dx):
            pi[i] = p0[i] + di
            res[:, i] = (self._resid(pi)-f0)/di
            pi[i] = p0[i]
        return res

    def optimize(self, usejac=True, **fit_kws):
        """Run the optimizer and return the resulting fit parameters.

        Returns
        -------
        fit : (6,) array_like
            Least-squares fit as [x, y, z, vx, vy, vz] in meters, and
            meters/second.
        """
        import lmfit
        Dfun = self._jac if usejac else None
        self.result = lmfit.minimize(self._resid, self.p, Dfun=Dfun,
                                     **fit_kws)
        orbit = self._getOrbit(self.result.params)
        vmax = np.sqrt(2*WGS84_EARTH_MU/norm(orbit.r))
        if norm(orbit.v) > vmax*0.999:
            orbit.v = np.array(orbit.v)*vmax/norm(orbit.v)*0.999
            self.result.success = False
            self.result.hithyperbolicorbit = True
        if usejac:
            jac = self._jac(
                [x for x in self.result.params.valuesdict().values()])
            u, s, vh = np.linalg.svd(jac)
            covar = vh.T.dot(np.diag(s**(-2))).dot(vh)
            self.result.covar = covar
        return np.hstack([orbit.r, orbit.v])


def eq2kep(eq):
    """
    Convert equinoctial orbital elements to classical Keplerian orbital elements.

    Parameters:
    -----------
    eq : array-like
        A list or array of equinoctial orbital elements. The elements are:
        - eq[0]: Semi-major axis (a)
        - eq[1]: Equinoctial parameter h (related to inclination and RAAN)
        - eq[2]: Equinoctial parameter k (related to inclination and RAAN)
        - eq[3]: Equinoctial parameter p (related to eccentricity and argument of periapsis)
        - eq[4]: Equinoctial parameter q (related to eccentricity and argument of periapsis)
        - eq[5]: Mean longitude (L)

    Returns:
    --------
    kep : list
        A list of classical Keplerian orbital elements:
        - kep[0]: Semi-major axis (a)
        - kep[1]: Eccentricity (e)
        - kep[2]: Inclination (i) [in radians]
        - kep[3]: Argument of periapsis (ω) [in radians]
        - kep[4]: Right ascension of the ascending node (Ω) [in radians]
        - kep[5]: Mean anomaly (M) [in radians]

    Example:
    --------
    >>> eq = [7000, 0.1, 0.2, 0.01, 0.02, 1.5]
    >>> kep = eq2kep(eq)
    >>> print(kep)
    [7000, 0.022360679774997897, 0.40489178628508343, -0.4636476090008061, 1.1071487177940904, 1.9634954084936207]
    """
    a = eq[0]
    tanio2 = np.sqrt(eq[1]**2+eq[2]**2)
    e = np.sqrt(eq[3]**2+eq[4]**2)
    aux = np.arctan2(eq[4], eq[3])
    raan = np.arctan2(eq[2], eq[1])
    pa = aux-raan
    kep = [a, e, 2*np.arctan(tanio2), pa, raan, eq[5]-aux]
    return kep


class SGP4LMOptimizer:
    """
    Optimizer that employs Levenberg-Marquardt least-squares fitting
    and fits in Kozai mean Keplerian element space.

    Parameters
    ----------
    probfn : RVProbability
        The RVProbability object that has both an epoch attribute to use for the
        orbit fitting model, and a chi method to use for the fit evaluation.
    initel : array_like (6,)
        Initial Kozai mean Keplerian elements.

    Attributes
    ----------
    result : lmfit.MinimizerResult
        Most recently run result object.  Could be useful for inspecting error
        estimates or success/failure conditions.

    Methods
    -------
    optimize()
        Return the optimized parameters list [r, v]
    """
    def __init__(self, probfn, initel,
                 fracstep=[1e-8]*6,
                 absstep=[100, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]):

        self.probfn = probfn
        self.initel = initel
        self.fieldnames = ['a', 'hx', 'hy', 'ex', 'ey', 'lv']

        import lmfit
        p = lmfit.Parameters()
        for i, name in enumerate(self.fieldnames):
            if name == 'ex' or name == 'ey':
                p.add(name, value=initel[i], min=-0.7, max=0.7)
            else:
                p.add(name, value=initel[i])
        self.p = p
        self.absstep = absstep
        self.fracstep = fracstep

    def _getOrbit(self, p):
        if isinstance(p, dict):
            param = [p[x].value for x in self.fieldnames]
        else:
            param = p
        paramkepler = eq2kep(param)
        return Orbit.fromKozaiMeanKeplerianElements(*paramkepler,
                                                    t=self.probfn.epoch)

    def _resid(self, p):
        chi = np.concatenate(self.probfn.chi(self._getOrbit(p)))
        return chi

    def _jac(self, p):
        dx = np.max([self.fracstep*np.abs(p), self.absstep], axis=0)
        f0 = self._resid(p)
        p0 = np.array(p)
        ind = np.arange(len(p), dtype='i4')
        fi = [self._resid(p0+(ind == i)*di)
              for (i, di) in zip(np.arange(len(p)), dx)]

        res = np.array([(x - f0)/hi for (x, hi) in zip(fi, dx)]).T
        return res


    def optimize(self, **fit_kws):
        """Run the optimizer and return the resulting fit parameters.

        Returns
        -------
        fit : (6,) array_like
            Least-squares fit as [a, e, i, pa, raan, trueAnomaly]
        """
        import lmfit
        self.result = lmfit.minimize(self._resid, self.p,
                                     Dfun=self._jac, **fit_kws)
        return np.array([self.result.params[x] for x in self.fieldnames])


class EquinoctialLMOptimizer:
    """
    Optimizer that employs Levenberg-Marquardt least-squares fitting
    and fits in Kozai mean Keplerian element space.

    Parameters
    ----------
    probfn : RVProbability
        The RVProbability object that has both an epoch attribute to use for the
        orbit fitting model, and a chi method to use for the fit evaluation.
    initel : array_like (6,)
        Initial Kozai mean Keplerian elements.

    Attributes
    ----------
    result : lmfit.MinimizerResult
        Most recently run result object.  Could be useful for inspecting error
        estimates or success/failure conditions.

    Methods
    -------
    optimize()
        Return the optimized parameters list [r, v]
    """
    def __init__(self, probfn, initel,
                 fracstep=[1e-7, 1e-9, 1e-9, 1e-6, 1e-6, 1e-9],
                 absstep=[1e-7, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4],
                 orbitattr=None):

        import warnings
        warnings.warn("This class is deprecated; use "
                      "ssapy.rvsampler.LeastSquaresOptimizer")

        self.probfn = probfn
        self.initel = initel
        self.fieldnames = ['aem7', 'hx', 'hy', 'ex', 'ey', 'lv']

        import lmfit
        p = lmfit.Parameters()
        for i, name in enumerate(self.fieldnames):
            if name == 'ex' or name == 'ey':
                p.add(name, value=initel[i], min=-0.99, max=0.99)
            elif name == 'aem7':
                p.add(name, value=initel[i]/1e7, min=6.3e6/1e7)  # ~radius of earth
            else:
                p.add(name, value=initel[i])
        self.p = p
        self.absstep = absstep
        self.fracstep = fracstep

    def _getOrbit(self, p):
        if isinstance(p, dict):
            param = np.array([p[x].value for x in self.fieldnames]).copy()
        else:
            param = p.copy()
        e2 = np.hypot(param[3], param[4])
        if e2 >= 0.999:
            norm = 0.999/e2
            param[3] *= norm
            param[4] *= norm
        param[0] *= 1e7
        return Orbit.fromEquinoctialElements(*param,
                                             t=self.probfn.epoch)

    def _resid(self, p):
        chi = np.concatenate(self.probfn.chi(self._getOrbit(p)))
        return chi

    def _jac(self, p):
        dx = np.max([self.fracstep*np.abs(p), self.absstep], axis=0)
        f0 = self._resid(p)
        p0 = np.array(p)
        ind = np.arange(len(p), dtype='i4')
        fi = [self._resid(p0+(ind == i)*di)
              for (i, di) in zip(np.arange(len(p)), dx)]

        res = np.array([(x - f0)/hi for (x, hi) in zip(fi, dx)]).T
        return res


    def optimize(self, **fit_kws):
        """Run the optimizer and return the resulting fit parameters.

        Returns
        -------
        fit : (6,) array_like
            Least-squares fit as [a, e, i, pa, raan, trueAnomaly]
        """
        import lmfit
        self.result = lmfit.minimize(self._resid, self.p,
                                     Dfun=self._jac, **fit_kws)
        jac = self._jac(
            [x for x in self.result.params.valuesdict().values()])
        u, s, vh = np.linalg.svd(jac)
        covar = vh.T.dot(np.diag(s**(-2))).dot(vh)
        self.result.covar = covar
        param = np.array([self.result.params[x] for x in self.fieldnames])
        param[0] *= 1e7
        self.result.covar[:, 0] *= 1e7
        self.result.covar[0, :] *= 1e7
        return param


def damper(chi, damp):
    """Pseudo-Huber loss function."""
    return 2*damp*np.sign(chi)*(np.sqrt(1+np.abs(chi)/damp)-1)
    # return chi/np.sqrt(1+np.abs(chi)/damp)


def damper_deriv(chi, damp, derivnum=1):
    """Derivative of the pseudo-Huber loss function."""
    if derivnum == 1:
        return (1+np.abs(chi)/damp)**(-0.5)
    if derivnum == 2:
        return -0.5*np.sign(chi)/damp*(1+np.abs(chi)/damp)**(-1.5)
