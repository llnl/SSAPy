""" This module provides functions for coordinate transformations, vector operations, celestial mechanics, and mathematical utilities. """


import sys
import os
import numpy as np
from astropy.time import Time
import astropy.units as u
from typing import Union, Tuple

from . import datadir
from .constants import RGEO, EARTH_RADIUS, MOON_RADIUS, WGS84_EARTH_OMEGA

try:
    import erfa
except ImportError:
    import astropy._erfa as erfa


def find_file(filename, ext=None):
    """ Find a file in the current directory or the ssapy datadir.  If ext is
    not None, also try appending ext to the filename.
    """
    candidates = [
        filename,
        os.path.join(datadir, filename),
        filename + ext,
        os.path.join(datadir, filename + ext),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(filename)


def _wrapToPi(angle):
    """
    Wrap an angle to the range [-π, π].

    Parameters:
    -----------
    angle : float or array-like
        The input angle(s) in radians. Can be a single value or an array of values.

    Returns:
    --------
    wrapped_angle : float or array-like
        The angle(s) wrapped to the range [-π, π].

    Example:
    --------
    >>> _wrapToPi(4)
    -2.283185307179586
    >>> _wrapToPi(-4)
    2.283185307179586
    >>> _wrapToPi([3.5, -3.5, 6.5])
    array([-2.78318531,  2.78318531,  0.28318531])
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def num_wraps(ang):
    """
    Return number of times angle `ang` has wrapped around.

    Returns
    -------
    int
        Number of wraps
    """
    import astropy.units as u
    if isinstance(ang, u.Quantity):
        ang = float(ang / u.rad)
    return int(np.floor(ang / (2 * np.pi)))


class lazy_property:
    """
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    """

    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value


def newton_raphson(guess, f, fprime=None, eps=3e-16, maxiter=100, **kwargs):
    """
    Find a root of a univariate function with known gradient using
    Newton-Raphson iterations.

    Parameters
    ----------
    guess : float
        Initial guess.
    f : callable function of 1 argument.
        Function for which to find a zero.  If fprime is None, then the return
        value of f should be a 2-tuple with the value of the function and the
        first derivative.  If fprime is given separately, then f should just
        return the value of the function.
    fprime : callable function of 1 argument, optional
        Derivative of f.  Default None.
    eps : float, optional
        Absolute tolerance for finding a zero.  Default 3e-16.
    maxiter : int, optional
        Maximum number of iterations to try.  Default 100.

    Returns
    -------
    solution: float
    """
    delta = np.inf
    niter = 0
    x = guess
    while np.abs(delta) > eps and niter < maxiter:
        if fprime is None:
            val, grad = f(x, **kwargs)
        else:
            val = f(x, **kwargs)
            grad = fprime(x, **kwargs)
        delta = val / grad
        x -= delta
        niter += 1
    return x


def find_extrema_brackets(fs):
    """
    Find triplets bracketing extrema from an ordered set of function
    evaluations.
    """
    # Want to find places where the sign of the finite difference changes.
    diffs = np.trim_zeros(np.diff(fs), 'b')
    w = np.where(diffs == 0)[0]
    while np.any(w):
        diffs[w] = diffs[w + 1]
        w = np.where(diffs == 0)[0]
    w = np.where(diffs[:-1] * diffs[1:] < 0)[0]
    out = []
    for wi in w:
        j = 2
        while fs[wi + j] == fs[wi + 1]:
            j += 1
        out.append((wi, wi + 1, wi + j))
    return out


def find_all_zeros(f, low, high, n=100):
    """
    Attempt to find all zeros of a function between given bounds.  Uses n
    initial samples to characterize the zero-landscape of the interval.
    """
    from scipy.optimize import minimize_scalar, brentq
    # Step one: Evaluate the function n times.
    xs = np.linspace(low, high, n)
    fs = np.array([f(x) for x in xs])
    # Step two: Identify extrema existence.
    extrema = find_extrema_brackets(fs)
    # Step three: Fill in points for extrema that may lead to a zero-crossing.
    newx, newf = [], []
    for extremum in extrema:
        is_minimum = fs[extremum[1]] < fs[extremum[0]]
        if (fs[extremum[1]] - fs[extremum[0]]) * fs[extremum[0]] > 0:
            continue
        # make a record of function evaluations.

        def f2(x):
            result = f(x)
            newx.append(x)
            newf.append(result)
            if not is_minimum:
                result *= -1
            return result
        minimize_scalar(f2, tuple(xs[i] for i in extremum))

    xs = np.concatenate([xs, newx])
    fs = np.concatenate([fs, newf])
    a = np.argsort(xs)
    xs = xs[a]
    fs = fs[a]
    # Step four: Find zero crossings
    zeros = list(xs[fs == 0.0])
    for w in np.where(fs[1:] * fs[:-1] < 0)[0]:
        zeros.append(brentq(f, xs[w], xs[w + 1]))
    return zeros


# Not quite as fast as specializing to shape = (3,), but more portable, and
# faster than np.linalg.norm().
# These compute the norm over the last axis and maintain the other leading axes.
def normSq(arr):
    """
    Compute the squared norm of an array over the last axis while preserving leading axes.

    This function calculates the squared norm of vectors along the last axis of the input array 
    using Einstein summation notation (`np.einsum`). It is designed to be portable and efficient, 
    offering better performance than `np.linalg.norm()` for this specific use case.

    Parameters:
    -----------
    arr : array-like
        Input array where the squared norm is computed along the last axis. 
        The array can have any shape, with the last axis representing the vector components.

    Returns:
    --------
    norm_squared : array-like
        The squared norm of the input array computed along the last axis. 
        The output shape matches the input shape, excluding the last axis.

    Notes:
    ------
    - This implementation is more portable and faster than using `np.linalg.norm()` 
      for computing norms over the last axis, especially for large arrays.
    - It is slightly less specialized compared to methods optimized for fixed shapes 
      (e.g., shape `(3,)`), but it generalizes well to arrays of arbitrary shape.
    - The computation uses `np.einsum("...i,...i", arr, arr)`, which efficiently performs 
      the summation of squared components along the last axis.

    Example:
    --------
    >>> import numpy as np
    >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
    >>> normSq(arr)
    array([14, 77])
    """
    return np.einsum("...i,...i", arr, arr)


# Faster to not dispatch back to normSq
def norm(arr):
    """
    Compute the Euclidean norm of an array over the last axis while preserving leading axes.

    Parameters:
    -----------
    arr : array-like
        Input array where the Euclidean norm is computed along the last axis. 
        The array can have any shape, with the last axis representing the vector components.

    Returns:
    --------
    norm : array-like
        The Euclidean norm of the input array computed along the last axis. 
        The output shape matches the input shape, excluding the last axis.

    Notes:
    ------
    - This implementation directly computes the norm using `np.sqrt` and `np.einsum`, 
      avoiding the intermediate step of calling `normSq`. This improves performance.
    - The computation uses `np.einsum("...i,...i", arr, arr)` to efficiently sum the squared 
      components along the last axis, followed by taking the square root.
    - Faster than using `np.linalg.norm()` for this specific use case and generalizes well 
      to arrays of arbitrary shape.

    Example:
    --------
    >>> import numpy as np
    >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
    >>> norm(arr)
    array([3.74165739, 8.77496439])
    """
    return np.sqrt(np.einsum("...i,...i", arr, arr))


# Faster to not dispatch back to norm
def normed(arr):
    """
    Normalize an array along the last axis to have unit length.

    This function computes the normalized version of the input array by dividing each vector 
    along the last axis by its Euclidean norm. The normalization ensures that the resulting 
    vectors have a magnitude of 1.

    Parameters:
    -----------
    arr : array-like
        Input array where normalization is applied along the last axis. 
        The array can have any shape, with the last axis representing the vector components.

    Returns:
    --------
    normalized : array-like
        The normalized array with unit-length vectors along the last axis. 
        The output shape matches the input shape.
    """
    return arr / np.sqrt(np.einsum("...i,...i", arr, arr))[..., None]


def einsum_norm(a, indices='ij,ji->i'):
    """
    Compute the norm of an array using Einstein summation notation with customizable indices.

    This function calculates the norm of an input array using `np.einsum` with user-defined 
    summation indices. By default, it computes the Euclidean norm along the specified axes 
    using the provided summation pattern. The result is the square root of the summation 
    output.

    Parameters:
    -----------
    a : array-like
        Input array for which the norm is computed. The array can have any shape, and the 
        summation pattern determines how the norm is calculated across its dimensions.
    indices : str, optional
        A string representing the Einstein summation pattern. The default value is 
        `'ij,ji->i'`, which computes the norm along specific axes. Users can customize 
        this pattern to suit their needs.

    Returns:
    --------
    norm : array-like
        The computed norm of the input array based on the specified summation pattern. 
        The output shape depends on the summation indices provided.
    """
    return np.sqrt(np.einsum(indices, a, a))


def unitAngle3(r1, r2):
    """Robustly compute angle between unit vectors r1 and r2.
    Vectorized for multiple triplets.
    """
    r1, r2 = np.broadcast_arrays(r1, r2)
    dsq = normSq(r1 - r2)
    out = 2 * np.arcsin(0.5 * np.sqrt(dsq))
    # Following will almost never be the case.  Exception is when r1, r2 are
    # nearly antipodal.
    w = dsq < 3.99
    if not np.all(w):
        try:
            out[~w] = np.pi - np.arcsin(norm(np.cross(r1[~w], r2[~w])))
        except (TypeError, IndexError):
            out = np.pi - np.arcsin(norm(np.cross(r1, r2)))
    return out


def cluster_emcee_walkers(
    chain, lnprob, lnprior, thresh_multiplier=1, verbose=False
):
    """
    Down-select emcee walkers to those with the largest posterior mean.

    Follows the algorithm of Hou, Goodman, Hogg et al. (2012), An affine-invariant
    sampler for exoplanet fitting and discovery in radial velocity data.
    The Astrophysical Journal, 745(2), 198.
    """
    import emcee
    from distutils.version import LooseVersion
    if LooseVersion(emcee.__version__) < LooseVersion("3.0"):
        raise ValueError("emcee version at least 3.0rc2 required")
    if verbose:
        out = "Clustering emcee walkers with threshold multiplier {:3.2f}"
        out = out.format(thresh_multiplier)
        print(out)
    chain = np.array(chain)
    lnprob = np.array(lnprob)
    # ## lnprob.shape == (Nsteps, Nwalkers) => lk.shape == (Nwalkers,)
    lk = -np.mean(np.array(lnprob), axis=0)
    nwalkers = len(lk)
    ndx = np.argsort(lk)
    lks = lk[ndx]
    d = np.diff(lks)
    thresh = np.cumsum(d) / np.arange(1, nwalkers)
    selection = d > (thresh_multiplier * thresh)
    if np.any(selection):
        nkeep = np.argmax(selection)
    else:
        nkeep = nwalkers
    if verbose:
        print("chain, lnprob:", chain.shape, lnprob.shape)
    chain = chain[:, ndx[0:nkeep], :]
    lnprob = lnprob[:, ndx[0:nkeep]]
    lnprior = lnprior[:, ndx[0:nkeep]]
    if verbose:
        print("New chain, lnprob:", chain.shape, lnprob.shape)
    return chain, lnprob, lnprior


def subsample_high_lnprob(chain, lnprob, lnprior, nSample, thresh=-10):
    """Select MCMC samples with probabilities above some relative threshold

    Parameters
    ----------
    chain : array_like, (nWalker, nStep, 6)
        Input MCMC chain from EmceeSampler.sample()
    lnprob : array_like, (nWalker, nStep)
        Input MCMC lnprob from EmceeSampler.sample()
    lnprior : array_like, (nWalker, nStep)
        Input MCMC lnprior from EmceeSampler.sample()
    nSample : int
        Number of samples to return.  If fewer than nSample samples
        exceed threshold, then return the nSample most probable
        samples.
    thresh : float, optional
        Threshold w.r.t. max(lnprob) below which to exclude samples.

    Returns
    -------
    samples : array_like, (nSample, 6)
        Output samples
    """
    chain = chain.reshape(-1, 6)
    lnprob = lnprob.reshape(-1)
    lnprior = lnprior.reshape(-1)
    asort = np.argsort(-lnprob)
    lnprobmax = lnprob[asort[0]]
    goodLnProbThresh = lnprobmax + thresh
    nGood = np.searchsorted(-lnprob, -goodLnProbThresh, sorter=asort)
    nGood = max(nGood, nSample)
    permutation = np.random.permutation(nGood)
    s = asort[permutation[:nSample]]
    return chain[s], lnprob[s], lnprior[s]


def resample(particles, ln_weights, obs_times=None, pod=False):
    """
    Resample particles to achieve more uniform weights.
    This produces the same number of particles as are input (i.n. dimension
    matches "particles")
    From "Bayesian Multiple Target Tracking" (2nd ed.) p. 100

    :param particles: 2D array of particles. Each row is an nD particles
    :type particles: numpy.ndarray

    :param ln_weights: 1D array of particle log-weights. Length should match
        number of particles
    :type ln_weights: numpy.ndarray

    :param obs_times: Time at which observation occured. This is only used if
        pod=True, defaults to None
    :type obs_times: [type], optional

    :param pod: Specify whether this is the special case of regularizing orbit
        parameters, defaults to False
    :type pod: bool, optional

    :return: Resampled particles and weights (same number as input)
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    import bisect
    num_particles = particles.shape[0]
    # weights /= np.sum(weights)
    weights = get_normed_weights(ln_weights)

    # 1.
    cumul_weights = np.cumsum(weights)

    # 2.
    # This is cumulative weights for a new set of particles with equal weights
    cumul_new_weights = np.arange(num_particles, dtype=float) / num_particles
    # The offset of the first value sets the sampling used to match the
    # cumulative weight ranges of the old particles to the new particles
    new_weight_offset = np.random.uniform(high=1. / num_particles)
    cumul_new_weights += new_weight_offset

    # 3.
    # For j=1:J, do the following
    # For m such that C_{j-1} <= u_m <= C_j, set xi^m = x^j
    # Select the new particle values based on the weights of the original
    # particles. Particles with large weight are repeated, while particles with
    # small weights may be eliminated.
    new_particle_inds = [bisect.bisect_left(cumul_weights, x)
                         for x in cumul_new_weights]
    resampled_states = particles[new_particle_inds, ]

    # 4. Perturb states (i.e., regularization)
    if pod:
        # Special case: regularization for preliminary orbit determination
        # For now, we're ignoring this and doing it the same way
        reg_delta, reg_weights = regularize_default(particles, weights)
        # 5. Redefine weights to uniform distribution
        new_particles = resampled_states
        new_particles[:, 0:6] = resampled_states[:, 0:6] + reg_delta
    else:
        reg_delta, reg_weights = regularize_default(particles, weights)
        # 5. Redefine weights to uniform distribution
        new_particles = resampled_states + reg_delta

    return new_particles, reg_weights


def get_normed_weights(ln_weights):
    """
    Computes normalized weights from log-weights.

    Parameters:
        ln_weights (ndarray): Logarithmic weights.

    Returns:
        ndarray: Normalized weights summing to 1.
    """
    ln_wts_norm = np.logaddexp.reduce(ln_weights)
    weights = np.exp(ln_weights - ln_wts_norm)
    return weights


def regularize_default(particles, weights, num_particles_out=None, dimension=6):
    """
    Perform particle regularization. This generates a perturbation from the
    particles' original values to prevent singularity issues.
    From "Bayesian Multiple Target Tracking" (2nd ed.) p. 101-102

    :param particles: Particles to reqularize. Each row is an nD particle
    :type particles: numpy.ndarray

    :param weights: 1D array of particle weight. Should be same length as number
         of particles.
    :type weights: numpy.ndarray

    :param num_particles_out: Number of particles to return, defaults to None.
        If not specified, will return the same number of particles as in the
        input particles.
    :type num_particles_out: int | None, optional

    :param dimension: Dimension of the parameter space for resampling. Assumes
        the first `dimension` columns of `particles` are the parameters to use.
        Any remaining columns are carried through without modification (e.g.,
        time columns). Default: 6
    :type dimension: int, optional

    :return: Deltas from original particles and their weights
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    num_particles_in, dim_in = particles.shape
    if num_particles_out is None:
        num_particles_out = num_particles_in
    kernel_cov = get_kernel_cov(particles[:, 0:dimension], weights)
    # print("kernel_cov:", kernel_cov)
    window_size = (4. / (num_particles_in * (dimension + 1))) ** (1. / (dimension + 4.))
    # Use a smaller window size for multi-modal distributions
    adj_windows_size = window_size / 2.
    # Generate deltas of new particles from a normal distribution with zero mean
    # and covariance according to the input particles
    delta = np.random.multivariate_normal(mean=np.zeros(dimension),
                                          cov=kernel_cov * (adj_windows_size**2),
                                          size=num_particles_out)
    weights_out = np.ones(num_particles_out) / float(num_particles_out)
    return delta, weights_out


def get_kernel_cov(kernel_mat, weights):
    """
    Get the covariance matrix of kernel_mat. This a wrapper for numpy's cov

    :param kernel_mat: Data matrix
    :type kernel_mat: numpy.ndarray

    :param weights: 1D array of weights of the data
    :type weights: numpy.ndarray

    :return: numpy.ndarray
    :rtype: Covariance matrix
    """
    return np.cov(kernel_mat.transpose(), aweights=weights, bias=True)


class LRU_Cache:
    """Simplified Least Recently Used Cache.

    Mostly stolen from http://code.activestate.com/recipes/577970-simplified-lru-cache/,
    but added a method for dynamic resizing.  The least recently used cached item is
    overwritten on a cache miss.

    Parameters:
        user_function:  A python function to cache.
        maxsize:        Maximum number of inputs to cache.  [Default: 1024]

    Example::

        >>> def slow_function(*args) # A slow-to-evaluate python function
        >>>    ...
        >>>
        >>> v1 = slow_function(*k1)  # Calling function is slow
        >>> v1 = slow_function(*k1)  # Calling again with same args is still slow
        >>> cache = LRU_Cache(slow_function)
        >>> v1 = cache(*k1)  # Returns slow_function(*k1), slowly the first time
        >>> v1 = cache(*k1)  # Returns slow_function(*k1) again, but fast this time.
    """
    def __init__(self, user_function, maxsize=1024):
        if maxsize < 0:
            raise ValueError("Invalid maxsize", maxsize)

        # Link layout:     [PREV, NEXT, KEY, RESULT]
        self.root = root = [None, None, None, None]
        self.user_function = user_function
        self.cache = cache = {}

        last = root
        for i in range(maxsize):
            key = object()
            cache[key] = last[1] = last = [last, root, key, None]
        root[0] = last

    def __call__(self, *key):
        if len(self.cache) == 0:  # useful to allow zero when profiling...
            return self.user_function(*key)
        cache = self.cache
        root = self.root
        link = cache.get(key)
        if link is not None:
            # Cache hit: move link to last position
            # print("hit")
            link_prev, link_next, _, result = link
            link_prev[1] = link_next
            link_next[0] = link_prev
            last = root[0]
            last[1] = root[0] = link
            link[0] = last
            link[1] = root
            return result
        # Cache miss: evaluate and insert new key/value at root, then increment root
        #             so that just-evaluated value is in last position.
        # print("miss")
        result = self.user_function(*key)
        root = self.root  # re-establish root in case user_function modified it due to recursion
        root[2] = key
        root[3] = result
        oldroot = root
        root = self.root = root[1]
        root[2], oldkey = None, root[2]
        root[3], _ = None, root[3]
        del cache[oldkey]
        cache[key] = oldroot
        return result

    def resize(self, maxsize):
        """Resize the cache.

        Increasing the size of the cache is non-destructive, i.e., previously cached inputs remain
        in the cache.  Decreasing the size of the cache will necessarily remove items from the
        cache if the cache is already filled.  Items are removed in least recently used order.

        Parameters:
            maxsize:    The new maximum number of inputs to cache.
        """
        oldsize = len(self.cache)
        if maxsize == oldsize:
            return
        else:
            root = self.root
            cache = self.cache
            if maxsize < 0:
                raise ValueError("Invalid maxsize", maxsize)
            if maxsize < oldsize:
                for i in range(oldsize - maxsize):
                    # Delete root.next
                    current_next_link = root[1]
                    new_next_link = root[1] = root[1][1]
                    new_next_link[0] = root
                    del cache[current_next_link[2]]
            else:  # maxsize > oldsize:
                for i in range(maxsize - oldsize):
                    # Insert between root and root.next
                    key = object()
                    cache[key] = link = [root, root[1], key, None]
                    root[1][0] = link
                    root[1] = link


def catalog_to_apparent(
    ra, dec, t, observer=None, pmra=0, pmdec=0, parallax=0,
    skipAberration=False
):
    """
    Convert ra/dec of stars from catalog positions (J2000/ICRS) to apparent
    positions, correcting for proper motion, parallax, annual and diurnal
    aberration.

    Parameters
    ----------
    ra : array_like
        J2000 right ascension in radians.
    dec : array_like
        J2000 declination in radians.
    t : float or astropy.time.Time
        If float, then should correspond to GPS seconds; i.e., seconds since
        1980-01-06 00:00:00 UTC
    observer : Observer, optional
        Observer to use for diurnal aberration correction.  If None, then no
        diurnal aberration correction is performed.
    pmra, pmdec : array_like, optional
        proper motion in right ascension / declination, in milliarcsec per year
    parallax : array_like, optional
        annual parallax in arcseconds
    skipAberration : bool, optional
        Don't apply aberration correction.  Mostly useful during testing...

    Returns
    -------
    ara, adec : array_like
        Apparent RA and dec in radians
    """
    from astropy.coordinates import get_body_barycentric_posvel
    from astropy.time import Time
    import astropy.units as u

    if isinstance(t, Time):
        tTime = t
        t = t.gps
    else:
        tTime = Time(t, format='gps')

    # Code below modeled after SOFA iauPmpx, iauAb routines, but a bit simpler
    # since SSA requirements are only ~arcsec or so.
    # aulty = (u.au / (299792458 * u.m / u.s)).to(u.year).value
    # SOFA is the Standards of Fundamental Astronomy.
    obsPos, obsVel = get_body_barycentric_posvel('earth', tTime)
    pob = obsPos.xyz.to(u.AU).value
    vob = obsVel.xyz.to(u.m / u.s).value
    sr, cr = np.sin(ra), np.cos(ra)
    sd, cd = np.sin(dec), np.cos(dec)
    x = cr * cd
    y = sr * cd
    z = sd
    p = np.array([x, y, z]).T
    dt = (t - Time("J2000").gps) / (86400 * 365.25)
    pdrad = np.deg2rad(pmdec / (1000 * 3600))
    prrad = np.deg2rad(pmra / (1000 * 3600)) / cd
    pxrad = np.deg2rad(parallax / 3600)

    # Proper motion and parallax
    pdz = z * pdrad
    pm = np.array([
        -prrad * y - pdz * cr,
        prrad * x - pdz * sr,
        pdrad * cd
    ]).T
    p += dt * pm - pxrad[..., None] * pob
    p = normed(p)

    # Aberration
    if not skipAberration:
        if observer is not None:
            _, dvob = observer.getRV(tTime)
            vob += dvob
        p += vob / 299792458.
        p = normed(p)

    ra = np.arctan2(p[:, 1], p[:, 0])
    dec = np.arcsin(p[:, 2])
    return ra, dec


def rv_to_ntw(r, v, rcoord):
    """Convert coordinates to NTW coordinates, using r, v to define NTW system.

    T gives the projection of (rcoord - r) along V  (tangent to track)
    W gives the projection of (rcoord - r) along (V cross r) (normal to plane)
    N gives the projection of (rcoord - r) along (V cross (V cross r))
        (in plane, perpendicular to T)

    Parameters
    ----------
    r : array_like (n, 3)
        central positions defining coordinate system
    v : array_like (n, 3)
        velocity defining coordinate system
    rcoord : array_like (n, 3)
        positions to transform to NTW coordinates

    Returns
    -------
    ntw : array_like (n, 3)
        n, t, w positions
    """
    def dot(x, y):
        return np.einsum('...i, ...i', x, y)
    dr = rcoord - r
    t = dot(normed(v), dr)
    w = dot(normed(np.cross(r, v)), dr)
    n = dot(normed(np.cross(v, np.cross(r, v))), dr)
    return np.array([n, t, w]).T.copy()


def ntw_to_r(r, v, ntw, relative=False):
    """Convert NTW coordinates to cartesian coordinates, using r, v to define
    NTW system.

    T gives the projection of (rcoord - r) along V  (tangent to track)
    W gives the projection of (rcoord - r) along (V cross r) (normal to plane)
    N gives the projection of (rcoord - r) along (V cross (V cross r))
        (in plane, perpendicular to T)

    Parameters
    ----------
    r : array_like (n, 3)
        central positions defining coordinate system
    v : array_like (n, 3)
        velocity defining coordinate system
    ntw : array_like (n, 3)
        ntw coordinates to transform to cartesian coordinates
    relative : bool
        if True, just rotate the NTW coordinates to Cartesian; do not offset
        the origin so that NTW = 0 -> Cartesian r.

    Returns
    -------
    r : array_like (n, 3)
        cartesian x, y, z coordinates
    """
    tvec = normed(v)
    wvec = normed(np.cross(r, v))
    nvec = normed(np.cross(tvec, wvec))
    mat = np.array([nvec, tvec, wvec])
    ret = np.dot(ntw, mat)
    if not relative:
        ret += r
    return ret


def lb_to_tp(lb, b):
    """Convert lb-like coordinates to theta-phi like coordinates.

    Here 'theta-phi' coordinates refers to the system where theta is the
    angle between zenith and the point in question, and phi is the
    corresponding azimuthal angle.

    This just sets theta = pi - b and renames lb -> phi.  Everything is in
    radians.
    """
    return np.pi / 2 - b, lb % (2 * np.pi)


def tp_to_lb(t, p):
    """Convert theta-phi-like coordinates to lb-like coordinates.

    Here 'theta-phi' coordinates refers to the system where theta is the
    angle between zenith and the point in question, and phi is the
    corresponding azimuthal angle.

    This just sets b = pi - theta and renames phi -> lb.  Everything is in
    radians.
    """
    return p % (2 * np.pi), np.pi / 2 - t


def tp_to_unit(t, p):
    """Convert theta-phi-like coordinates to unit vectors.

    Here 'theta-phi' coordinates refers to the system where theta is the
    angle between zenith and the point in question, and phi is the
    corresponding azimuthal angle.

    Everything is in radians.
    """
    z = np.cos(t)
    x = np.cos(p) * np.sin(t)
    y = np.sin(p) * np.sin(t)
    return np.concatenate([q[..., np.newaxis] for q in (x, y, z)], axis=-1)


def lb_to_unit(r, d):
    """Convert lb-like coordinates to unit vectors.

    Everything is in radians.
    """
    return tp_to_unit(*lb_to_tp(r, d))


def unit_to_tp(unit):
    """Convert unit vectors to theta-phi-like coordinates.

    Here 'theta-phi' coordinates refers to the system where theta is the
    angle between zenith and the point in question, and phi is the
    corresponding azimuthal angle.

    Everything is in radians.
    """
    norm = np.sqrt(np.sum(unit**2., axis=-1))
    unit = unit / norm[..., None]
    t = np.arccos(unit[..., 2])
    p = np.arctan2(unit[..., 1], unit[..., 0])
    return t, p % (2 * np.pi)


def xyz_to_tp(x, y, z):
    """Convert x, y, z vectors to theta-phi-like coordinates.

    Here 'theta-phi' coordinates refers to the system where theta is the
    angle between zenith and the point in question, and phi is the
    corresponding azimuthal angle.

    Everything is in radians.
    """
    norm = np.sqrt(x**2 + y**2 + z**2)
    t = np.arccos(z / norm)
    p = np.arctan2(y / norm, x / norm)
    return t, p % (2 * np.pi)


def unit_to_lb(unit):
    """Convert unit vectors to lb-like coordinates.

    Everything is in radians.
    """
    return tp_to_lb(*unit_to_tp(unit))


def xyz_to_lb(x, y, z):
    """Convert x, y, z vectors to lb-like coordinates.

    Everything is in radians.
    """
    return tp_to_lb(*xyz_to_tp(x, y, z))


def lb_to_tan(lb, b, mul=None, mub=None, lcen=None, bcen=None):
    """Convert lb-like coordinates & proper motions to orthographic tangent plane.

    All units are in radians.  If mul is None (default), transformed
    proper motions will not be returned.  The tangent plane is always chosen
    so that +Y is towards b = 90, and +X is towards +lb.

    Parameters
    ----------
    lb : array_like (n)
        right ascension of point
    b : array_like (n)
        declination of point
    mul : array_like (n)
        proper motion in ra of point (arbitrary units)
        rate of change in lb is mul / np.cos(b); i.e., length of proper motion
        vector on sphere is np.hypot(mul, mub)
    mub : array_like (n)
        proper motion in dec of point (arbitrary units)
    lcen : array_like (n)
        right ascension to use for center of tangent plane
        if None, use spherical mean of (lb, b)
    bcen : array_like (n)
        declination to use for center of tangent plane

    Returns
    -------
    if mul is None, (x, y) otherwise (x, y, vx, vy)
    x : array_like (n)
        x coordinate of tangent plane projection of lb, b
    y : array_like (n)
        y coordinate of tangent plane projection of lb, b
    vx : array_like (n)
        x coordinate of tangent plane projection of (mul, mub)
    vy : array_like (n)
        y coordinate of tangent plane projection of (mul, mub)
    """
    up = np.array([0, 0, 1])
    unit = lb_to_unit(lb, b)
    if lcen is None:
        lcen, bcen = unit_to_lb(np.mean(unit, axis=0).reshape(1, -1))
    unitcen = lb_to_unit(lcen, bcen)
    if len(unitcen.shape) == 1:
        unitcen = unitcen[None, :]
    rahat = np.cross(up, unitcen)
    m = norm(rahat) < 1e-10
    rahat[m, :] = np.array([1, 0, 0])[None, :]
    rahat /= np.sqrt(np.sum(rahat**2, axis=1, keepdims=True))
    dechat = np.cross(unitcen, rahat)
    dechat /= np.sqrt(np.sum(dechat**2, axis=1, keepdims=True))
    xx = np.sum(rahat * unit, axis=1)
    yy = np.sum(dechat * unit, axis=1)
    res = (xx, yy)
    if mul is not None:
        rahat2 = np.cross(up, unit)
        rahat2 /= np.sqrt(np.sum(rahat2**2, axis=1, keepdims=True))
        dechat2 = np.cross(unit, rahat2)
        dechat2 /= np.sqrt(np.sum(dechat2**2, axis=1, keepdims=True))
        vv = mul[:, None] * rahat2 + mub[:, None] * dechat2
        vx = np.sum(rahat * vv, axis=1)
        vy = np.sum(dechat * vv, axis=1)
        rr = np.hypot(xx, yy)
        m = np.abs(rr) > 1e-9
        vr = (vx[m] * xx[m] + vy[m] * yy[m]) / rr[m]
        va = (vy[m] * xx[m] - vx[m] * yy[m]) / rr[m]
        vr /= np.sum(unitcen[m, :] * unit[m, :], axis=1)
        vx[m] = vr * xx[m] / rr[m] - va * yy[m] / rr[m]
        vy[m] = vr * yy[m] / rr[m] + va * xx[m] / rr[m]
        res = res + (vx, vy)
    return res


def tan_to_lb(xx, yy, lcen, bcen):
    """Convert orthographic tangent plane coordinates to lb coordinates.

    Parameters
    ----------
    xx : array_like (n)
        tangent plane x coordinate (radians)
    yy : array_like (n)
        targent plane y coordinate (radians
    lcen : float, array_like (n)
        right ascension of center of tangent plane
    bcen : float, array_linke (n)
        declination of center of tangent plane

    Returns
    -------
    lb : array_like (n)
        right ascension corresponding to xx, yy
    b : array_like (n)
        declination corresponding to xx, yy
    """
    unitcen = lb_to_unit(lcen, bcen)
    if len(unitcen.shape) == 1:
        unitcen = unitcen[None, :]
    up = np.array([0, 0, 1])
    rahat = np.cross(up, unitcen)
    m = norm(rahat) < 1e-10
    rahat[m, :] = np.array([1, 0, 0])[None, :]
    rahat /= np.sqrt(np.sum(rahat**2, axis=1, keepdims=True))
    dechat = np.cross(unitcen, rahat)
    dechat /= np.sqrt(np.sum(dechat**2, axis=1, keepdims=True))
    xcoord = xx
    ycoord = yy
    zcoord = np.sqrt(1 - xcoord**2 - ycoord**2)
    unit = (xcoord.reshape(-1, 1) * rahat + ycoord.reshape(-1, 1) * dechat + zcoord.reshape(-1, 1) * unitcen)
    return unit_to_lb(unit)


def sample_points(x, C, npts, sqrt=False):
    """Sample points around x according to covariance matrix.

    Parameters
    ----------
    x : array_like (n)
        point to sample around
    C : array_like (n, n)
        covariance matrix corresponding to x
    npts : int
        number of points to sample
    sqrt : bool
        use sqrt(C) rather than an SVD.  The SVD is often more stable.

    Returns
    -------
    Gaussian samples around x corresponding to covariance matrix C
    """
    n = C.shape[0]
    xa = np.random.randn(npts, n)
    if not sqrt:
        sqrtdiag = np.sqrt(np.diag(C))
        scalecovar = sqrtdiag[None, :] * sqrtdiag[:, None]
        uu, ss, vvh = np.linalg.svd(C / scalecovar)
        if np.any(ss < 0):
            raise ValueError('negative eigenvalues in C!')
        sqrtC = uu.dot(np.diag(ss**0.5)).dot(vvh)
    else:
        sqrtC = C
        sqrtdiag = 1
    xa = sqrtC.dot(xa.T).T
    if not sqrt:
        xa *= sqrtdiag[None, :]
    xa += x[None, :]
    return xa


def sigma_points(f, x, C, scale=1, fixed_dimensions=None):
    """Compute f(sigma points) for sigma points of C around x.

    There are many possible definitions of sigma points.  This one
    takes the singular value decomposition of C and uses
    the eigenvectors times sqrt(dimension)*scale*(+/-1) as the sigma points.
    It then evaluates the given function f at x plus those sigma points.

    Parameters
    ----------
    f : function
        the function to evaluate at the sigma points
    x : array_like (n)
        the central value to evaluate the function around
    C : array_like (n, n)
        the covariance matrix corresponding to x
    scale : float
        return scale-sigma points rather than n-sigma points.  e.g.,
        for 5 sigma, set scale = 5.
    fixed_dimensions : array_like, (n), bool
        boolean array specifying dimensions of x that are fixed
        and not specified in C
    """
    if fixed_dimensions is None:
        fixed = np.zeros(len(x), dtype='bool')
    else:
        fixed = np.array(fixed_dimensions).astype('bool') is not False
    free = ~fixed
    # sqrtC = linalg.sqrtm(C)
    sqrtdiag = np.sqrt(np.diag(C))
    scalecovar = sqrtdiag.reshape(1, -1) * sqrtdiag.reshape(-1, 1)
    uu, ss, vvh = np.linalg.svd(C / scalecovar)
    sqrtC = uu.dot(np.diag(ss**0.5)).dot(vvh)
    n = C.shape[0]
    xsigma = [x[free]] + [x[free] + np.sqrt(n) * sgn * scale * sqrtCvec * sqrtdiag
                          for sqrtCvec in sqrtC for sgn in [-1, 1]]
    xsigma = np.array(xsigma)
    xsigmaall = np.zeros((xsigma.shape[0], xsigma.shape[1] + np.sum(fixed)),
                         dtype=xsigma.dtype)
    xsigmaall[:, free] = xsigma
    xsigmaall[:, fixed] = x[None, fixed]
    if f is not None:
        fsigma = f(xsigmaall)
    else:
        fsigma = xsigmaall
    return fsigma


def unscented_transform_mean_covar(f, x, C, scale=1):
    """Compute mean and covariance matrix using unscented transform given a
    transformation f, a point x, and a covariance C.

    This uses the sigma point convention from sigma_points.  It assumes that
    f(sigma_points)[i] is f evaluated at the ith sigma point.  If f does
    not obey this convention, this function will produce undefined results.

    Parameters
    ----------
    f : function
        the function to evaluate at the sigma points.
    x : array_like (n)
        the central value to evaluate the function around
    C : array_like (n, n)
        the covariance matrix corresponding to x
    scale : float
        return scale-sigma points rather than n-sigma points.  e.g.,
        for 5 sigma, set scale = 5.
    """
    fsigma = sigma_points(f, x, C, scale=scale)
    mean = fsigma[0]
    dmean = fsigma[1:, ...] - mean[:, ...]
    covar = np.cov(dmean, ddof=0)
    return mean, covar


def _gpsToTT(t):
    """
    Convert GPS time to Terrestrial Time (TT).

    Parameters:
    ----------
    t : Time or float
        If `t` is an instance of `astropy.time.Time`, it is assumed to represent GPS time in the form of an Astropy Time object.
        If `t` is a float, it is assumed to be GPS time in seconds.

    Returns:
    -------
    float
        The equivalent time in Terrestrial Time (TT) expressed in days since the GPS epoch (January 6, 1980).

    Notes:
    -----
    The conversion involves adding a constant offset of 51.184 seconds to the GPS time,
    then converting the result into days by dividing by 86400 (the number of seconds in a day).
    The epoch for GPS is represented as a Modified Julian Date (MJD) of 44244.0.
    """
    if isinstance(t, Time):
        t = t.gps
    return 44244.0 + (t + 51.184) / 86400


def sunPos(t, fast=True):
    """Compute GCRF position of the sun.

    Parameters
    ----------
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC
    fast : bool
        Use fast approximation?

    Returns
    -------
    r : array_like (n)
        position in meters
    """
    if isinstance(t, Time):
        t = t.gps
    if fast:
        # MG section 3.3.2
        T = (_gpsToTT(t) - 51544.5) / 36525.0
        M = 6.239998880168239 + 628.3019326367721 * T
        lam = (4.938234585592756 + M + 0.03341335890206922 * np.sin(M) + 0.00034906585039886593 * np.sin(2 * M))
        rs = (149.619 - 2.499 * np.cos(M) - 0.021 * np.cos(2 * M)) * 1e9
        obliquity = 0.40909280420293637
        co, so = np.cos(obliquity), np.sin(obliquity)
        cl, sl = np.cos(lam), np.sin(lam)
        r = rs * np.array([cl, sl * co, sl * so])
    else:
        pvh, _ = erfa.epv00(2400000.5, _gpsToTT(t))
        r = pvh['p'] * -149597870700  # AU -> m
    return r


def moonPos(t):
    """Compute GCRF position of the moon.

    Parameters
    ----------
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC

    Returns
    -------
    r : array_like (n)
        position in meters
    """
    if isinstance(t, Time):
        t = t.gps
    # MG section 3.3.2
    T = (_gpsToTT(t) - 51544.5) / 36525.0
    # fundamental arguments (3.47)
    L0 = 3.810335976843669 + 8399.684719711557 * T
    lb = 2.3555473221057053 + 8328.69142518676 * T
    lp = 6.23999591310851 + 628.3019403162209 * T
    D = 5.198467889454092 + 7771.377143901714 * T
    F = 1.6279179861529427 + 8433.46617912181 * T
    # moon longitude (3.48)
    dL = (22640 * np.sin(lb) + 769 * np.sin(2 * lb) - 4586 * np.sin(lb - 2 * D) + 2370 * np.sin(2 * D) - 668 * np.sin(lp) - 412 * np.sin(2 * F) - 212 * np.sin(2 * lb - 2 * D) - 206 * np.sin(lb + lp - 2 * D) + 192 * np.sin(lb + 2 * D) - 165 * np.sin(lp - 2 * D) + 148 * np.sin(lb - lp) - 125 * np.sin(D) - 110 * np.sin(lb + lp) - 55 * np.sin(2 * F - 2 * D))

    L = L0 + np.deg2rad(dL / 3600)
    # moon latitude (3.49)
    beta = np.deg2rad((
        18520 * np.sin(F + L - L0 + np.deg2rad((412 * np.sin(2 * F) + 541 * np.sin(lp)) / 3600)) - 526 * np.sin(F - 2 * D) + 44 * np.sin(lb + F - 2 * D) - 31 * np.sin(-lb + F - 2 * D) - 25 * np.sin(-2 * lb + F) - 23 * np.sin(lp + F - 2 * D) + 21 * np.sin(-lb + F) + 11 * np.sin(-lp + F - 2 * D)
    ) / 3600)
    # moon distance (3.50)
    r = (
        385000 - 20905 * np.cos(lb) - 3699 * np.cos(2 * D - lb) - 2956 * np.cos(2 * D) - 570 * np.cos(2 * lb) + 246 * np.cos(2 * lb - 2 * D) - 205 * np.cos(lp - 2 * D) - 171 * np.cos(lb + 2 * D) - 152 * np.cos(lb + lp - 2 * D)
    ) * 1e3
    r_ecliptic = r * np.array([
        np.cos(L) * np.cos(beta),
        np.sin(L) * np.cos(beta),
        np.sin(beta)
    ])
    obliquity = 0.40909280420293637
    co, so = np.cos(obliquity), np.sin(obliquity)
    rot = np.array([[1, 0, 0], [0, co, so], [0, -so, co]])
    return rot.T @ r_ecliptic


def iers_interp(t):
    """Interpolate IERS values

    Parameters
    ----------
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC

    Returns
    -------
    dut1: array_like (n)
        Time difference UT1 - TT in days.
    pmx, pmy : array_like (n)
        Polar motion values in radians.
    """
    from astropy.utils import iers
    from scipy.interpolate import interp1d
    from astropy import units as u
    if isinstance(t, Time):
        t = t.gps
    if not hasattr(iers_interp, '_interp'):
        table = iers.earth_orientation_table.get()
        ts = Time(table['MJD'], format='mjd', scale='utc')
        tgps = ts.gps
        dut1tt = ts.ut1.mjd - ts.tt.mjd
        pmx = table['PM_x'].to(u.rad).value
        pmy = table['PM_y'].to(u.rad).value
        iers_interp._interp = interp1d(
            tgps,
            np.array([dut1tt, pmx, pmy]),
            bounds_error=False,
            fill_value=(
                np.array([dut1tt[0], pmx[0], pmy[0]]),
                np.array([dut1tt[-1], pmx[-1], pmy[-1]])
            )
        )
    return iers_interp._interp(t)


def gcrf_to_teme(t):
    """ Return the rotation matrix that converts GCRS cartesian coordinates
    to TEME cartesian coordinates.

    Parameters
    ----------
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC

    Returns
    -------
    rot : array (3,3)
        Rotation matrix to apply to GCRS to yield TEME.
    """
    if isinstance(t, Time):
        t = t.gps
    d_ut1_tt_mjd, _, _ = iers_interp(t)
    mjd_tt = _gpsToTT(t)
    ut1 = mjd_tt + d_ut1_tt_mjd
    gst = erfa.gmst82(2400000.5, ut1)
    era = erfa.era00(2400000.5, ut1)
    c2i = erfa.c2i00b(2400000.5, mjd_tt)
    return erfa.rxr(erfa.rv2m([0, 0, era - gst]), c2i)


def teme_to_gcrf(t):
    """ Return the rotation matrix that converts TEME cartesian coordinates
    to GCRS cartesian coordinates.

    Parameters
    ----------
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC

    Returns
    -------
    rot : array (3,3)
        Rotation matrix to apply to TEME to yield GCRS.
    """
    return erfa.tr(gcrf_to_teme(t))



# VECTOR FUNCTIONS FOR COORDINATE MATH
def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def get_angle(a, b, c):
    """
    Calculate the angle between two vectors where b is the vertex of the angle.

    This function computes the angle between vectors `ba` and `bc`, where `b` is the vertex and `a` and `c` are the endpoints of the angle.

    Parameters:
    ----------
    a : (n, 3) numpy.ndarray
        Array of coordinates representing the first vector.
    b : (n, 3) numpy.ndarray
        Array of coordinates representing the vertex of the angle.
    c : (n, 3) numpy.ndarray
        Array of coordinates representing the second vector.

    Returns:
    -------
    numpy.ndarray
        Array of angles (in radians) between the vectors `ba` and `bc`.

    Notes:
    ------
    - The function handles multiple vectors by using broadcasting.
    - The angle is calculated using the dot product formula and the arccosine function.

    Example:
    --------
    >>> a = np.array([[1, 0, 0]])
    >>> b = np.array([[0, 0, 0]])
    >>> c = np.array([[0, 1, 0]])
    >>> get_angle(a, b, c)
    array([1.57079633])
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c)
    ba = np.subtract(a, b)
    bc = np.subtract(c, b)
    cosine_angle = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    return np.arccos(cosine_angle)


def dms_to_rad(coords):
    """
    Convert coordinates from degrees, minutes, and seconds (DMS) format to radians.

    Parameters:
    -----------
        coords (str, list, or tuple): 
            - A single DMS coordinate as a string (e.g., "12d34m56s").
            - A list or tuple of DMS coordinates as strings.

    Returns:
    --------
        float or list: 
            - If a single coordinate is provided, returns its value in radians.
            - If multiple coordinates are provided (list or tuple), returns a list 
              of values in radians.

    Example:
    --------
        dms_to_rad("12d34m56s") -> 0.219725
        dms_to_rad(["12d34m56s", "45d67m89s"]) -> [0.219725, 0.798488]
    """
    from astropy.coordinates import Angle
    if isinstance(coords, (list, tuple)):
        return [Angle(coord).radian for coord in coords]
    else:
        return Angle(coords).radian
    return


def dms_to_deg(coords):
    """
    Convert coordinates from degrees, minutes, and seconds (DMS) format to decimal degrees.

    Parameters:
    -----------
        coords (str, list, or tuple): 
            - A single DMS coordinate as a string (e.g., "12d34m56s").
            - A list or tuple of DMS coordinates as strings.

    Returns:
    --------
        float or list: 
            - If a single coordinate is provided, returns its value in decimal degrees.
            - If multiple coordinates are provided (list or tuple), returns a list 
              of values in decimal degrees.

    Example:
    --------
        dms_to_deg("12d34m56s") -> 12.582222
        dms_to_deg(["12d34m56s", "45d67m89s"]) -> [12.582222, 46.135806]
    """
    from astropy.coordinates import Angle
    if isinstance(coords, (list, tuple)):
        return [Angle(coord).deg for coord in coords]
    else:
        return Angle(coords).deg
    return


def rad0to2pi(angles):
    """
    Normalize angles in radians to the range [0, 2π].

    Parameters:
    -----------
        angles (float or numpy array): 
            - A single angle in radians or an array of angles in radians.
            - Negative angles will be adjusted to fall within the range [0, 2π].

    Returns:
    --------
        float or numpy array: 
            - The normalized angle(s) in radians within the range [0, 2π].

    Example:
    --------
        rad0to2pi(-1.0) -> 5.283185307179586
        rad0to2pi(np.array([-1.0, 3.0])) -> array([5.28318531, 3.0])
    """
    return (2 * np.pi + angles) * (angles < 0) + angles * (angles > 0)


def deg0to360(array_):
    """
    Normalize angles in degrees to the range [0, 360].

    Parameters:
    -----------
        array_ (int, float, or iterable): 
            - A single angle in degrees (int or float).
            - An iterable (e.g., list, tuple, or numpy array) of angles in degrees.

    Returns:
    --------
        int, float, or list: 
            - If a single angle is provided, returns the normalized angle in the range [0, 360].
            - If an iterable of angles is provided, returns a list of normalized angles in the range [0, 360].

    Example:
    --------
        deg0to360(370) -> 10
        deg0to360([-10, 370, 720]) -> [350, 10, 0]
    """
    try:
        return [i % 360 for i in array_]
    except TypeError:
        return array_ % 360


def deg0to360array(array_):
    """
    Normalize an array of angles in degrees to the range [0, 360].

    Parameters:
    -----------
        array_ (iterable): 
            - An iterable (e.g., list, tuple, or numpy array) of angles in degrees.

    Returns:
    --------
        list: 
            - A list of normalized angles in the range [0, 360].

    Example:
    --------
        deg0to360array([-10, 370, 720]) -> [350, 10, 0]
    """
    return [i % 360 for i in array_]


def deg90to90(val_in):
    """
    Normalize angles to the range [-90, 90].

    This function adjusts angles such that they fall within the range [-90, 90]. 
    It works for both single values and iterable inputs.

    Parameters:
    -----------
        val_in (int, float, or iterable): 
            - A single angle (int or float).
            - An iterable (e.g., list, tuple, or numpy array) of angles.

    Returns:
    --------
        int, float, or list: 
            - If a single angle is provided, returns the normalized angle in the range [-90, 90].
            - If an iterable of angles is provided, returns a list of normalized angles in the range [-90, 90].

    Example:
    --------
        deg90to90(100) -> 10
        deg90to90([-100, 200, -270]) -> [-10, -70, -90]
    """
    if hasattr(val_in, "__len__"):
        val_out = []
        for i, v in enumerate(val_in):
            while v < -90:
                v += 90
            while v > 90:
                v -= 90
            val_out.append(v)
    else:
        while val_in < -90:
            val_in += 90
        while val_in > 90:
            val_in -= 90
        val_out = val_in
    return val_out


def deg90to90array(array_):
    """
    Normalize an array of angles to the range [0, 90].

    This function adjusts angles in an iterable such that they fall within the range [0, 90] using the modulo operation.

    Parameters:
    -----------
        array_ (iterable): 
            - An iterable (e.g., list, tuple, or numpy array) of angles.

    Returns:
    --------
        list: 
            - A list of normalized angles in the range [0, 90].

    Example:
    --------
        deg90to90array([95, 180, 270]) -> [5, 0, 0]
    """
    return [i % 90 for i in array_]


def cart2sph_deg(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates in degrees.

    This function converts Cartesian coordinates (x, y, z) to spherical coordinates:
    azimuth (az), elevation (el), and radius (r). The azimuth and elevation angles are returned in degrees.

    Parameters:
    -----------
        x (float or array-like): The x-coordinate(s).
        y (float or array-like): The y-coordinate(s).
        z (float or array-like): The z-coordinate(s).

    Returns:
    --------
        tuple: 
            - az (float or array-like): Azimuth angle in degrees (angle in the x-y plane from the positive x-axis).
            - el (float or array-like): Elevation angle in degrees (angle from the x-y plane to the z-axis).
            - r (float or array-like): Radius (distance from the origin).

    Example:
    --------
        cart2sph_deg(1, 1, 1) -> (45.0, 35.264389682754654, 1.7320508075688772)
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy) * (180 / np.pi)
    az = (np.arctan2(y, x)) * (180 / np.pi)
    return az, el, r


def cart_to_cyl(x, y, z):
    """
    Convert Cartesian coordinates to cylindrical coordinates.

    This function converts Cartesian coordinates (x, y, z) to cylindrical coordinates:
    radial distance (r), azimuthal angle (theta), and height (z).

    Parameters:
    -----------
        x (float or array-like): The x-coordinate(s).
        y (float or array-like): The y-coordinate(s).
        z (float or array-like): The z-coordinate(s).

    Returns:
    --------
        tuple:
            - r (float or array-like): Radial distance from the origin in the x-y plane.
            - theta (float or array-like): Azimuthal angle in radians (angle in the x-y plane from the positive x-axis).
            - z (float or array-like): Height along the z-axis (unchanged from input).

    Example:
    --------
        cart_to_cyl(1, 1, 1) -> (1.4142135623730951, 0.7853981633974483, 1)
    """
    r = np.linalg.norm([x, y])
    theta = np.arctan2(y, x)
    return r, theta, z


def inert2rot(x, y, xe, ye, xs=0, ys=0):  # Places Earth at (-1,0)
    """
    Transform inertial coordinates to rotated coordinates relative to the Earth.

    This function calculates the rotated coordinates of a point (x, y) relative to the Earth,
    which is assumed to be located at a fixed position (-1, 0) in the rotated frame. The rotation
    is performed based on the relative position of the Earth (xe, ye) and an optional reference 
    point (xs, ys).

    Parameters:
    -----------
        x (float): The x-coordinate of the point in the inertial frame.
        y (float): The y-coordinate of the point in the inertial frame.
        xe (float): The x-coordinate of the Earth in the inertial frame.
        ye (float): The y-coordinate of the Earth in the inertial frame.
        xs (float, optional): The x-coordinate of the reference point (default is 0).
        ys (float, optional): The y-coordinate of the reference point (default is 0).

    Returns:
    --------
        tuple:
            - xrot (float): The x-coordinate of the point in the rotated frame.
            - yrot (float): The y-coordinate of the point in the rotated frame.

    Example:
    --------
        inert2rot(2, 3, -1, 0) -> (-3.0, -3.605551275463989)
    """
    earth_theta = np.arctan2(ye - ys, xe - xs)
    theta = np.arctan2(y - ys, x - xs)
    distance = np.sqrt(np.power((x - xs), 2) + np.power((y - ys), 2))
    xrot = distance * np.cos(np.pi + (theta - earth_theta))
    yrot = distance * np.sin(np.pi + (theta - earth_theta))
    return xrot, yrot


def sim_lonlatrad(x, y, z, xe, ye, ze, xs, ys, zs):
    """
    Simulate longitude, latitude, and radius in a geocentric frame with the Sun at (0, 0).

    This function calculates the longitude, latitude, and radius of a point (x, y, z) relative 
    to the Earth and the Sun. The input coordinates are first shifted to a geocentric frame 
    (Earth at the origin), then converted to spherical coordinates (longitude, latitude, radius). 
    The output is corrected so that the Sun is positioned at (0, 0) in the frame.

    Parameters:
    -----------
        x (float): The x-coordinate of the point in the inertial frame.
        y (float): The y-coordinate of the point in the inertial frame.
        z (float): The z-coordinate of the point in the inertial frame.
        xe (float): The x-coordinate of the Earth in the inertial frame.
        ye (float): The y-coordinate of the Earth in the inertial frame.
        ze (float): The z-coordinate of the Earth in the inertial frame.
        xs (float): The x-coordinate of the Sun in the inertial frame.
        ys (float): The y-coordinate of the Sun in the inertial frame.
        zs (float): The z-coordinate of the Sun in the inertial frame.

    Returns:
    --------
        tuple:
            - longitude (float): The corrected longitude of the point in degrees.
            - latitude (float): The corrected latitude of the point in degrees.
            - radius (float): The radial distance of the point from the Earth.

    Notes:
    ------
        - The function assumes the existence of `cart2sph_deg`, which converts Cartesian 
          coordinates to spherical coordinates in degrees.
        - The function assumes the existence of `deg0to360`, which ensures longitude values 
          are within the range [0, 360] degrees.

    Example:
    --------
        sim_lonlatrad(1, 2, 3, -1, 0, 0, 0, 0, 0) -> (180.0, 45.0, 3.7416573867739413)
    """
    # convert all to geo coordinates
    x = x - xe
    y = y - ye
    z = z - ze
    xs = xs - xe
    ys = ys - ye
    zs = zs - ze
    # convert x y z to lon lat radius
    longitude, latitude, radius = cart2sph_deg(x, y, z)
    slongitude, slatitude, sradius = cart2sph_deg(xs, ys, zs)
    # correct so that Sun is at (0,0)
    longitude = deg0to360(slongitude - longitude)
    latitude = latitude - slatitude
    return longitude, latitude, radius


def sun_ra_dec(time_):
    """
    Calculate the Right Ascension (RA) and Declination (Dec) of the Sun at a given time.

    This function computes the Sun's position in the sky in terms of its Right Ascension (RA) 
    and Declination (Dec) in radians, based on the provided time. It uses the `get_body` 
    function from the `.body` module to retrieve the Sun's coordinates.

    Parameters:
    -----------
        time_ (float): The time in Modified Julian Date (MJD) format.

    Returns:
    --------
        tuple:
            - ra (float): The Sun's Right Ascension in radians.
            - dec (float): The Sun's Declination in radians.

    Notes:
    ------
        - The function assumes the existence of a `get_body` function in the `.body` module, 
          which calculates the celestial coordinates of the Sun.
        - The `Time` class from `astropy.time` is used to handle the MJD time format.

    Example:
    --------
        sun_ra_dec(60000.0) -> (3.141592653589793, -0.40909280422232897)
    """
    from .body import get_body
    out = get_body(Time(time_, format='mjd'))
    return out.ra.to('rad').value, out.dec.to('rad').value


def ra_dec(r=None, v=None, x=None, y=None, z=None, vx=None, vy=None, vz=None, r_earth=np.array([0, 0, 0]), v_earth=np.array([0, 0, 0]), input_unit='si'):
    """
    Calculate the Right Ascension (RA) and Declination (Dec) of an object relative to Earth's position.

    This function computes the RA and Dec of an object based on its position and velocity vectors. 
    The Earth's position and velocity are subtracted from the input to determine the object's 
    coordinates relative to Earth. The RA is returned in radians within the range [0, 2π], and 
    the Dec is returned in radians.

    Parameters:
    -----------
        r (ndarray, optional): Position vector of the object in 3D space (shape: Nx3). Default is None.
        v (ndarray, optional): Velocity vector of the object in 3D space (shape: Nx3). Default is None.
        x (float, optional): X-coordinate of the object's position. Default is None.
        y (float, optional): Y-coordinate of the object's position. Default is None.
        z (float, optional): Z-coordinate of the object's position. Default is None.
        vx (float, optional): X-component of the object's velocity. Default is None.
        vy (float, optional): Y-component of the object's velocity. Default is None.
        vz (float, optional): Z-component of the object's velocity. Default is None.
        r_earth (ndarray, optional): Earth's position vector in 3D space (shape: 3). Default is [0, 0, 0].
        v_earth (ndarray, optional): Earth's velocity vector in 3D space (shape: 3). Default is [0, 0, 0].
        input_unit (str, optional): Unit of the input values ('si' for meters and seconds). Default is 'si'.

    Returns:
    --------
        tuple:
            - ra (ndarray): Right Ascension of the object in radians (shape: N).
            - dec (ndarray): Declination of the object in radians (shape: N).

    Raises:
    -------
        ValueError: If neither `r` and `v` nor individual coordinates (`x`, `y`, `z`, `vx`, `vy`, `vz`) are provided.

    Notes:
    ------
        - If `r` and `v` are not provided, the function expects individual position (`x`, `y`, `z`) 
          and velocity (`vx`, `vy`, `vz`) components to construct the vectors.
        - The function assumes the existence of `einsum_norm`, which calculates the norm of vectors 
          using Einstein summation notation.
        - The function assumes the existence of `rad0to2pi`, which ensures RA values are within the range [0, 2π].

    Example:
    --------
        ra_dec(x=1.0, y=2.0, z=3.0, vx=0.1, vy=0.2, vz=0.3) -> (array([1.10714872]), array([0.64052231]))
    """
    if r is None or v is None:
        if x is not None and y is not None and z is not None and vx is not None and vy is not None and vz is not None:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])
        else:
            raise ValueError("Either provide r and v arrays or individual coordinates (x, y, z) and velocities (vx, vy, vz)")

    # Subtract Earth's position and velocity from the input arrays
    r = r - r_earth
    v = v - v_earth

    d_earth_mag = einsum_norm(r, 'ij,ij->i')
    ra = rad0to2pi(np.arctan2(r[:, 1], r[:, 0]))  # in radians
    dec = np.arcsin(r[:, 2] / d_earth_mag)
    return ra, dec


def lonlat_distance(lat1, lat2, lon1, lon2):
    """Calculate the great-circle distance between two points 
    on Earth's surface using the Haversine formula.

    Parameters
    ----------
    lat1 : float
        Latitude of the first point in radians.
    lat2 : float
        Latitude of the second point in radians.
    lon1 : float
        Longitude of the first point in radians.
    lon2 : float
        Longitude of the second point in radians.

    Returns
    -------
    distance : float
        Distance between the two points in kilometers.
    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Calculate distance in kilometers, use 3956 for miles
    distance = c * EARTH_RADIUS
    return distance


def altitude_to_zenithangle(altitude, deg=True):
    """
    Convert altitude angle to zenith angle.

    Example:
    --------
        altitude_to_zenithangle(45, deg=True) -> 45
        altitude_to_zenithangle(np.pi / 4, deg=False) -> np.pi / 4
    """
    if deg:
        out = 90 - altitude
    else:
        out = np.pi / 2 - altitude
    return out


def zenithangle_to_altitude(zenith_angle, deg=True):
    """
    Convert zenith angle to altitude angle.

    Parameters:
    -----------
        zenith_angle (float or ndarray): Zenith angle of the object (in degrees if `deg=True`, otherwise in radians).
        deg (bool, optional): If True, input and output are in degrees; if False, in radians. Default is True.

    Returns:
    --------
        float or ndarray: Altitude angle corresponding to the input zenith angle.
    """
    if deg:
        out = 90 - zenith_angle
    else:
        out = np.pi / 2 - zenith_angle
    return out


def rightascension_to_hourangle(right_ascension, local_time):
    """
    Convert right ascension to hour angle.

    This function calculates the hour angle of a celestial object based on its right ascension 
    and the local time. The hour angle represents the angular distance between the object's 
    current position and the local meridian.

    Parameters:
    -----------
        right_ascension (float or str): The right ascension of the object. Can be provided as a 
                                        decimal degree value or as a string in "HH:MM:SS" format.
        local_time (float or str): The local time. Can be provided as a decimal degree value or 
                                   as a string in "HH:MM:SS" format.

    Returns:
    --------
        str: The hour angle in "HH:MM:SS" format.

    Notes:
    ------
        - If `right_ascension` or `local_time` is provided as a decimal degree, it is converted 
          to the appropriate "HH:MM:SS" or "DD:MM:SS" format internally.
        - Handles cases where the right ascension exceeds the local time by adjusting the local 
          time to account for the 24-hour cycle.

    Example:
    --------
        rightascension_to_hourangle("10:30:00", "12:45:00") -> "02:15:00"
        rightascension_to_hourangle(157.5, 191.25) -> "02:15:00"
    """
    if type(right_ascension) is not str:
        right_ascension = dd_to_hms(right_ascension)
    if type(local_time) is not str:
        local_time = dd_to_dms(local_time)
    _ra = float(right_ascension.split(':')[0])
    _lt = float(local_time.split(':')[0])
    if _ra > _lt:
        __ltm, __lts = local_time.split(':')[1:]
        local_time = f'{24 + _lt}:{__ltm}:{__lts}'

    return dd_to_dms(hms_to_dd(local_time) - hms_to_dd(right_ascension))


def equatorial_to_horizontal(observer_latitude, declination, right_ascension=None, hour_angle=None, local_time=None, hms=False):
    """
    Convert equatorial coordinates (declination and either right ascension or hour angle) 
    to horizontal coordinates (azimuth and altitude) for a given observer's latitude.

    Parameters:
    -----------
    observer_latitude (float): Latitude of the observer in degrees.
    declination (float): Declination of the celestial object in degrees.
    right_ascension (float, optional): Right ascension of the celestial object in hours. 
                                       If provided, `local_time` is required to calculate hour angle.
    hour_angle (float, optional): Hour angle of the celestial object in degrees or hours. 
                                   If provided, it will be used directly for calculations.
    local_time (float, optional): Local time in hours, used to compute hour angle from right ascension.
    hms (bool, optional): If True, interprets hour angle or right ascension as hours-minutes-seconds (HMS) 
                          and converts them to decimal degrees.

    Returns:
    --------
    tuple: A tuple containing:
        - azimuth (float): Azimuth angle in degrees, measured clockwise from north.
        - altitude (float): Altitude angle in degrees, measured above the horizon.

    Notes:
    ------
    - Either `right_ascension` or `hour_angle` must be provided for the calculation.
    - If both `right_ascension` and `hour_angle` are provided, `hour_angle` will take precedence.
    """
    if right_ascension is not None:
        hour_angle = rightascension_to_hourangle(right_ascension, local_time)
        if hms:
            hour_angle = hms_to_dd(hour_angle)
    elif hour_angle is not None:
        if hms:
            hour_angle = hms_to_dd(hour_angle)
    elif right_ascension is not None and hour_angle is not None:
        print('Both right_ascension and hour_angle parameters are provided.\nUsing hour_angle for calculations.')
        if hms:
            hour_angle = hms_to_dd(hour_angle)
    else:
        print('Either right_ascension or hour_angle must be provided.')

    observer_latitude, hour_angle, declination = np.radians([observer_latitude, hour_angle, declination])

    zenith_angle = np.arccos(np.sin(observer_latitude) * np.sin(declination) + np.cos(observer_latitude) * np.cos(declination) * np.cos(hour_angle))

    altitude = zenithangle_to_altitude(zenith_angle, deg=False)

    _num = np.sin(declination) - np.sin(observer_latitude) * np.cos(zenith_angle)
    _den = np.cos(observer_latitude) * np.sin(zenith_angle)
    azimuth = np.arccos(_num / _den)

    if observer_latitude < 0:
        azimuth = np.pi - azimuth
    altitude, azimuth = np.degrees([altitude, azimuth])

    return azimuth, altitude


def horizontal_to_equatorial(observer_latitude, azimuth, altitude):
    """
    Convert horizontal coordinates (azimuth and altitude) to equatorial coordinates 
    (hour angle and declination).

    This function calculates the hour angle and declination of a celestial object 
    based on its horizontal coordinates and the observer's latitude.

    Parameters:
    -----------
        observer_latitude (float): Latitude of the observer in degrees.
        azimuth (float): Azimuth angle of the object in degrees (measured clockwise from north).
        altitude (float): Altitude angle of the object in degrees (above the horizon).

    Returns:
    --------
        tuple: A tuple containing:
            - hour_angle (float): Hour angle of the object in degrees.
            - declination (float): Declination of the object in degrees.

    Notes:
    ------
        - The function assumes the input angles are in degrees and internally converts them 
          to radians for calculations.
        - Adjusts for southern hemisphere observations by flipping zenith angle signs.
        - The hour angle is calculated using trigonometric relationships, with corrections 
          for specific latitude and declination conditions.

    Example:
    --------
        horizontal_to_equatorial(45.0, 120.0, 30.0) -> (hour_angle, declination)
    """

    altitude, azimuth, latitude = np.radians([altitude, azimuth, observer_latitude])
    zenith_angle = zenithangle_to_altitude(altitude)

    zenith_angle = [-zenith_angle if latitude < 0 else zenith_angle][0]

    declination = np.sin(latitude) * np.cos(zenith_angle)
    declination = declination + (np.cos(latitude) * np.sin(zenith_angle) * np.cos(azimuth))
    declination = np.arcsin(declination)

    _num = np.cos(zenith_angle) - np.sin(latitude) * np.sin(declination)
    _den = np.cos(latitude) * np.cos(declination)
    hour_angle = np.arccos(_num / _den)

    if (latitude > 0 > declination) or (latitude < 0 < declination):
        hour_angle = 2 * np.pi - hour_angle

    declination, hour_angle = np.degrees([declination, hour_angle])

    return hour_angle, declination


_ecliptic = 0.409092601  # np.radians(23.43927944)
cos_ec = 0.9174821430960974
sin_ec = 0.3977769690414367


def equatorial_xyz_to_ecliptic_xyz(xq, yq, zq):
    """
    Convert equatorial rectangular coordinates (X, Y, Z) to ecliptic rectangular coordinates.

    This function transforms the position of an object from the equatorial coordinate system 
    to the ecliptic coordinate system using the obliquity of the ecliptic.

    Parameters:
    -----------
        xq (float): X-coordinate in the equatorial coordinate system.
        yq (float): Y-coordinate in the equatorial coordinate system.
        zq (float): Z-coordinate in the equatorial coordinate system.

    Returns:
    --------
        tuple: A tuple containing:
            - xc (float): X-coordinate in the ecliptic coordinate system (unchanged from equatorial X).
            - yc (float): Y-coordinate in the ecliptic coordinate system.
            - zc (float): Z-coordinate in the ecliptic coordinate system.

    Notes:
    ------
        - The transformation uses the obliquity of the ecliptic (`sin_ec` and `cos_ec`) to rotate 
          the Y and Z components.
        - The obliquity of the ecliptic (`sin_ec` and `cos_ec`) must be defined globally or imported 
          prior to calling this function.

    Example:
    --------
        equatorial_xyz_to_ecliptic_xyz(1.0, 0.5, 0.3) -> (xc, yc, zc)
    """
    xc = xq
    yc = cos_ec * yq + sin_ec * zq
    zc = -sin_ec * yq + cos_ec * zq
    return xc, yc, zc


def ecliptic_xyz_to_equatorial_xyz(xc, yc, zc):
    """
    Convert ecliptic rectangular coordinates (X, Y, Z) to equatorial rectangular coordinates.

    This function transforms the position of an object from the ecliptic coordinate system 
    to the equatorial coordinate system using the obliquity of the ecliptic.

    Parameters:
    -----------
        xc (float): X-coordinate in the ecliptic coordinate system.
        yc (float): Y-coordinate in the ecliptic coordinate system.
        zc (float): Z-coordinate in the ecliptic coordinate system.

    Returns:
    --------
        tuple: A tuple containing:
            - xq (float): X-coordinate in the equatorial coordinate system (unchanged from ecliptic X).
            - yq (float): Y-coordinate in the equatorial coordinate system.
            - zq (float): Z-coordinate in the equatorial coordinate system.

    Notes:
    ------
        - The transformation uses the obliquity of the ecliptic (`sin_ec` and `cos_ec`) to rotate 
          the Y and Z components.
        - The obliquity of the ecliptic (`sin_ec` and `cos_ec`) must be defined globally or imported 
          prior to calling this function.

    Example:
    --------
        ecliptic_xyz_to_equatorial_xyz(1.0, 0.5, 0.3) -> (xq, yq, zq)
    """
    xq = xc
    yq = cos_ec * yc - sin_ec * zc
    zq = sin_ec * yc + cos_ec * zc
    return xq, yq, zq


def xyz_to_ecliptic(xc, yc, zc, xe=0, ye=0, ze=0, degrees=False):
    """
    Convert rectangular coordinates (X, Y, Z) to ecliptic longitude and latitude.

    This function computes the ecliptic longitude and latitude of an object relative to the Earth 
    or another reference point, given its rectangular coordinates in the ecliptic coordinate system.

    Parameters:
    -----------
        xc (float): X-coordinate of the object in the ecliptic coordinate system.
        yc (float): Y-coordinate of the object in the ecliptic coordinate system.
        zc (float): Z-coordinate of the object in the ecliptic coordinate system.
        xe (float, optional): X-coordinate of the reference point (default is 0, typically Earth's position).
        ye (float, optional): Y-coordinate of the reference point (default is 0, typically Earth's position).
        ze (float, optional): Z-coordinate of the reference point (default is 0, typically Earth's position).
        degrees (bool, optional): If `True`, returns the longitude and latitude in degrees; 
                                  otherwise, returns them in radians (default is `False`).

    Returns:
    --------
        tuple: A tuple containing:
            - ec_longitude (float): Ecliptic longitude of the object (in radians or degrees).
            - ec_latitude (float): Ecliptic latitude of the object (in radians or degrees).

    Notes:
    ------
        - The calculation involves finding the vector from the reference point to the object 
          and determining its magnitude and angular position.
        - The `rad0to2pi` function ensures the longitude is normalized to the range [0, 2π] in radians.
        - The `np.arctan2` function is used to compute the longitude, and `np.arcsin` is used for latitude.
        - If `degrees=True`, the results are converted from radians to degrees using `np.degrees`.

    Example:
    --------
        xyz_to_ecliptic(1.0, 0.5, 0.3, xe=0.1, ye=0.2, ze=0.3, degrees=True) -> (longitude, latitude)
    """
    x_ast_to_earth = xc - xe
    y_ast_to_earth = yc - ye
    z_ast_to_earth = zc - ze
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth, 2) + np.power(y_ast_to_earth, 2) + np.power(z_ast_to_earth, 2))
    ec_longitude = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))  # in radians
    ec_latitude = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ec_longitude), np.degrees(ec_latitude)
    else:
        return ec_longitude, ec_latitude


def xyz_to_equatorial(xq, yq, zq, xe=0, ye=0, ze=0, degrees=False):
    """
    Convert rectangular coordinates (X, Y, Z) to equatorial right ascension (RA) and declination (DEC).

    This function computes the equatorial coordinates of an object relative to the Earth 
    or another reference point, given its rectangular coordinates in the equatorial coordinate system.

    Parameters:
    -----------
        xq (float): X-coordinate of the object in the equatorial coordinate system.
        yq (float): Y-coordinate of the object in the equatorial coordinate system.
        zq (float): Z-coordinate of the object in the equatorial coordinate system.
        xe (float, optional): X-coordinate of the reference point (default is 0, typically Earth's position).
        ye (float, optional): Y-coordinate of the reference point (default is 0, typically Earth's position).
        ze (float, optional): Z-coordinate of the reference point (default is 0, typically Earth's position).
        degrees (bool, optional): If `True`, returns the RA and DEC in degrees; otherwise, returns them in radians (default is `False`).

    Returns:
    --------
        tuple: A tuple containing:
            - ra (float): Right ascension of the object (in radians or degrees).
            - dec (float): Declination of the object (in radians or degrees).

    Notes:
    ------
        - The calculation assumes the XY plane corresponds to the celestial equator, 
          and the -X axis points toward the vernal equinox.
        - The `rad0to2pi` function ensures the RA is normalized to the range [0, 2π] in radians.
        - The `np.arctan2` function is used to compute the RA, and `np.arcsin` is used for DEC.
        - If `degrees=True`, the results are converted from radians to degrees using `np.degrees`.

    Example:
    --------
        xyz_to_equatorial(1.0, 0.5, 0.3, xe=0.1, ye=0.2, ze=0.3, degrees=True) -> (ra, dec)
    """
    # RA / DEC calculation - assumes XY plane to be celestial equator, and -x axis to be vernal equinox
    x_ast_to_earth = xq - xe
    y_ast_to_earth = yq - ye
    z_ast_to_earth = zq - ze
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth, 2) + np.power(y_ast_to_earth, 2) + np.power(z_ast_to_earth, 2))
    ra = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))  # in radians
    dec = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec


def ecliptic_xyz_to_equatorial(xc, yc, zc, xe=0, ye=0, ze=0, degrees=False):
    """
    Convert ecliptic Cartesian coordinates (X, Y, Z) to equatorial right ascension (RA) and declination (DEC).

    This function first converts ecliptic Cartesian coordinates to equatorial Cartesian coordinates 
    and then computes the equatorial coordinates (RA and DEC) of an object relative to the Earth or another reference point.

    Parameters:
    -----------
        xc (float): X-coordinate of the object in the ecliptic coordinate system.
        yc (float): Y-coordinate of the object in the ecliptic coordinate system.
        zc (float): Z-coordinate of the object in the ecliptic coordinate system.
        xe (float, optional): X-coordinate of the reference point (default is 0, typically Earth's position).
        ye (float, optional): Y-coordinate of the reference point (default is 0, typically Earth's position).
        ze (float, optional): Z-coordinate of the reference point (default is 0, typically Earth's position).
        degrees (bool, optional): If `True`, returns the RA and DEC in degrees; otherwise, returns them in radians (default is `False`).

    Returns:
    --------
        tuple: A tuple containing:
            - ra (float): Right ascension of the object (in radians or degrees).
            - dec (float): Declination of the object (in radians or degrees).

    Notes:
    ------
        - The function relies on `ecliptic_xyz_to_equatorial_xyz` to perform the conversion 
          from ecliptic Cartesian coordinates to equatorial Cartesian coordinates.
        - The calculation assumes the XY plane corresponds to the celestial equator, 
          and the -X axis points toward the vernal equinox.
        - The `rad0to2pi` function ensures the RA is normalized to the range [0, 2π] in radians.
        - The `np.arctan2` function is used to compute the RA, and `np.arcsin` is used for DEC.
        - If `degrees=True`, the results are converted from radians to degrees using `np.degrees`.

    Example:
    --------
        ecliptic_xyz_to_equatorial(1.0, 0.5, 0.3, xe=0.1, ye=0.2, ze=0.3, degrees=True) -> (ra, dec)
    """
    # Convert ecliptic cartesian into equitorial cartesian
    x_ast_to_earth, y_ast_to_earth, z_ast_to_earth = ecliptic_xyz_to_equatorial_xyz(xc - xe, yc - ye, zc - ze)
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth, 2) + np.power(y_ast_to_earth, 2) + np.power(z_ast_to_earth, 2))
    ra = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))  # in radians
    dec = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec


def equatorial_to_ecliptic(right_ascension, declination, degrees=False):
    """
    Convert equatorial coordinates (RA, DEC) to ecliptic longitude and latitude.

    This function transforms equatorial right ascension (RA) and declination (DEC) into 
    ecliptic longitude and latitude, taking into account the obliquity of the ecliptic.

    Parameters:
    -----------
        right_ascension (float): Right ascension of the object (in degrees or radians).
        declination (float): Declination of the object (in degrees or radians).
        degrees (bool, optional): If `True`, assumes input is in degrees and returns output in degrees; 
                                  otherwise, assumes input is in radians and returns output in radians (default is `False`).

    Returns:
    --------
        tuple: A tuple containing:
            - ec_longitude (float): Ecliptic longitude of the object (in radians or degrees).
            - ec_latitude (float): Ecliptic latitude of the object (in radians or degrees).

    Notes:
    ------
        - The calculation uses the obliquity of the ecliptic, which is the tilt of Earth's axis relative to its orbit.
          The constants `cos_ec` and `sin_ec` represent the cosine and sine of the obliquity angle, respectively.
        - The `rad0to2pi` function ensures the ecliptic longitude is normalized to the range [0, 2π] in radians.
        - The `deg0to360` function ensures the ecliptic longitude is normalized to the range [0, 360] in degrees.
        - If `degrees=True`, the input is converted from degrees to radians using `np.radians`, and the output is converted 
          back to degrees using `np.degrees`.

    Example:
    --------
        equatorial_to_ecliptic(180.0, 45.0, degrees=True) -> (ec_longitude, ec_latitude)
    """
    ra, dec = np.radians(right_ascension), np.radians(declination)
    ec_latitude = np.arcsin(cos_ec * np.sin(dec) - sin_ec * np.cos(dec) * np.sin(ra))
    ec_longitude = np.arctan((cos_ec * np.cos(dec) * np.sin(ra) + sin_ec * np.sin(dec)) / (np.cos(dec) * np.cos(ra)))
    if degrees:
        return deg0to360(np.degrees(ec_longitude)), np.degrees(ec_latitude)
    else:
        return rad0to2pi(ec_longitude), ec_latitude


def ecliptic_to_equatorial(lon, lat, degrees=False):
    """
    Convert ecliptic coordinates (longitude, latitude) to equatorial right ascension (RA) and declination (DEC).

    This function transforms ecliptic longitude and latitude into equatorial right ascension (RA) and declination (DEC),
    taking into account the obliquity of the ecliptic.

    Parameters:
    -----------
        lon (float): Ecliptic longitude of the object (in degrees or radians).
        lat (float): Ecliptic latitude of the object (in degrees or radians).
        degrees (bool, optional): If `True`, assumes input is in degrees and returns output in degrees; 
                                  otherwise, assumes input is in radians and returns output in radians (default is `False`).

    Returns:
    --------
        tuple: A tuple containing:
            - ra (float): Right ascension of the object (in radians or degrees).
            - dec (float): Declination of the object (in radians or degrees).

    Notes:
    ------
        - The calculation uses the obliquity of the ecliptic, which is the tilt of Earth's axis relative to its orbit.
          The constants `cos_ec` and `sin_ec` represent the cosine and sine of the obliquity angle, respectively.
        - The `np.arctan` function computes the RA, and `np.arcsin` computes the DEC.
        - If `degrees=True`, the input is converted from degrees to radians using `np.radians`, and the output is converted 
          back to degrees using `np.degrees`.

    Example:
    --------
        ecliptic_to_equatorial(180.0, 45.0, degrees=True) -> (ra, dec)
    """
    lon, lat = np.radians(lon), np.radians(lat)
    ra = np.arctan((cos_ec * np.cos(lat) * np.sin(lon) - sin_ec * np.sin(lat)) / (np.cos(lat) * np.cos(lon)))
    dec = np.arcsin(cos_ec * np.sin(lat) + sin_ec * np.cos(lat) * np.sin(lon))
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec


# Stolen from https://github.com/lsst/utils/blob/main/python/lsst/utils/wrappers.py
INTRINSIC_SPECIAL_ATTRIBUTES = frozenset(
    (
        "__qualname__",
        "__module__",
        "__metaclass__",
        "__dict__",
        "__weakref__",
        "__class__",
        "__subclasshook__",
        "__name__",
        "__doc__",
    )
)


def isAttributeSafeToTransfer(name, value):
    # Stolen from https://github.com/lsst/utils/blob/main/python/lsst/utils/wrappers.py
    """Return True if an attribute is safe to monkeypatch-transfer to another
    class.
    This rejects special methods that are defined automatically for all
    classes, leaving only those explicitly defined in a class decorated by
    `continueClass` or registered with an instance of `TemplateMeta`.
    """
    if name.startswith("__") and (
        value is getattr(object, name, None) or name in INTRINSIC_SPECIAL_ATTRIBUTES
    ):
        return False
    return True


def continueClass(cls):
    # Stolen from https://github.com/lsst/utils/blob/main/python/lsst/utils/wrappers.py
    """Re-open the decorated class, adding any new definitions into the
    original.
    For example:
    .. code-block:: python
        class Foo:
            pass
        @continueClass
        class Foo:
            def run(self):
                return None
    is equivalent to:
    .. code-block:: python
        class Foo:
            def run(self):
                return None
    .. warning::
        Python's built-in `super` function does not behave properly in classes
        decorated with `continueClass`.  Base class methods must be invoked
        directly using their explicit types instead.
    """
    orig = getattr(sys.modules[cls.__module__], cls.__name__)
    for name in dir(cls):
        # Common descriptors like classmethod and staticmethod can only be
        # accessed without invoking their magic if we use __dict__; if we use
        # getattr on those we'll get e.g. a bound method instance on the dummy
        # class rather than a classmethod instance we can put on the target
        # class.
        attr = cls.__dict__.get(name, None) or getattr(cls, name)
        if isAttributeSafeToTransfer(name, attr):
            setattr(orig, name, attr)
    return orig


def dms_to_dd(dms):  # Degree minute second to Degree decimal
    """
    Converts coordinates from Degree-Minute-Second (DMS) format to Decimal Degrees (DD).

    Parameters:
    -----------
        dms (str or list of str): A single DMS string (e.g., "12:34:56") or a list of DMS strings 
                                  (e.g., ["12:34:56", "-45:30:15"]). Each string should represent 
                                  degrees, minutes, and seconds separated by colons.

    Returns:
    --------
        float or list of float: 
            - If the input is a single DMS string or a list with one element, returns a single float 
              representing the decimal degree value.
            - If the input is a list of multiple DMS strings, returns a list of floats representing 
              the decimal degree values.

    Notes:
    ------
        - Negative degrees are handled correctly, ensuring that the minutes and seconds are also 
          treated as negative when converting to decimal degrees.
        - The function supports both single DMS strings and lists of DMS strings for batch conversion.

    Example:
    --------
        >>> dms_to_dd("12:34:56")
        12.582222222222223

        >>> dms_to_dd(["12:34:56", "-45:30:15"])
        [12.582222222222223, -45.50416666666667]
    """
    dms, out = [[dms] if type(dms) is str else dms][0], []
    for i in dms:
        deg, minute, sec = [float(j) for j in i.split(':')]
        if deg < 0:
            minute, sec = float(f'-{minute}'), float(f'-{sec}')
        out.append(deg + minute / 60 + sec / 3600)
    return [out[0] if type(dms) is str or len(dms) == 1 else out][0]


def dd_to_dms(degree_decimal):
    """
    Converts a Decimal Degree (DD) value to Degree-Minute-Second (DMS) format.

    Parameters:
    -----------
        degree_decimal (float): A single decimal degree value to be converted to DMS format. 
                                Positive values represent north/east, and negative values represent 
                                south/west.

    Returns:
    --------
        str: A string representing the DMS format (e.g., "12:34:56"). The format includes:
            - Degrees as an integer.
            - Minutes as an integer.
            - Seconds as a float (rounded to 4 decimal places if necessary).

    Notes:
    ------
        - Handles negative decimal degree values correctly, ensuring the DMS format reflects the 
          correct sign for degrees, minutes, and seconds.
        - Ensures seconds are properly rounded and handles edge cases where seconds reach 60, 
          incrementing minutes accordingly.
        - Returns seconds as an integer if the value is a whole number.

    Example:
    --------
        >>> dd_to_dms(12.582222222222223)
        '12:34:56'

        >>> dd_to_dms(-45.50416666666667)
        '-45:30:15'

        >>> dd_to_dms(0.0002777777777777778)
        '0:0:1'
    """
    _d, __d = np.trunc(degree_decimal), degree_decimal - np.trunc(degree_decimal)
    __d = [-__d if degree_decimal < 0 else __d][0]
    _m, __m = np.trunc(__d * 60), __d * 60 - np.trunc(__d * 60)
    _s = round(__m * 60, 4)
    _s = [int(_s) if int(_s) == _s else _s][0]
    if _s == 60:
        _m, _s = _m + 1, '00'
    elif _s > 60:
        _m, _s = _m + 1, _s - 60

    return f'{int(_d)}:{int(_m)}:{_s}'


def hms_to_dd(hms):
    """
    Converts Hour-Minute-Second (HMS) format to Decimal Degrees (DD).

    Parameters:
    -----------
        hms (str or list of str): A single HMS string (e.g., "12:34:56") or a list of HMS strings 
                                  (e.g., ["12:34:56", "15:45:30"]). Each string should represent 
                                  hours, minutes, and seconds separated by colons.

    Returns:
    --------
        float or list of float:
            - If the input is a single HMS string or a list with one element, returns a single float 
              representing the decimal degree value.
            - If the input is a list of multiple HMS strings, returns a list of floats representing 
              the decimal degree values.

    Notes:
    ------
        - HMS values are converted to decimal degrees using the formula:
          Decimal Degrees = (Hours * 15) + (Minutes / 4) + (Seconds / 240).
        - Negative HMS values are not allowed, and the function will print an error message if 
          encountered.
        - The function supports both single HMS strings and lists of HMS strings for batch conversion.

    Example:
    --------
        >>> hms_to_dd("12:34:56")
        188.73333333333332

        >>> hms_to_dd(["12:34:56", "15:45:30"])
        [188.73333333333332, 236.375]
    """
    _type = type(hms)
    hms, out = [[hms] if _type == str else hms][0], []
    for i in hms:
        if i[0] != '-':
            hour, minute, sec = i.split(':')
            hour, minute, sec = float(hour), float(minute), float(sec)
            out.append(hour * 15 + (minute / 4) + (sec / 240))
        else:
            print('hms cannot be negative.')

    return [out[0] if _type == str or len(hms) == 1 else out][0]


def dd_to_hms(degree_decimal):
    """
    Converts Decimal Degrees (DD) to Hour-Minute-Second (HMS) format.

    Parameters:
    -----------
        degree_decimal (float or str): 
            - A decimal degree value (float) to be converted to HMS format.
            - If the input is a string in DMS format (e.g., "12:34:56"), it will be converted to DD 
              using the `dms_to_dd` function before processing.

    Returns:
    --------
        str: A string representing the HMS format (e.g., "12:34:56"). The format includes:
            - Hours as an integer.
            - Minutes as an integer.
            - Seconds as a float (rounded to 4 decimal places if necessary).

    Notes:
    ------
        - Decimal degrees are divided by 15 to convert to hours.
        - If the input DD value is negative, the function assumes the absolute value for conversion 
          and prints a warning message.
        - Handles edge cases where seconds reach 60, incrementing minutes accordingly.
        - Returns seconds as an integer if the value is a whole number.

    Example:
    --------
        >>> dd_to_hms(188.73333333333332)
        '12:34:56'

        >>> dd_to_hms(-236.375)
        '15:45:30'  # Assumes positive value for conversion.

        >>> dd_to_hms("12:34:56")  # DMS string converted to DD first.
        '0:50:18.4'
    """
    if type(degree_decimal) is str:
        degree_decimal = dms_to_dd(degree_decimal)
    if degree_decimal < 0:
        print('dd for HMS conversion cannot be negative, assuming positive.')
        _dd = -degree_decimal / 15
    else:
        _dd = degree_decimal / 15
    _h, __h = np.trunc(_dd), _dd - np.trunc(_dd)
    _m, __m = np.trunc(__h * 60), __h * 60 - np.trunc(__h * 60)
    _s = round(__m * 60, 4)
    _s = [int(_s) if int(_s) == _s else _s][0]
    if _s == 60:
        _m, _s = _m + 1, '00'
    elif _s > 60:
        _m, _s = _m + 1, _s - 60

    return f'{int(_h)}:{int(_m)}:{_s}'


def interpolate_points_between(r, m):
    """
    Interpolates points between the given points.

    Args:
        r: An (n, 3) numpy array of the original points.
        m: The number of points to interpolate between each pair of points.

    Returns:
        An (n * m) numpy array of the interpolated points.
    """
    n = len(r)
    interpolated_points = np.empty((0, 3))
    for i in range(n):
        # Generate 1D linearly spaced arrays along each axis using np.linspace()
        x = np.linspace(r[i, 0], r[i, 0], m)
        y = np.linspace(r[i, 1], r[i, 1], m)
        z = np.linspace(r[i, 2], r[i, 2], m)
        interpolated_points = np.vstack((interpolated_points, np.array([x, y, z]).T))
    return np.vstack((interpolated_points, r[-1]))
