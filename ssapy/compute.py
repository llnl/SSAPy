"""
Tools for orbital dynamics, satellite tracking, and celestial mechanics.
"""

import numpy as np
from astropy.time import Time
import astropy.units as u

from .constants import EARTH_RADIUS, EARTH_MU
from .propagator import KeplerianPropagator
from .utils import (
    norm, normed, unitAngle3, LRU_Cache, lb_to_unit, sunPos, _gpsToTT,
    iers_interp
)
from .orbit import Orbit
from .ellipsoid import Ellipsoid

try:
    import erfa
except ImportError:
    import astropy._erfa as erfa


def _doSqueeze(squeezeOrbit, squeezeTime, *args):
    out = []
    for i, arg in enumerate(args):
        if squeezeTime:
            if squeezeOrbit:
                out.append(arg[0, 0])  # scalar
            else:
                out.append(np.squeeze(arg, axis=1))  # (nOrbit,)
        else:
            if squeezeOrbit:
                out.append(np.squeeze(arg, axis=0))  # (nTime,)
            else:
                out.append(arg)  # (nOrbit, nTime)
    if len(out) == 1:
        return out[0]
    return tuple(out)


def _countOrbit(orbit):
    # orbit is one of:
    # 1) scalar Orbit
    # 2) vector Orbit
    # 3) list of scalar Orbit
    # convert to (2), set nOrbit, squeezeOrbit, and orbit.

    # check 1) and 2)
    if isinstance(orbit, Orbit):
        # check shape of r to decide scalar vs vector
        if orbit.r.ndim == 1:  # scalar Orbit
            nOrbit = 1
            squeezeOrbit = True
            if 'kozaiMeanKeplerianElements' in orbit.__dict__:
                kMKE = orbit.__dict__['kozaiMeanKeplerianElements']
            else:
                kMKE = None
            orbit = Orbit(
                np.atleast_2d(orbit.r),
                np.atleast_2d(orbit.v),
                np.atleast_1d(orbit.t),
                mu=orbit.mu,
                propkw={k: np.atleast_1d(v) for k, v in orbit.propkw.items()},
            )
            # Copying just kozaiMeanKeplerianElements for now, though maybe
            # ought to retain other (all?) lazy_properties attributes too?
            if kMKE is not None:
                orbit.kozaiMeanKeplerianElements = kMKE
        else:  # vector Orbit
            nOrbit = orbit.r.shape[0]
            squeezeOrbit = False
    else:
        import warnings
        warnings.warn(
            "list of Orbit syntax is deprecated.  "
            "Please use a vector Orbit instead.",
            DeprecationWarning
        )
        # list of scalar orbit
        nOrbit = len(orbit)
        squeezeOrbit = False
        # assumes all orbits have the same keywords and mu!
        propkeys = [] if nOrbit == 0 else orbit[0].propkw.keys()
        mu = None if nOrbit == 0 else orbit[0].mu
        orbit = Orbit(
            np.array([o.r for o in orbit]),
            np.array([o.v for o in orbit]),
            np.array([o.t for o in orbit]),
            mu=mu,
            propkw={k: np.array([o.propkw[k] for o in orbit])
                    for k in propkeys}
        )
    return nOrbit, squeezeOrbit, orbit


def _countTime(time):
    if isinstance(time, Time):
        time = time.gps
    squeezeTime = False
    try:
        nTime = len(time)
    except TypeError:
        time = np.array([time])
        nTime = 1
        squeezeTime = True
    return nTime, squeezeTime, time


def _countR(r):
    # orbit is one of:
    # 1) scalar r
    # 2) vector r
    # 3) list of scalar Orbit
    # convert to (2), set nOrbit, squeezeOrbit, and orbit.
    squeezeR = False
    if np.shape(r)[-1] == 3:
        pass
    else:
        raise ValueError(f"Incorrect r dimensions. Expected shape (n, 3), but got {np.shape(r)}.")
    # check 1) and 2)
    if r.ndim < 3:  # scalar r
        nR = 1
        r = np.reshape(np.atleast_3d(r), (nR, np.shape(r)[0], np.shape(r)[1]))
        squeezeR = True
    else:
        nR = np.shape(r)[0]
    return nR, squeezeR, r


def _processObserver(
    observer, obsPos, obsVel, nTime, squeezeTime, time, doObsVel=False
):
    # We always need obsPos output, but only sometimes obsVel.  So only require
    # obsVel if doObsVel is True.  (but always return it, and always compute it
    # if what we got as input was in an observer).
    squeezeObsPos = False
    if observer is None:
        if obsPos is None:
            raise ValueError(
                "Exactly one of obsPos and observer must be specified"
            )
        if doObsVel and obsVel is None:
            raise ValueError(
                "Exactly one of obsVel and observer required for doObsVel"
            )

    if observer is not None:
        if obsPos is not None:
            raise ValueError(
                "Exactly one of obsPos and observer must be specified"
            )
        if doObsVel and obsVel is not None:
            raise ValueError(
                "Exactly one of obsVel and observer required for doObsVel"
            )
        try:
            nObservers = len(observer)
        except TypeError:
            observer = [observer]
            nObservers = 1
            squeezeObsPos = squeezeTime
        # Now generate obsPos
        if nObservers == nTime:
            obsPos = np.empty((nObservers, 3), dtype=float)
            obsVel = np.empty((nObservers, 3), dtype=float)
            for i, (obs, t) in enumerate(zip(observer, time)):
                obsPos[i], obsVel[i] = obs.getRV(t)
        else:
            if nObservers == 1:
                obsPos, obsVel = observer[0].getRV(time)
            elif nTime == 1:
                obsPos = np.empty((nObservers, 3), dtype=float)
                obsVel = np.empty((nObservers, 3), dtype=float)
                for i, obs in enumerate(observer):
                    obsPos[i], obsVel[i] = obs.getRV(time[0])
            else:
                raise ValueError("observer and time must be broadcastable")

    if doObsVel:
        obsPos, obsVel = np.broadcast_arrays(obsPos, obsVel)
    obsPos = np.atleast_1d(obsPos)
    if obsPos.ndim == 2:
        nObsPos = obsPos.shape[0]
    else:
        obsPos = np.atleast_2d(obsPos)
        nObsPos = 1
        squeezeObsPos = True
    # times and obsPos must align
    if nObsPos != nTime:
        if nObsPos == 1:
            obsPos = np.broadcast_to(obsPos, (nTime, 3))
            if doObsVel:
                obsVel = np.broadcast_to(obsVel, (nTime, 3))
            nObsPos = nTime
        elif nTime == 1:
            time = np.broadcast_to(time, (nObsPos,))
            nTime = nObsPos
        else:
            raise ValueError("obsPos and time must be broadcastable")
    squeezeTime &= squeezeObsPos
    return obsPos, obsVel, time, squeezeTime


class HashableArrayContainer:
    """
    A container for NumPy arrays that makes them hashable and immutable.

    This class wraps a NumPy array and ensures that it is immutable by setting
    its `writeable` flag to `False`. It also provides implementations for
    `__hash__` and `__eq__`, enabling the wrapped array to be used as a key in
    hash-based data structures like dictionaries and sets.

    Attributes:
        arr (numpy.ndarray): The wrapped NumPy array, which is immutable.

    Methods:
        __hash__():
            Returns a hash value for the array based on its byte representation.
        
        __eq__(rhs):
            Compares the wrapped array with another `HashableArrayContainer` instance
            for equality based on element-wise comparison.
    
    Examples:
        >>> import numpy as np
        >>> arr1 = np.array([1, 2, 3])
        >>> arr2 = np.array([1, 2, 3])
        >>> container1 = HashableArrayContainer(arr1)
        >>> container2 = HashableArrayContainer(arr2)
        >>> container1 == container2
        True
        >>> hash(container1) == hash(container2)
        True
        >>> my_dict = {container1: "value"}
        >>> my_dict[container2]
        'value'
    """

    def __init__(self, arr):
        self.arr = arr
        self.arr.flags.writeable = False

    def __hash__(self):
        return hash(self.arr.data.tobytes())

    def __eq__(self, rhs):
        return np.all(self.arr == rhs.arr)


def rv(orbit, time, propagator=KeplerianPropagator()):
    """Calculate positions and velocities on the outer product of all supplied
    orbits and times.

    Parameters
    ----------
    orbit : Orbit or list of Orbit (n,)
        Desired orbit(s)
    time : array_like or astropy.time.Time (m,)
        If float (array), then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    propagator : Propagator, optional
        The propagator instance to use.

    Notes
    -----
    If orbit or time is scalar valued (as opposed to a list of Orbit, e.g.) then
    the corresponding dimension of the output will be squeezed out.

    For Keplerian orbit propagation it is more efficient to use a "vector Orbit"
    instead of a list of single scalar Orbits.

    Returns
    -------
    r : array_like (n, m, 3)
        Position in meters.
    v : array_like (n, m, 3)
        Velocity in meters per second.
    """
    nOrbit, squeezeOrbit, orbit = _countOrbit(orbit)
    nTime, squeezeTime, time = _countTime(time)
    # print(nOrbit, squeezeOrbit, orbit, nTime, squeezeTime, time)
    outR, outV = _rv(orbit, HashableArrayContainer(time), propagator)

    return _doSqueeze(squeezeOrbit, squeezeTime, outR, outV)


def __rv(orbit, time, propagator):
    outR, outV = propagator._getRVMany(orbit, time.arr)

    return outR, outV


_rv = LRU_Cache(__rv, maxsize=16)


def groundTrack(orbit, time, propagator=KeplerianPropagator(), format='geodetic'):
    """Calculate satellite ground track on the outer product of all supplied times and
    state vectors or orbits.

    Parameters
    ----------
    r : array_like (n,3) Position (m)

    time : array_like or astropy.time.Time (m,)
        If float (array), then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    or

    orbit : r : array_like (n,3) Position (m) or
        Orbit or list of Orbit (n,)
        Desired orbit(s)
    propagator : Propagator, optional
        The propagator instance to use.
    format : 'geodetic' or 'cartesian'
        If 'geodetic', then returns longitude, latitude, height.
        If 'cartesian', then returns xyz in ITRF frame.

    Notes
    -----
    If orbit or time is scalar valued (as opposed to a list of Orbit, e.g.) then
    the corresponding dimension of the output will be squeezed out.

    For Keplerian orbit propagation it is more efficient to use a "vector Orbit"
    instead of a list of single scalar Orbits.

    Returns
    -------
    lon, lat, height : array_like (n, m, 3)
        Radians and meters.

    or

    x, y, z : array_like(n, m, 3)
        Meters.
    """
    if format not in ['cartesian', 'geodetic']:
        raise ValueError("Format must be either 'cartesian' or 'geodetic'")

    nTime, squeezeTime, time = _countTime(time)
    if isinstance(orbit, Orbit):
        nOrbit, squeezeOrbit, orbit = _countOrbit(orbit)
        r, v = rv(orbit, time, propagator=propagator)  # (n, m, 3)
    else:
        nOrbit, squeezeOrbit, r = _countR(orbit)  # (n, m, 3)

    # Reverse the math in EarthObserver.getRV
    mjd_tt = _gpsToTT(time)
    d_ut1_tt_mjd, pmx, pmy = iers_interp(time)
    pn = erfa.pnm80(2400000.5, mjd_tt)
    gst = erfa.gst94(2400000.5, mjd_tt + d_ut1_tt_mjd)
    cg, sg = np.cos(gst), np.sin(gst)
    gstMat = np.zeros([len(time), 3, 3], dtype=float)
    gstMat[:, 0, 0] = cg
    gstMat[:, 0, 1] = sg
    gstMat[:, 1, 0] = -sg
    gstMat[:, 1, 1] = cg
    gstMat[:, 2, 2] = 1.0

    polar = np.zeros([len(time), 3, 3], dtype=float)
    polar[:] = np.eye(3)

    polar[:, 0, 2] = pmx
    polar[:, 1, 2] = -pmy
    polar[:, 2, 0] = -pmx
    polar[:, 2, 1] = pmy

    U = gstMat @ pn
    itrs = np.einsum(
        "tab,ntb->nta",
        (polar @ U),
        r
    )
    x = itrs[..., 0]
    y = itrs[..., 1]
    z = itrs[..., 2]
    if format == 'cartesian':
        return _doSqueeze(squeezeOrbit, squeezeTime, x, y, z)
    elif format == 'geodetic':
        ellipsoid = Ellipsoid()
        lon, lat, height = ellipsoid.cartToSphere(x, y, z)
        return _doSqueeze(squeezeOrbit, squeezeTime, lon, lat, height)


def _obsAngleCorrection(
    r, v, obsPos, obsVel, orbit, time, propagator, correctionType, max_iter=10
):
    if correctionType == "linear":
        # First order correction is to back up satellite by v * dt
        # where dt is the approximate time delay from the 0th order distance.
        # Important: *don't* do anything to the observer position, since we
        # really did do the observation at the originally given time and place.
        dt = norm(r - obsPos) / 299792458
        r -= np.einsum("nmi,nm->nmi", v, dt)
        # no change to v since linear assumption _is_ that v is constant for the
        # light time duration.
    elif correctionType == "exact":
        # Solve |r(t-dt) - r_obs(t)| = c dt for dt.
        dist = norm(r - obsPos)
        dt, dt_previous = dist / 299792458, np.inf
        iter = 0
        while np.any(np.abs(dt - dt_previous)) > 1e-12:  # picosecond accurate
            if iter > max_iter:
                raise RuntimeError(
                    "Exact light time correction did not converge in "
                    "{} iterations".format(iter)
                )
            # Unfortunately, can't just do
            # r, v = rv(orbit, time-dt, propagator=propagator)
            # because ssapy.rv isn't parallelized for n orbits and n, m times.
            # so do one orbit at a time.
            r = np.empty((len(orbit), len(time), 3))
            v = np.empty((len(orbit), len(time), 3))
            for i, o in enumerate(orbit):
                r[i], v[i] = rv(o, time - dt[i], propagator=propagator)
            dist = norm(r - obsPos)
            dt, dt_previous = dist / 299792458, dt
            iter += 1
    else:
        raise ValueError(
            "Invalid value for correctionType: {}"
            .format(correctionType)
        )
    # Apply velocity correction (i.e., diurnal aberration for an Earth-based
    # observer, and whatever you call the analogous correction for an orbital
    # observer).
    dr = norm(r - obsPos)
    r += obsVel * dr[..., None] / 299792458
    return r, v


def dircos(
    orbit, time, obsPos=None, obsVel=None, observer=None,
    propagator=KeplerianPropagator(), obsAngleCorrection=None
):
    """ Calculate observed direction-cosines of orbiting objects as viewed at
    specified times and positions.

    The direction cosines are the cosines of the angles between the vector
    pointing towards the orbiting object and the x, y, z axes.  An equivalent
    description is that they are the components of the unit vector pointing
    towards the orbiting object.

    Parameters
    ----------
    orbit : Orbit or list of Orbit (n,)
        Orbit(s) for which to calculate direction cosines.
    time : array_like or astropy.time.Time (m,)
        If float (array), then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    obsPos : array_like (m, 3), optional
        Position of observer at given time.
    obsVel : array_like (m, 3), optional
        Velocity of observer at given time.  Only required if correcting for
        diurnal aberration.
    observer : Observer or list of Observers (m,), optional
        Observer(s) used to calculate obsPos and obsVel.
    propagator : Propagator, optional
        The propagator instance to use.
    obsAngleCorrection : {None, "linear", "exact"}, optional
        Correct actual angle to observed angle, meaning account for light-time
        delay, and aberration due to the observer's velocity.  None means don't
        do any correction.  "linear" means do an aberration correction and first
        order light-time correction.  "exact" means do an aberration correction
        and iteratively solve for the exact light-time correction.  (The
        "linear" correction is almost always sufficiently accurate).

    Notes
    -----
    Exactly 1 of `obsPos` and `observer` must be supplied.  `observer` and
    `obsPos` follow similar broadcasting rules as detailed below explicitly only
    for `obsPos`.

    The length of `time` and `obsPos` must match or be broadcastable to match.
    If `orbit` is scalar-valued (an Orbit instead of a list of Orbit), then that
    dimension will be squeezed out in the return value.  Likewise, if both
    `time` and `obsPos` are scalar, that dimension will be squeezed out.

    For Keplerian orbit propagation it is more efficient to use a "vector Orbit"
    instead of a list of single scalar Orbits.

    When doing light time corrections, the time argument is the arrival time of
    the photons at the observer, as opposed to the emission time at the
    satellite.

    Returns
    -------
    dircos : array_like (n, m, 3)
        Direction cosines on the outer product of orbit(s) and time/obsPos.
    """
    nTime, squeezeTime, time = _countTime(time)
    nOrbit, squeezeOrbit, orbit = _countOrbit(orbit)
    obsPos, obsVel, time, squeezeTime = _processObserver(
        observer, obsPos, obsVel, nTime, squeezeTime, time,
        doObsVel=obsAngleCorrection
    )

    # At this point, should have len(orbit) == n
    # and len(obsPos) == len(time) == nTime
    # So get positions of orbits...
    r, v = rv(orbit, time, propagator=propagator)  # (n, m, 3)
    if obsAngleCorrection:
        r, v = _obsAngleCorrection(
            r, v, obsPos, obsVel, orbit, time, propagator, obsAngleCorrection
        )
    dc = normed(r - obsPos)

    return _doSqueeze(squeezeOrbit, squeezeTime, dc)


def radec(
    orbit, time, obsPos=None, obsVel=None, observer=None,
    propagator=KeplerianPropagator(), obsAngleCorrection=None, rate=False
):
    """ Calculate observed right ascension, declination, and slant range of
    orbiting objects as viewed at specified times and positions.

    Parameters
    ----------
    orbit : Orbit or list of Orbit (n,)
        Orbit(s) for which to calculate ra and dec.
    time : array_like or astropy.time.Time (m,)
        If float (array), then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    obsPos : array_like (m, 3), optional
        Positions of observers at given times.
    obsVel : array_like (m, 3), optional
        Velocity of observers at given times.
    observer : Observer or list of Observers (m,), optional
        Observer(s) used to calculate obsPos.
    propagator : Propagator, optional
        The propagator instance to use.
    rate : bool
        If True, return additionally the time derivatives of the quantity, times
        cos(dec) in the case of right ascension.
    obsAngleCorrection : {None, "linear", "exact"}, optional
        Correct actual angle to observed angle, meaning account for light-time
        delay, and aberration due to the observer's velocity.  None means don't
        do any correction.  "linear" means do an aberration correction and first
        order light-time correction.  "exact" means do an aberration correction
        and iteratively solve for the exact light-time correction.  (The
        "linear" correction is almost always sufficiently accurate).

    Notes
    -----
    Exactly 1 of `obsPos` and `observer` must be supplied.  `observer` and
    `obsPos` follow similar broadcasting rules as detailed below explicitly only
    for `obsPos`.

    The length of `time` and `obsPos` must match or be broadcastable to match.
    If `orbit` is scalar-valued (an Orbit instead of a list of Orbit), then that
    dimension will be squeezed out in the return value.  Likewise, if both
    `time` and `obsPos` are scalar, that dimension will be squeezed out.

    For Keplerian orbit propagation it is more efficient to use a "vector Orbit"
    instead of a list of single scalar Orbits.

    When doing light time corrections, the time argument is the arrival time of
    the photons at the observer, as opposed to the emission time at the
    satellite.

    Returns
    -------
    ra, dec : array_like (n, m)
        Right ascension and declination in radians.
    range : array_like (n, m)
        (Slant) range in meters.
    If rate, also:
    raRate : array_like (n, m)
        Time derivatives of right ascension times cos(dec), rad / sec.
    decRate : array_like (n, m)
        Time derivative of declination, rad / sec.
    rangeRate : array_like (n, m)
        Time derivative of range, meter / s.
    """
    nTime, squeezeTime, time = _countTime(time)
    nOrbit, squeezeOrbit, orbit = _countOrbit(orbit)
    obsPos, obsVel, time, squeezeTime = _processObserver(
        observer, obsPos, obsVel, nTime, squeezeTime, time,
        doObsVel=obsAngleCorrection
    )

    if rate and obsVel is None:
        raise ValueError('obsVel must not be None if rate is True.')

    # At this point, should have len(orbit) == n
    # and len(obsPos) == len(time) == nTime
    # So get positions of orbits...
    r, v = rv(orbit, time, propagator=propagator)  # (n, m, 3)
    if obsAngleCorrection:
        r, v = _obsAngleCorrection(
            r, v, obsPos, obsVel, orbit, time, propagator, obsAngleCorrection
        )

    ra, dec, slantRange, raRate, decRate, rangeRate = rvObsToRaDecRate(
        r, v, obsPos, obsVel
    )
    res = (ra, dec, slantRange)
    if rate:
        res = res + (raRate, decRate, rangeRate)

    return _doSqueeze(*((squeezeOrbit, squeezeTime) + res))


def altaz(
    orbit, time, observer, propagator=KeplerianPropagator(),
    obsAngleCorrection=None
):
    """ Calculate observed altitude and azimuth of orbiting objects as viewed at
    specified times and locations.

    Parameters
    ----------
    orbit : Orbit or list of Orbit (n,)
        Orbits for which to calculate direction cosines.
    time : array_like or astropy.time.Time (m,)
        If float (array), then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    observer : EarthObserver
        Where on Earth to compute alt and az.
    propagator : Propagator, optional
        The propagator instance to use.
    obsAngleCorrection : {None, "linear", "exact"}, optional
        Correct actual angle to observed angle, meaning account for light-time
        delay, and aberration due to the observer's velocity.  None means don't
        do any correction.  "linear" means do an aberration correction and first
        order light-time correction.  "exact" means do an aberration correction
        and iteratively solve for the exact light-time correction.  (The
        "linear" correction is almost always sufficiently accurate).

    Notes
    -----
    If `orbit` is scalar-valued (an Orbit instead of a list of Orbit), then that
    dimension will be squeezed out in the return value.  Likewise, if `time` is
    scalar, that dimension will be squeezed out.

    For Keplerian orbit propagation it is more efficient to use a "vector Orbit"
    instead of a list of single scalar Orbits.

    When doing light time corrections, the time argument is the arrival time of
    the photons at the observer, as opposed to the emission time at the
    satellite.

    Returns
    -------
    alt, az : array_like (n, m)
        Altitude and azimuth in radians.
    """
    from astropy.coordinates import SkyCoord, AltAz, GCRS

    nTime, squeezeTime, time = _countTime(time)
    nOrbit, squeezeOrbit, orbit = _countOrbit(orbit)
    obsPos, obsVel = observer.getRV(time)

    r, v = rv(orbit, time, propagator=propagator)  # (n, m, 3)
    if obsAngleCorrection:
        r, v = _obsAngleCorrection(
            r, v, obsPos, obsVel, orbit, time, propagator, obsAngleCorrection
        )

    # Using astropy backend for AltAz computation
    sc = SkyCoord(
        x=r[..., 0], y=r[..., 1], z=r[..., 2],
        unit='m',
        representation_type='cartesian',
        frame=GCRS(obstime=Time(time, format='gps'))
    )
    aa = sc.transform_to(AltAz(location=observer._location))

    return _doSqueeze(squeezeOrbit, squeezeTime, aa.alt.radian, aa.az.radian)


def quickAltAz(
    orbit, time, observer, propagator=KeplerianPropagator(),
    obsAngleCorrection=None
):
    """ Quickly estimate observed altitude and azimuth of orbiting objects as
    viewed at specified times and locations.

    This algorithm approximates "up" as pointing directly away from the center
    of the Earth, instead of normal to the reference ellipsoid.  Use `altAz` if
    you want values wrt the reference ellipsoid.

    Parameters
    ----------
    orbit : Orbit or list of Orbit (n,)
        Orbits for which to calculate direction cosines.
    time : array_like or astropy.time.Time (m,)
        If float (array), then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    observer : EarthObserver
        Where on Earth to compute alt and az.
    propagator : Propagator, optional
        The propagator instance to use.
    obsAngleCorrection : {None, "linear", "exact"}, optional
        Correct actual angle to observed angle, meaning account for light-time
        delay, and aberration due to the observer's velocity.  None means don't
        do any correction.  "linear" means do an aberration correction and first
        order light-time correction.  "exact" means do an aberration correction
        and iteratively solve for the exact light-time correction.  (The
        "linear" correction is almost always sufficiently accurate).

    Notes
    -----
    If `orbit` is scalar-valued (an Orbit instead of a list of Orbit), then that
    dimension will be squeezed out in the return value.  Likewise, if `time` is
    scalar, that dimension will be squeezed out.

    For Keplerian orbit propagation it is more efficient to use a "vector Orbit"
    instead of a list of single scalar Orbits.

    When doing light time corrections, the time argument is the arrival time of
    the photons at the observer, as opposed to the emission time at the
    satellite.

    Returns
    -------
    alt, az : array_like (n, m)
        Altitude and azimuth in radians.
    """
    nTime, squeezeTime, time = _countTime(time)
    nOrbit, squeezeOrbit, orbit = _countOrbit(orbit)

    ro, vo = observer.getRV(time)  # (m, 3)
    r, v = rv(orbit, time, propagator=propagator)  # (n, m, 3)
    if obsAngleCorrection:
        r, v = _obsAngleCorrection(
            r, v, ro, vo, orbit, time, propagator, obsAngleCorrection
        )

    up = normed(ro)
    east = normed(vo)
    north = np.cross(up, east)

    dr = r - ro
    northing = np.einsum("nmi,mi->nm", dr, north)
    easting = np.einsum("nmi,mi->nm", dr, east)

    alt = np.pi / 2 - unitAngle3(
        normed(np.broadcast_to(ro, r.shape)), normed(dr)
    )
    az = np.arctan2(easting, northing)
    az %= 2 * np.pi

    return _doSqueeze(squeezeOrbit, squeezeTime, alt, az)


def radecRate(
    orbit, time, obsPos=None, obsVel=None, observer=None,
    propagator=KeplerianPropagator(),
    obsAngleCorrection=None
):
    """Calculate ra/dec rate and slant range rate of orbit at specified times
    and observer positions and velocities.

    DEPRECATED.  Use radec(..., rate=True) in new code.

    Parameters
    ----------
    orbit : Orbit or list of Orbit (n,)
        Orbit(s) for which to calculate slant range rate.
    time : array_like or astropy.time.Time (m,)
        If float (array), then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    obsPos : array_like (m, 3), optional
        Positions of observers at given times.
    obsVel : array_like (m, 3), optional
        Velocity of observers at given times.
    observer : Observer or list of Observers (m,), optional
        Observer(s) used to calculate obsPos and obsVel.
    propagator : Propagator, optional
        The propagator instance to use.
    obsAngleCorrection : {None, "linear", "exact"}, optional
        Correct actual angle to observed angle, meaning account for light-time
        delay, and aberration due to the observer's velocity.  None means don't
        do any correction.  "linear" means do an aberration correction and first
        order light-time correction.  "exact" means do an aberration correction
        and iteratively solve for the exact light-time correction.  (The
        "linear" correction is almost always sufficiently accurate).

    Notes
    -----
    Exactly 1 of `obsPos` and `observer` must be supplied.  `observer` and
    `obsPos` follow similar broadcasting rules as detailed below explicitly only
    for `obsPos`.  If `obsPos` is specified, `obsVel` must also be specified and
    congruent to `obsPos`.

    The length of `time` and `obsPos` must match or be broadcastable to match.
    If `orbit` is scalar-valued (an Orbit instead of a list of Orbit), then that
    dimension will be squeezed out in the return value.  Likewise, if both
    `time` and `obsPos` are scalar, that dimension will be squeezed out.

    For Keplerian orbit propagation it is more efficient to use a "vector Orbit"
    instead of a list of single scalar Orbits.

    When doing light time corrections, the time argument is the arrival time of
    the photons at the observer, as opposed to the emission time at the
    satellite.

    Returns
    -------
    ra : array_like (n, m)
        right ascension in radians
    raRate : array_like (n, m)
        Rate of change of right ascension*cos(dec) in radians per second.
    dec : array_link (n, m)
        declination in radians
    decRate : array_like (n, m)
        Rate of change of declination in radians per second.
    slantRange : array_like (n, m)
        Range in meters
    slantRangeRate : array_like (n, m)
        Slant range rate in meters per second.
    """
    import warnings
    warnings.warn("This function is deprecated; use "
                  "ssapy.compute.radec(..., rate=True)")
    ra, dec, slant, raRate, decRate, slantRate = radec(
        orbit, time, obsPos=obsPos, obsVel=obsVel, observer=observer,
        propagator=propagator, obsAngleCorrection=obsAngleCorrection
    )

    return raRate, decRate, slantRate


def rvObsToRaDecRate(r, v, obsPos=None, obsVel=None):
    """Convert object and observer position and velocity to angles.

    This only does the geometric part; it ignores light travel time,
    which may be applied to the object location before input.  Assumes
    that r, v, obsPos, and obsVel all have common shape.

    Parameters
    ----------
    r : array_like (..., 3)
        object position in meters
    v : array_like (..., 3)
        object velocity in meters per second
    obsPos : array_like (..., 3)
        observer position in meters
    obsVel : array_like (..., 3), optional
        observer velocity in meters per second

    Returns
    -------
    ra : array_like (...)
        right ascension in radians
    dec : array_link (...)
        declination in radians
    slantRange : array_like (...)
        Range in meters
    raRate : array_like (...)
        Rate of change of right ascension*cos(dec) in radians per second.
    decRate : array_like (...)
        Rate of change of declination in radians per second.
    slantRangeRate : array_like (...)
        Slant range rate in meters per second.
    """
    if obsPos is None:
        obsPos = np.zeros_like(r)

    dr = r - obsPos
    slantRange = norm(dr)
    ra = np.arctan2(dr[..., 1], dr[..., 0])
    dec = np.arcsin(dr[..., 2] / slantRange)

    if obsVel is None:
        obsVel = np.zeros_like(v)
    dv = v - obsVel

    # Now need to rotate to ra/dec coords
    sd, cd = np.sin(dec), np.cos(dec)
    sa, ca = np.sin(ra), np.cos(ra)

    rHat = normed(dr)  # (n, m, 3)

    raHat = np.zeros_like(rHat, dtype=float)
    raHat[..., 0] = -sa
    raHat[..., 1] = ca

    decHat = np.zeros_like(raHat)
    decHat[..., 0] = -sd * ca
    decHat[..., 1] = -sd * sa
    decHat[..., 2] = cd

    raRate = np.einsum("...i,...i->...", dv, raHat) / slantRange
    decRate = np.einsum("...i,...i->...", dv, decHat) / slantRange
    rangeRate = np.einsum("...i,...i->...", dv, rHat)
    return ra, dec, slantRange, raRate, decRate, rangeRate


def radecRateObsToRV(ra, dec, slantRange, raRate=None, decRate=None, slantRangeRate=None, obsPos=None, obsVel=None):
    """Convert object angles and observer position to 3D observer position

    This only does the geometric part; it ignores light travel time.  This
    is the inverse of rvObsToRaDecRate.

    If obsVel is None, then the returned velocity will also be None.

    Parameters
    ----------
    ra : array_like (...)
        right ascension in radians
    dec : array_like (...)
        declination in radians
    slantRange : array_like (...)
        Range in meters
    raRate : array_like (...)
        Rate of change of right ascension*cos(dec) in radians per second.
    decRate : array_like (...)
        Rate of change of declination in radians per second.
    slantRangeRate : array_like (...)
        Slant range rate in meters per second.
    obsPos : array_like (..., 3)
        Observer position in meters
    obsVel : array_like (..., 3)
        Observer velocity in meters

    Returns
    -------
    r : array_like (..., 3)
        object position in meters
    v : array_like (..., 3)
        object velocity in meters per second
        observer velocity in meters per second

    v is None if obsVel is None.
    """

    if obsPos is None:
        raise ValueError('obsPos must be set!')

    rHat = lb_to_unit(ra, dec)

    sd, cd = np.sin(dec), np.cos(dec)
    sa, ca = np.sin(ra), np.cos(ra)

    raHat = np.zeros_like(rHat, dtype=float)
    raHat[..., 0] = -sa
    raHat[..., 1] = ca

    decHat = np.zeros_like(raHat)
    decHat[..., 0] = -sd * ca
    decHat[..., 1] = -sd * sa
    decHat[..., 2] = cd
    if len(rHat.shape) > 1:
        slantRange = np.array(slantRange)[..., None]
        raRate = np.array(raRate)[..., None]
        decRate = np.array(decRate)[..., None]
        slantRangeRate = np.array(slantRangeRate)[..., None]

    r = obsPos + rHat * slantRange
    if obsVel is None:
        v = None
    else:
        v = (obsVel + rHat * slantRangeRate + slantRange * (raHat * raRate + decHat * decRate))
    return r, v


def earthShadowCoords(r, time):
    """Determine components of position `r` parallel and perpendicular to sun
    unit vector.

    The sun unit vector points from the center of the sun through the center of
    the Earth.  Decomposing a satellite position into these coordinates yields a
    simple model of whether or not the satellite is in Earth's shadow:

        r_par, r_perp = earthShadowCoords(r, time)
        inShadow = r_par > 0 and r_perp < EARTH_RADIUS

    Parameters
    ----------
    r : ndarray (3,)
        Position (GCRF) in meters
    time : float or astropy.time.Time
        If float, then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC

    Returns
    -------
    r_par : float
        Position of satellite projected onto the sun unit vector in meters.
    r_perp : float
        Distance of satellite from Earth-Sun line in meters.
    """
    if isinstance(time, Time):
        time = time.gps
    r_sun = sunPos(time, fast=False)
    if r.ndim == 1:
        n_sun = - r_sun / norm(r_sun)
        r_par = np.dot(r, n_sun)
        r_perp = norm(r - n_sun * r_par)
    else:
        n_sun = - r_sun / norm(r_sun)[:, None]
        r_par = np.sum(r * n_sun, axis=1)
        r_perp = norm(r - n_sun * r_par[:, None])
    return r_par, r_perp


def find_passes(
    orbit, observers,
    tStart, tSpan, dt,
    propagator=KeplerianPropagator(),
    horizon=np.deg2rad(20)
):
    """Find satellite overhead passes for a collection of observers.

    Uses a brute force test of a grid of time points from tStart to tStart+tSpan
    separated by dt.

    Returns passes even if they occur during the daytime or if the satellite is
    not illuminated by the sun.  The only criterion for a successful "pass" is
    for the topocentric altitude of the satellite to be above the input
    `horizon`.  More details about a pass can subsequently be obtained by
    running the `refine_passes` function.

    Note this function is only suitable for `EarthObserver`s and not
    `OrbitalObserver`s.

    Parameters
    ----------
    orbit : Orbit
        Satellite orbit in question.
    observers : List of EarthObserver.
        Earth observers for which to check satellite visibility.
    tStart : float or astropy.time.Time
        Beginning of search window.
        If float, then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    tSpan : float or Quantity
        Time span in which to search for passes.
        If float, then seconds.
    dt : float or Quantity
        Time increment to use during search.  Satellite visibility will be
        computed every dt increment.  Smaller values will decrease the
        probability that a short duration pass will be missed, but will make the
        search take longer to complete.
        If float, then seconds.
    propagator : Propagator, optional
        The propagator instance to use.
    horizon : float or Quantity, optional
        Minimum altitude for which to consider a satellite "visible".
        If float, then should be in radians.

    Returns
    -------
    passDict : dict
        keys are `EarthObserver`s.
        values are lists (possibly empty) of `astropy.Time` corresponding to
        visible passes of the satellite.  Only one time is returned per pass,
        so multiple times in the return list indicate multiple distinct passes.
    """
    import astropy.units as u
    if isinstance(tStart, Time):
        tStart = tStart.gps
    if isinstance(tSpan, u.Quantity):
        tSpan = tSpan.to(u.s).value
    if isinstance(dt, u.Quantity):
        dt = dt.to(u.s).value
    if isinstance(horizon, u.Quantity):
        horizon = horizon.to(u.rad).value
    times = np.arange(tStart, tStart + tSpan, dt)
    out = {}
    for observer in observers:
        out[observer] = []
        alt, _ = quickAltAz(orbit, times, observer, propagator=propagator)
        w = np.nonzero(alt > horizon)[0]
        for wi in w:
            if (wi - 1) in w:  # already caught this pass
                continue
            out[observer].append(Time(times[wi], format='gps'))
    return out


def refine_pass(
    orbit,
    observer,
    time,
    propagator=KeplerianPropagator(),
    horizon=np.deg2rad(20),
    maxSpan=86400.0
):
    """Refine a satellite overhead pass.

    Parameters
    ----------
    orbit : Orbit
        Orbit in question.
    observer : EarthObserver.
        Observer for which to refine pass.
    time : float or astropy.time.Time
        A time when satellite is visible to observer.  (Found using
        find_passes, for instance)
        If float, then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    propagator : Propagator, optional
        The propagator instance to use.
    horizon : float or Quantity, optional
        Minimum altitude for which to consider a satellite "visible".
        If float, then should be in radians.
    maxSpan : float or Quantity, optional
        Maximum amount of time before or after `time` to search for
        rise/set times, or time of max altitude.
        If float, then seconds.

    Returns
    -------
    dict
        Key/values are:
        tStart : Time
            Time at which Orbit rises above horizon
        tEnd : Time
            Time at which Orbit rises above horizon
        tMaxAlt : Time
            Time Orbit passes through maximum altitude
        maxAlt : Quantity
            Maximum altitude
        duration : Quantity
            Duration of pass.
        illumAtStart : bool
            Is satellite illuminated at `tStart`?
        illumAtEnd : bool
            Is satellite illuminated at `tEnd`?
        tTerminator : Time or None
            If illumAtStart != illumAtEnd, then time satellite passes through
            cylindrical terminator shadow.  Otherwise, None.
        sunAltStart : Quantity
            Altitude of sun at `tStart`
        sunAltEnd : Quantity
            Altitude of sun at `tEnd`
    """
    from scipy.optimize import bisect, minimize_scalar
    import astropy.units as u
    import warnings

    if isinstance(time, Time):
        time = time.gps
    if isinstance(horizon, u.Quantity):
        horizon = horizon.to(u.rad).value
    if isinstance(maxSpan, u.Quantity):
        maxSpan = maxSpan.to(u.s).value

    def dalt(t):
        return (
            horizon - quickAltAz(orbit, t, observer, propagator=propagator)[0]
        )

    # Height "outside" of cylindrical Earth shadow model.
    def dshadow(t):
        r, _ = rv(orbit, t, propagator=propagator)
        r_par, r_perp = earthShadowCoords(r, t)
        height = r_perp - EARTH_RADIUS
        if r_par < 0:
            # a bit of a hack, if the sat is on the sun side, then we'll
            # return abs(dshadow) so that the value is positive, which indicates
            # illumination.
            return abs(height)
        else:
            return height

    # Bracket and then bisect to find rise/set times
    def bracket(f, x, dx, xmax):
        x0, x1 = x, x + dx
        f0, f1 = f(x0), f(x1)
        while f0 * f1 > 0 and abs(x1 - x0) < xmax:
            x1 += dx
            f1 = f(x1)
        return x1

    tLow = bracket(dalt, time, -300, maxSpan)
    if dalt(tLow) * dalt(time) > 0:  # didn't bracket, use tLow and issue warning
        warnings.warn("Failed to bracket.  tStart is not rise time!")
        tStart = tLow
    else:
        tStart = bisect(dalt, tLow, time)
    tStart = Time(tStart, format='gps')
    tStart.format = 'iso'

    tHigh = bracket(dalt, time, 300, maxSpan)
    if dalt(tHigh) * dalt(time) > 0:
        warnings.warn("Failed to bracket.  tEnd is not set time!")
        tEnd = tHigh
    else:
        tEnd = bisect(dalt, time, tHigh)
    tEnd = Time(tEnd, format='gps')
    tEnd.format = 'iso'

    # Find maximum altitude
    dalt1 = dalt(tStart.gps)
    dalt2 = dalt(time)
    dalt3 = dalt(tEnd.gps)
    if dalt2 < dalt1 and dalt2 < dalt3:
        result = minimize_scalar(dalt, (tStart.gps, time, tEnd.gps))
        tMaxAlt = Time(result.x, format='gps')
        tMaxAlt.format = 'iso'
        maxAlt = -result.fun + horizon
    else:
        ama = np.argmin([dalt1, dalt2, dalt3])
        tMaxAlt = Time([tStart.gps, time, tEnd.gps][ama], format='gps')
        tMaxAlt.format = 'iso'
        maxAlt = -[dalt1, dalt2, dalt3][ama] + horizon
    duration = (tEnd - tStart).to(u.min)

    # Find illumination state
    illumAtStart = dshadow(tStart.gps) > 0
    sunAltStart = np.rad2deg(observer.sunAlt(tStart.gps)) * u.deg
    illumAtEnd = dshadow(tEnd.gps) > 0
    sunAltEnd = np.rad2deg(observer.sunAlt(tEnd.gps)) * u.deg

    # If we transition between shadow/illuminated, then find the time of
    # transition (time of terminator pass)
    if illumAtStart != illumAtEnd:
        tTerminator = bisect(dshadow, tStart.gps, tEnd.gps)
        tTerminator = Time(tTerminator, format='gps')
        tTerminator.format = 'iso'
    else:
        tTerminator = None

    return {
        "orbit": orbit,
        "observer": observer,
        "propagator": propagator,
        "horizon": horizon,
        "tStart": tStart,
        "tEnd": tEnd,
        "altStart": np.rad2deg(-dalt(tStart.gps) + horizon) * u.deg,
        "altEnd": np.rad2deg(-dalt(tEnd.gps) + horizon) * u.deg,
        "tMaxAlt": tMaxAlt,
        "maxAlt": np.rad2deg(maxAlt) * u.deg,
        "duration": duration,
        "illumAtStart": illumAtStart,
        "illumAtEnd": illumAtEnd,
        "tTerminator": tTerminator,
        "sunAltStart": sunAltStart,
        "sunAltEnd": sunAltEnd
    }
