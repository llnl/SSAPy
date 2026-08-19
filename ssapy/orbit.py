"""
Module to handle Earth satellite state vector manipulation and propagation.

Notes
-----
Although much of the Orbit class here is agnostic with respect to the
coordinate reference frame, we strongly recommend that positions and
velocities be specified in the GCRF frame.  Some classes in this module *do*
insist on this frame, (e.g., EarthObserver).  We will try to indicate where
the GCRF frame is required, but make no guarantees that we don't miss any
such instances.
"""


import numpy as np
import astropy
from .utils import (
    normSq, normed, lazy_property as _lazy_property, unitAngle3, sunPos,
    _gpsToTT, _wrapToPi, teme_to_gcrf, gcrf_to_teme, iers_interp
)
from .constants import EARTH_MU
from .propagator import KeplerianPropagator as _KeplerianPropagator

try:
    import erfa
except ImportError:
    # Let this raise
    import astropy._erfa as erfa


# Conversion routines for anomalies
def _ellipticalEccentricToTrueAnomaly(E, e):
    """Compute true anomaly from eccentric anomaly for elliptical orbit.

    Parameters
    ----------
    E : array_like
        Eccentric anomaly in radians.
    e : float
        Eccentricity

    Returns
    -------
    array_like
        True anomaly in radians.
    """
    beta = e / (1 + np.sqrt((1 - e) * (1 + e)))
    return E + 2 * np.arctan(beta * np.sin(E) / (1 - beta * np.cos(E)))


def _ellipticalTrueToEccentricAnomaly(v, e):
    """Compute eccentric anomaly from true anomaly for elliptical orbit.

    Parameters
    ----------
    v : array_like
        True anomaly in radians.
    e : float
        Eccentricity

    Returns
    -------
    array_like
        Eccentric anomaly in radians.
    """
    beta = e / (1 + np.sqrt(1 - e * e))
    return v - 2 * np.arctan(beta * np.sin(v) / (1 + beta * np.cos(v)))


def _ellipticalTrueToEccentricAnomalyMany(v, e):
    beta = e / (1 + np.sqrt(1 - e * e))
    return v - 2 * np.arctan(beta[:, None] * np.sin(v) / (1 + beta[:, None] * np.cos(v)))


def _hyperbolicEccentricToTrueAnomaly(H, e):
    """Compute hyperbolic true anomaly from hyperbolic eccentric anomaly for
    hyperbolic orbit.

    Parameters
    ----------
    H : array_like
        Hyperbolic eccentric anomaly in radians.
    e : float
        Eccentricity

    Returns
    -------
    array_like
        Hyperbolic true anomaly in radians.
    """
    return 2. * np.arctan(np.sqrt((e + 1) / (e - 1)) * np.tanh(H / 2))


def _hyperbolicEccentricToTrueAnomalyMany(H, e):
    return 2. * np.arctan(np.sqrt((e + 1) / (e - 1))[:, None] * np.tanh(H / 2))


def _hyperbolicTrueToEccentricAnomaly(v, e):
    """Compute hyperbolic eccentric anomaly from hyperbolic true anomaly for
    hyperbolic orbit.

    Parameters
    ----------
    v : array_like
        Hyperbolic true anomaly in radians.
    e : float
        Eccentricity

    Returns
    -------
    array_like
        Hyperbolic eccentric anomaly in radians.
    """
    return np.arcsinh(np.sqrt(e * e - 1) * np.sin(v) / (1 + e * np.cos(v)))


def _hyperbolicTrueToEccentricAnomalyMany(v, e):
    return np.arcsinh(
        np.sqrt(e * e - 1)[:, None] * np.sin(v) / (1 + e[:, None] * np.cos(v))
    )


def _ellipticalEccentricToMeanAnomaly(E, e):
    """Compute mean anomaly from eccentric anomaly for elliptical orbit.

    Parameters
    ----------
        E : array_like
            Eccentric anomaly in radians.
        e : float
            Eccentricity

    Returns
    -------
    array_like
        Mean anomaly in radians.
    """
    return E - e * np.sin(E)


@np.vectorize
def _ellipticalMeanToEccentricAnomaly(M, e):
    """Compute eccentric anomaly from mean anomaly for elliptical orbit.

    Parameters
    ----------
    M : array_like
        Mean anomaly in radians.
    e : float
        Eccentricity

    Returns
    -------
    array_like
        Eccentric anomaly in radians.
    """
    k1 = 3 * np.pi + 2
    k2 = np.pi - 1
    k3 = 6 * np.pi - 1
    A = 3 * k2 * k2 / k1
    B = k3 * k3 / (6 * k1)

    MM = _wrapToPi(M)  # map to [-Pi, Pi]
    if np.abs(MM) < 1. / 6:
        E = MM + e * (np.cbrt(6 * MM) - MM)
    else:
        if MM < 0:
            w = MM + np.pi
            E = MM + e * (A * w / (B - w) - np.pi - MM)
        else:
            w = MM - np.pi
            E = MM + e * (np.pi - A * w / (B - w) - MM)
    e1 = 1 - e
    noCancellationRisk = (e1 + E * E / 6) >= 0.1
    # three iterations.  each with one Halley and one Newton-Raphson step
    for _ in range(3):
        fdd = e * np.sin(E)
        fddd = e * np.cos(E)
        if noCancellationRisk:
            f = (E - fdd) - MM
            fd = 1 - fddd
        else:
            f = _eMeSinE(E, e) - MM
            s = np.sin(0.5 * E)
            fd = e1 + 2 * e * s * s
        dee = f * fd / (0.5 * f * fdd - fd * fd)

        w = fd + 0.5 * dee * (fdd + dee * fddd / 3)
        fd += dee * (fdd + 0.5 * dee * fddd)
        E -= (f - dee * (fd - w)) / fd
    E += M - MM
    return E


@np.vectorize
def _eMeSinE(E, e):
    """Accurate computation of E - e sin(E).
    Use when E is close to 0 and e close to 1,
    i.e., near periapsis of almost parabolic orbits.

    Parameters
    ----------
    E : array_like
        Eccentric anomaly in radians.
    e : float
        Eccentricity

    Returns
    -------
    array_like
        E - e sin(E)
    """
    x = (1 - e) * np.sin(E)
    mE2 = -E * E
    term = E
    d = 0
    x0 = np.nan
    while x != x0:
        d += 2
        term *= mE2 / (d * (d + 1))
        x0 = x
        x = x - term
    return x


def _hyperbolicEccentricToMeanAnomaly(H, e):
    """Compute hyperbolic mean anomaly from hyperbolic eccentric anomaly for
    hyperbolic orbit.

    Parameters
    ----------
    H : array_like
        Hyperbolic eccentric anomaly in radians.
    e : float
        Eccentricity

    Returns
    -------
    array_like
        Hyperbolic mean anomaly in radians.
    """
    return e * np.sinh(H) - H


@np.vectorize
def _hyperbolicMeanToEccentricAnomaly(M, e):
    """Compute hyperbolic eccentric anomaly from hyperbolic mean anomaly for
    hyperbolic orbit.

    Parameters
    ----------
    M : array_like
        Hyperbolic mean anomaly in radians.
    e : float
        Eccentricity

    Returns
    -------
    array_like
        Hyperbolic eccentric anomaly in radians.
    """
    if e < 1.6:
        if (-np.pi < M and M < 0.) or M > np.pi:
            H = M - e
        else:
            H = M + e
    else:
        if (e < 3.6 and abs(M) > np.pi):
            H = M - np.copysign(e, M)
        else:
            H = M / (e - 1)
    iter = 0
    while iter < 50:
        f3 = e * np.cosh(H)
        f2 = e * np.sinh(H)
        f1 = f3 - 1
        f0 = f2 - H - M
        f12 = 2 * f1
        d = f0 / f12
        fdf = f1 - d * f2
        ds = f0 / fdf
        shift = f0 / (fdf + ds * ds * f3 / 6)
        H -= shift
        if abs(shift) < 1e-12:
            return H
        iter += 1
    raise RuntimeError(
        "Unable to compute hyperbolic eccentric anomaly for M={}, e={}"
        .format(M, e)
    )


# Conversion routines for longitudes
def _ellipticalEccentricToTrueLongitude(lE, ex, ey):
    """Compute true longitude from eccentric longitude for elliptical orbit.

    Parameters
    ----------
    lE : array_like
        Eccentric longitude in radians.
    ex, ey : float
        Eccentricity vector components.

    Returns
    -------
    array_like
        True longitude in radians.
    """
    epsilon = np.sqrt(1 - ex * ex - ey * ey)
    cosLE = np.cos(lE)
    sinLE = np.sin(lE)
    num = ex * sinLE - ey * cosLE
    den = epsilon + 1 - ex * cosLE - ey * sinLE
    return lE + 2 * np.arctan(num / den)


def _ellipticalTrueToEccentricLongitude(lv, ex, ey):
    """Compute eccentric longitude from true longitude for elliptical orbit.

    Parameters
    ----------
    lv : array_like
        True longitude in radians.
    ex, ey : float
        Eccentricity vector components.

    Returns
    -------
    array_like
        Eccentric longitude in radians.
    """
    epsilon = np.sqrt(1 - ex * ex - ey * ey)
    cosLv = np.cos(lv)
    sinLv = np.sin(lv)
    num = ey * cosLv - ex * sinLv
    den = epsilon + 1 + ex * cosLv + ey * sinLv
    return lv + 2 * np.arctan(num / den)


def _ellipticalTrueToEccentricLongitudeMany(lv, ex, ey):
    # lv (n, m)
    # ex, ey (n,)
    epsilon = np.sqrt(1 - ex * ex - ey * ey)
    cosLv = np.cos(lv)
    sinLv = np.sin(lv)
    num = ey[:, None] * cosLv - ex[:, None] * sinLv
    den = epsilon[:, None] + 1 + ex[:, None] * cosLv + ey[:, None] * sinLv
    return lv + 2 * np.arctan(num / den)


def _hyperbolicEccentricToTrueLongitude(lH, ex, ey):
    """Compute hyperbolic true longitude from hyperbolic eccentric longitude for
    hyperbolic orbit.

    Parameters
    ----------
    lH : array_like
        Hyperbolic eccentric longitude in radians.
    ex, ey : float
        Eccentricity vector components.

    Returns
    -------
    array_like
        Hyperbolic true longitude in radians.
    """
    # is there a way to do this directly w/o computing e, w?
    e = np.sqrt(ex * ex + ey * ey)
    w = np.arctan2(ey, ex)
    H = lH - w
    v = _hyperbolicEccentricToTrueAnomaly(H, e)
    return v + w


def _hyperbolicTrueToEccentricLongitude(lv, ex, ey):
    """Compute hyperbolic eccentric longitude from hyperbolic true longitude for
    hyperbolic orbit.

    Parameters
    ----------
    lv : array_like
        Hyperbolic true longitude in radians.
    ex, ey : float
        Eccentricity vector components.

    Returns
    -------
    array_like
        Hyperbolic eccentric longitude in radians.
    """
    e = np.sqrt(ex * ex + ey * ey)
    w = np.arctan2(ey, ex)
    v = lv - w
    H = _hyperbolicTrueToEccentricAnomaly(v, e)
    return H + w


def _hyperbolicTrueToEccentricLongitudeMany(lv, ex, ey):
    e = np.sqrt(ex * ex + ey * ey)
    w = np.arctan2(ey, ex)
    v = lv - w[:, None]
    H = _hyperbolicTrueToEccentricAnomaly(v, e[:, None])
    return H + w[:, None]


def _ellipticalEccentricToMeanLongitude(lE, ex, ey):
    """Compute mean longitude from eccentric longitude for elliptical orbit.

    Parameters
    ----------
    lE : array_like
        Eccentric longitude in radians.
    ex, ey : float
        Eccentricity vector components.

    Returns
    -------
    array_like
        Mean longitude in radians.
    """
    return lE - ex * np.sin(lE) + ey * np.cos(lE)


def _ellipticalMeanToEccentricLongitude(lM, ex, ey):
    """Compute eccentric longitude from mean longitude for elliptical orbit.

    Parameters
    ----------
    lM : array_like
        Mean longitude in radians.
    ex, ey : float
        Eccentricity vector components.

    Returns
    -------
    array_like
        Eccentric longitude in radians.
    """
    shift = 1.0
    lEmlM = 0.0
    lE = lM
    cosLE = np.cos(lE)
    sinLE = np.sin(lE)
    # previous version iterated until a given convergence criterion was met.
    # This version always iterates exactly 6 times.  I found this is ~50% faster
    # in typical use (as part of RVSampler) since the operations are all
    # vectorized.
    for iter in range(6):
        f2 = ex * sinLE - ey * cosLE
        f1 = 1 - ex * cosLE - ey * sinLE
        f0 = lEmlM - f2
        f12 = 2 * f1
        shift = f0 * f12 / (f1 * f12 - f0 * f2)
        lEmlM -= shift
        lE = lM + lEmlM
        cosLE = np.cos(lE)
        sinLE = np.sin(lE)
    return lE


# Specialization for parallelizing over orbits and times simultaneously
def _ellipticalMeanToEccentricLongitudeMany(lM, ex, ey):
    shift = 1.0
    lEmlM = 0.0
    lE = lM
    cosLE = np.cos(lE)
    sinLE = np.sin(lE)
    # Always iterate exactly 6 times.  See above.
    for iter in range(6):
        f2 = np.array(sinLE)
        f1 = np.array(cosLE)

        f2 *= ex[:, None]
        cosLE *= ey[:, None]
        f2 -= cosLE

        f1 *= -ex[:, None]
        sinLE *= ey[:, None]
        f1 -= sinLE
        f1 += 1
        # above is equivalent to commented lines below, but rewritten to avoid
        # temporary array allocation as much as reasonable:
        # f2 = ex[:,None]*sinLE - ey[:,None]*cosLE
        # f1 = 1 - ex[:,None]*cosLE - ey[:,None]*sinLE
        f0 = lEmlM - f2

        shift = np.array(f0)
        shift *= f1
        shift *= 2
        f1 *= f1
        f1 *= 2
        f0 *= f2
        f1 -= f0
        shift /= f1
        # above block is equivalent to below
        # shift = f0*f12/(f1*f12 - f0*f2)

        lEmlM -= shift
        lE = lM + lEmlM
        cosLE = np.cos(lE)
        sinLE = np.sin(lE)
    return lE


def _hyperbolicEccentricToMeanLongitude(lH, ex, ey):
    """Compute hyperbolic mean longitude from hyperbolic eccentric longitude for
    hyperbolic orbit.

    Parameters
    ----------
    lH : array_like
        Hyperbolic eccentric longitude in radians.
    ex, ey : float
        Eccentricity vector components.

    Returns
    -------
    array_like
        Hyperbolic mean longitude in radians.
    """
    e = np.sqrt(ex * ex + ey * ey)
    w = np.arctan2(ey, ex)
    H = lH - w
    M = _hyperbolicEccentricToMeanAnomaly(H, e)
    return M + w


def _hyperbolicMeanToEccentricLongitude(lM, ex, ey):
    """Compute hyperbolic eccentric longitude from hyperbolic mean longitude for
    hyperbolic orbit.

    Parameters
    ----------
    lM : array_like
        Hyperbolic mean longitude in radians.
    ex, ey : float
        Eccentricity vector components.

    Returns
    -------
    array_like
        Hyperbolic eccentric longitude in radians.
    """
    e = np.sqrt(ex * ex + ey * ey)
    w = np.arctan2(ey, ex)
    M = lM - w
    H = _hyperbolicMeanToEccentricAnomaly(M, e)
    return H + w


def _hyperbolicMeanToEccentricLongitudeMany(lM, ex, ey):
    e = np.sqrt(ex * ex + ey * ey)
    w = np.arctan2(ey, ex)
    M = lM - w[:, None]
    H = _hyperbolicMeanToEccentricAnomaly(M, e[:, None])
    return H + w[:, None]


class Orbit:
    """
    Orbital state of one or more objects.

    This class represents one or more instantaneous (osculating) states of
    orbiting bodies.  Representations in Cartesian, Keplerian, and equinoctial
    elements are available.  The default initializer requires Cartesian
    elements, though class methods are also available to initialize with
    Keplerian or equinoctial elements.

    Note that generally this class can represent either a single scalar orbit,
    in which case attributes will generally be scalar floats, or a vector of
    orbits, in which case attributes will be ndarrays of floats.  For attributes
    that are intrinsically vectors even for a single orbit (position, velocity,
    ...), a 2d array will be used for a "vector of orbits", with the first
    dimension representing the different orbits.

    For simplicity, we will only indicate the "single scalar Orbit" dimensions
    in docstrings.  For a vector-Orbit, most scalar input arguments can be
    supplied as broadcastable arrays, and scalar attributes become array
    attributes.  When a multi-dimensional array is required, the first dimension
    is the one over different orbits.

    Parameters
    ----------
    r : (3,) array_like
        Position of orbiting object in meters.
    v : (3,) array_like
        Velocity of orbiting objects in meters per second.
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC
    mu : float, optional
        Gravitational constant of central body in m^3/s^2.  (Default: Earth's
        gravitational constant in WGS84).

    Attributes
    ----------
    r : (3,) array_like
        Position of orbiting object in meters.
    v : (3,) array_like
        Velocity of orbiting object in meters per second.
    t : float
        GPS seconds; i.e., seconds since 1980-01-06 00:00:00 UTC
    mu : float
        Gravitational constant of central body in m^3/s^2
    a : float
        Semimajor axis in meters.
    hx, hy : float
        Components of the equinoctial inclination vector.
    ex, ey : float
        Components of the equinoctial eccentricity vector.
    lv : float
        True longitude in radians.
    lE : float
        Eccentric longitude in radians.
    lM : float
        Mean longitude in radians.
    e : float
        Keplerian eccentricity.
    i : float
        Keplerian inclination in radians.
    pa : float
        Keplerian periapsis argument in radians.
    raan : float
        Keplerian right ascension of the ascending node in radians.
    trueAnomaly : float
        Keplerian true anomaly in radians.
    eccentricAnomaly : float
        Keplerian eccentric anomaly in radians.
    meanAnomaly : float
        Keplerian mean anomaly in radians.
    period : float
        Orbital period in seconds.
    meanMotion : float
        Keplerian mean motion in radians per second.
    p : float
        Semi-latus rectum in meters.
    angularMomentum : (3,) array_like
        (Specific) angular momentum in m^2/s.
    energy: float
        (Specific) orbital energy in m^2/s^2.
    LRL : (3,) array_like
        Laplace-Runge-Lenz vector in m^3/s^2.
    periapsis : (3,) array_like
        Periapsis coordinate of orbit in meters.
    apoapsis : (3,) array_like
        Apoapsis coordinate of orbit in meters.
    equinoctialElements
    keplerianElements

    Methods
    -------
    fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, t, mu)
        Construct an `Orbit` from Keplerian elements.
    fromEquinoctialElements(a, hx, hy, ex, ey, lv, t, mu)
        Construct an `Orbit` from Equinoctial elements.
    at(t, propagator)
        Propagate orbit to time t and return new Orbit.

    Notes
    -----
    Although much of the Orbit class here is agnostic with respect to the
    coordinate reference frame, we strongly recommend that positions and
    velocities be specified in the GCRF frame.  Some other classes in this
    module *do* insist on this frame, (e.g., EarthObserver).  We will try to
    indicate where the GCRF frame is required, but make no guarantees that we
    don't miss any such instances.
    """
    # Internally use equinoctial elements, calculating cartesian and keplerian
    # on demand.  Main ctor uses cartesian though
    def __init__(self, r, v, t, mu=EARTH_MU, propkw=None):
        if isinstance(t, astropy.time.Time):
            t = t.gps
        self.r, self.v = np.broadcast_arrays(r, v)
        self.propkw = dict() if propkw is None else propkw
        if self.r.ndim == 2:
            t = np.broadcast_to(t, self.r.shape[0])
            for key, value in self.propkw.items():
                self.propkw[key] = np.broadcast_to(value, self.r.shape[0])
        self.t = t
        self.mu = mu

    def __len__(self):
        if self.r.ndim == 1:
            return 1
        else:
            return self.r.shape[0]

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        if self._iter < len(self):
            # propagate kozai directly if possible
            if 'kozaiMeanKeplerianElements' in self.__dict__:
                kMKE = self.__dict__['kozaiMeanKeplerianElements']
            else:
                kMKE = None
            out = Orbit(
                self.r[self._iter],
                self.v[self._iter],
                self.t[self._iter],
                propkw={k: v[self._iter] for k, v in self.propkw.items()}
            )
            if kMKE is not None:
                out.kozaiMeanKeplerianElements = kMKE
            # Preserve native SGP4 record (Path A) so a TLE-derived orbit
            # reproduces direct sgp4 after scalar/vector round-trips.
            if hasattr(self, "_sat"):
                out._sat = self._sat
            if hasattr(self, "_tle"):
                out._tle = self._tle
            # TODO: iterate through already-instantiated lazy_properties and
            # copy them over too.
            self._iter += 1
            return out
        else:
            raise StopIteration

    def __getitem__(self, idx):
        if 'kozaiMeanKeplerianElements' in self.__dict__:
            kMKE = self.__dict__['kozaiMeanKeplerianElements']
        else:
            kMKE = None
        out = Orbit(self.r[idx], self.v[idx], self.t[idx],
                    propkw={k: v[idx] for k, v in self.propkw.items()})
        if kMKE is not None:
            out.kozaiMeanKeplerianElements = kMKE
        # Preserve native SGP4 record (Path A) on indexing.
        if hasattr(self, "_sat"):
            out._sat = self._sat
        if hasattr(self, "_tle"):
            out._tle = self._tle
        return out

    def __hash__(self):
        if not hasattr(self, '_hash'):
            self.r.flags.writeable = False
            self.v.flags.writeable = False
            if self.r.ndim == 1:
                self._hash = hash((
                    self.r.data.tobytes(),
                    self.v.data.tobytes(),
                    self.t,
                    self.mu) + tuple([(k, v.data.tobytes()) for k, v in self.propkw.items()])
                )
            else:
                self.t.flags.writeable = False
                self._hash = hash((
                    self.r.data.tobytes(),
                    self.v.data.tobytes(),
                    self.t.data.tobytes(),
                    self.mu) + tuple([(k, v.data.tobytes()) for k, v in self.propkw.items()])
                )
        return self._hash

    def __eq__(self, rhs):
        if self.r.ndim != rhs.r.ndim:
            return False
        else:
            return (
                np.all(self.r == rhs.r) and np.all(self.v == rhs.v) and np.all(self.mu == rhs.mu) and np.all(self.t == rhs.t) and np.all([np.all(self.propkw[k] == rhs.propkw[k]) for k in self.propkw]))

    @classmethod
    def fromEquinoctialElements(
        cls, a, hx, hy, ex, ey, lv, t, mu=EARTH_MU, propkw=None,
    ):
        """Construct an Orbit from equinoctial elements.

        Parameters
        ----------
        a : float
            Semimajor axis in meters.
        hx, hy : float
            Components of the equinoctial inclination vector.
        ex, ey : float
            Components of the equinoctial eccentricity vector.
        lv : float
            True longitude in radians.
        t : float or astropy.time.Time
            If float, then should correspond to GPS seconds; i.e., seconds since
            1980-01-06 00:00:00 UTC
        mu : float, optional
            Gravitational constant of central body in m^3/s^2.  (Default:
            Earth's gravitational constant in WGS84).

        Returns
        -------
        Orbit
            The Orbit with given parameters.
        """
        import numbers
        if isinstance(t, astropy.time.Time):
            t = t.gps
        if not all(isinstance(x, numbers.Real) for x in (a, hx, hy, ex, ey, lv, t)):
            a, hx, hy, ex, ey, lv, t = np.broadcast_arrays(
                a, hx, hy, ex, ey, lv, t
            )
            scalar = False
        else:
            scalar = True

        obj = cls.__new__(cls)
        obj.a = a
        obj.hx = hx
        obj.hy = hy
        obj.ex = ex
        obj.ey = ey
        obj.lv = lv
        obj.t = t
        obj.mu = mu

        # set reference axes of orbital plane
        cls._setPQEquinoctial(obj)

        # set position and velocity
        if scalar:
            r, v = cls._rvFromEquinoctial(obj, lv=np.atleast_2d(lv))
            obj.r = r[0, 0]
            obj.v = v[0, 0]
            obj.propkw = propkw if propkw is not None else dict()
        else:
            r, v = cls._rvFromEquinoctial(obj, lv=lv[:, None])
            obj.r = r[:, 0]
            obj.v = v[:, 0]
            obj.propkw = propkw if propkw is not None else dict()
            for k, v in obj.propkw.items():
                obj.propkw[k] = np.broadcast_to(v, obj.a.shape[0])
        return obj

    @classmethod
    def fromKeplerianElements(
        cls, a, e, i, pa, raan, trueAnomaly, t, mu=EARTH_MU, propkw=None,
    ):
        """Construct an Orbit from Keplerian elements.

        Parameters
        ----------
        a : float
            Semimajor axis in meters.
        e : float
            Keplerian eccentricity.
        i : float
            Keplerian inclination in radians.
        pa : float
            Keplerian periapsis argument in radians.
        raan : float
            Keplerian right ascension of the ascending node in radians.
        trueAnomaly : float
            Keplerian true anomaly in radians.
        t : float or astropy.time.Time
            If float, then should correspond to GPS seconds; i.e., seconds since
            1980-01-06 00:00:00 UTC
        mu : float, optional
            Gravitational constant of central body in m^3/s^2.  (Default:
            Earth's gravitational constant in WGS84).

        Returns
        -------
        Orbit
            The Orbit with given parameters.
        """
        import numbers
        if isinstance(t, astropy.time.Time):
            t = t.gps
        if not all(isinstance(x, numbers.Real) for x in (a, e, i, pa, raan, trueAnomaly, t)):
            a, e, i, pa, raan, trueAnomaly, t = np.broadcast_arrays(
                a, e, i, pa, raan, trueAnomaly, t
            )
            scalar = False
        else:
            scalar = True

        obj = cls.__new__(cls)
        obj.a = a
        obj.e = e
        obj.i = i
        obj.pa = pa
        obj.raan = raan
        obj.trueAnomaly = trueAnomaly
        obj.t = t
        obj.mu = mu

        # set reference axes of orbital plane
        cls._setPQKeplerian(obj)

        # set position and velocity
        if scalar:
            r, v = cls._rvFromKeplerian(obj, np.atleast_2d(trueAnomaly))
            obj.r = r[0, 0]
            obj.v = v[0, 0]
            obj.propkw = propkw if propkw is not None else dict()
        else:
            r, v = cls._rvFromKeplerian(obj, trueAnomaly[:, None])
            obj.r = r[:, 0]
            obj.v = v[:, 0]
            obj.propkw = propkw if propkw is not None else dict()
            for k, v in obj.propkw.items():
                obj.propkw[k] = np.broadcast_to(v, obj.a.shape[0])
        return obj

    @classmethod
    def fromKozaiMeanKeplerianElements(
        cls, a, e, i, pa, raan, trueAnomaly, t, mu=EARTH_MU,
        _useTEME=False, propkw=None
    ):
        """Construct an Orbit from Kozai mean Keplerian elements.  By default,
        this method also converts from the TEME reference frame to the GCRF
        frame.  This is mainly to support TLEs and SGP4, so the default
        gravitational parameter is WGS84.

        Parameters
        ----------
        a : float
            Semimajor axis in meters.
        e : float
            Keplerian eccentricity.
        i : float
            Keplerian inclination in radians.
        pa : float
            Keplerian periapsis argument in radians.
        raan : float
            Keplerian right ascension of the ascending node in radians.
        trueAnomaly : float
            Keplerian true anomaly in radians.
        t : float or astropy.time.Time
            If float, then should correspond to GPS seconds; i.e., seconds since
            1980-01-06 00:00:00 UTC
        mu : float, optional
            Gravitational constant of central body in m^3/s^2.  (Default:
            Earth's gravitational constant in WGS84).

        Returns
        -------
        Orbit
            The Orbit with given parameters.
        """
        # _useTEME hidden bool kwarg indicates to keep SGP4 result in TEME
        # frame, instead of transforming to GCRF.  The default of False is
        # almost always appropriate.  The exception is when fitting for kozai
        # elements where it's faster to just work in the TEME frame.
        from sgp4.api import Satrec, WGS84

        if a <= 0 or e >= 1:
            raise ValueError("Kozai mean elements require a bound orbit with a > 0 and e < 1")
        meanAnomaly = _ellipticalEccentricToMeanAnomaly(
            _ellipticalTrueToEccentricAnomaly(
                trueAnomaly % (2 * np.pi), e
            ),
            e
        )
        meanMotion = np.sqrt(mu / np.abs(a**3)) * 60.0  # rad/min

        if not isinstance(t, astropy.time.Time):
            t = astropy.time.Time(t, format='gps')
        mjd = t.mjd
        # epoch = mjd - astropy.time.Time("1949-12-31T00:00:00").mjd
        epoch = mjd - 33281.0

        sat = Satrec()
        sat.sgp4init(
            WGS84,           # gravity model
            'i',             # 'a' = old AFSPC mode, 'i' = improved mode
            0,               # satnum: Satellite number
            epoch,           # epoch: days since 1949 December 31 00:00 UT
            0.0,             # bstar: drag coefficient (kg/m2er)
            0.0,             # ndot: ballistic coefficient (revs/day)
            0.0,             # nddot: second derivative of mean motion (revs/day^3)
            e,               # ecco: eccentricity
            pa,              # argpo: argument of perigee (radians)
            i,               # inclo: inclination (radians)
            meanAnomaly,     # mo: mean anomaly (radians)
            meanMotion,      # no_kozai: mean motion (radians/minute)
            raan,            # nodeo: right ascension of ascending node (radians)
        )
        e, r, v = sat.sgp4_tsince(0)
        r = np.array(r)
        v = np.array(v)
        if not _useTEME:
            rot = teme_to_gcrf(t)
            r = rot @ r
            v = rot @ v
        r *= 1e3  # km -> m
        v *= 1e3  # km/s -> m/s
        return Orbit(r, v, t, mu=mu, propkw=propkw)

    @_lazy_property
    def kozaiMeanKeplerianElements(self):
        """Kozai mean Keplerian elements in TEME frame
        (a, e, i, pa, raan, trueAnomaly)
        """
        from scipy.optimize import least_squares
        # I don't know of a good closed form expression for Kozai elements, so
        # instead, just solve for them using SGP4 and a desired output state
        # vector.  The osculating elements are a good initial guess.

        rot = gcrf_to_teme(self.t)
        rTEME = rot @ self.r
        vTEME = rot @ self.v

        def resid(p):
            a, e, i, pa, raan, trueAnomaly = p
            orb = Orbit.fromKozaiMeanKeplerianElements(
                a, e, i, pa, raan, trueAnomaly,
                self.t,
                _useTEME=True  # stay in TEME frame
            )
            resid = np.hstack([(orb.r - rTEME) / 1e4, orb.v - vTEME])
            # print(f"{a:8.4f} {e:8.4f} {i:8.4f} {pa:8.4f} {raan:8.4f} {trueAnomaly:8.4f}", end='')
            # print(f"  {resid[0]:12.4f} {resid[1]:12.4f} {resid[2]:12.4f} {resid[3]:12.4f} {resid[4]:12.4f} {resid[5]:12.4f}")
            return resid
        lbounds = [0.0, 0.0, -np.inf, -np.inf, -np.inf, -np.inf]
        ubounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        opt = least_squares(
            resid,
            np.array(self.keplerianElements),
            xtol=1e-15, ftol=None, gtol=None,
            bounds=[lbounds, ubounds]
        )
        # Sometimes we seem to get stuck in a local minimum, particularly when
        # the eccentricity is close to zero.  So check if we've reached adequate
        # convergence, and try a (not-random) restart if not.
        if np.any(np.abs(opt.fun) > 1e-3):
            opt = least_squares(
                resid,
                opt.x + np.ones(6) * 1e-3,
                xtol=1e-15, ftol=None, gtol=None,
                bounds=[lbounds, ubounds]
            )
            # One final attempt
            if np.any(np.abs(opt.fun) > 1e-3):
                opt = least_squares(
                    resid,
                    opt.x + np.array([-1, 1, -1, 1, -1, 1]) * 1e-3,
                    xtol=1e-15, ftol=None, gtol=None,
                    bounds=[lbounds, ubounds]
                )
        return opt.x

    @classmethod
    def fromTLE(cls, sat_name, tle_filename, propkw=None):
        """Construct an Orbit from a TLE file

        Parameters
        ----------
        sat_name : str
            NORAD name of the satellite
        tle_filename : str
            Path and name of file where TLE is

        Returns
        -------
        Orbit
        """
        from .io import read_tle
        tle = read_tle(sat_name, tle_filename)
        return cls.fromTLETuple(tle, propkw=propkw)

    @classmethod
    def fromTLETuple(cls, tle, propkw=None):
        """Construct an Orbit from a TLE tuple

        Parameters
        ----------
        tle : 2-tuple of str
            Line1 and Line2 of TLE as strings

        Returns
        -------
        Orbit
        """
        from .io import _rvt_from_tle_tuple, parse_tle
        from sgp4.api import Satrec
        # Should set Kozai elements here directly from TLE.
        a, e, i, pa, raan, trueAnomaly, t = parse_tle(tle)
        r, v, t2 = _rvt_from_tle_tuple(tle)
        assert abs(t - t2) < 1e-5  # 10 microsec
        # Transform TEME -> GCRF
        rot = teme_to_gcrf(t)
        r = rot @ r
        v = rot @ v
        obj = Orbit(r, v, t, mu=EARTH_MU)
        obj.kozaiMeanKeplerianElements = a, e, i, pa, raan, trueAnomaly
        obj.propkw = propkw if propkw is not None else dict()
        # Path A: retain the native SGP4 record (and raw lines) so that
        # SGP4Propagator can reproduce direct-sgp4 results exactly, including
        # the drag (B*) and mean-motion-rate terms that parse_tle discards.
        # These are the *only* faithful way to propagate a TLE.
        obj._tle = tuple(tle)
        obj._sat = Satrec.twoline2rv(tle[0], tle[1])
        return obj

    def at(self, t, propagator=_KeplerianPropagator()):
        """Propagate this orbit to time t.

        Parameters
        ----------
        t : float or astropy.time.Time
            If float, then should correspond to GPS seconds; i.e., seconds since
            1980-01-06 00:00:00 UTC
        propagator : Propagator, optional
            The propagator instance to use.

        Returns
        -------
        Orbit
            Propagated Orbit.
        """
        from .compute import rv
        r, v = rv(self, t, propagator=propagator)
        return Orbit(r, v, t, mu=self.mu, propkw=self.propkw)

    def __repr__(self):
        out = "Orbit(r={!r}, v={!r}, t={!r}".format(self.r, self.v, self.t)
        if self.mu != EARTH_MU:
            out += ", mu={!r}".format(self.mu)
        out += ")"
        return out

    def _rvFromEquinoctial(self, lv=None, lE=None):
        # self is (n,) or scalar
        # assume lv/lE is always explicitly (n, m)
        # always return (n, m, 3)
        # calling code will squeeze
        # No checks done, but exactly one of lv, lE should be set.

        n = len(np.atleast_1d(self.a))
        if lv is not None:
            m = lv.shape[1]
        else:
            m = lE.shape[1]
        r = np.empty((n, m, 3))
        v = np.empty((n, m, 3))

        wBound = np.atleast_1d(self.a) > 0
        if np.any(wBound):
            a = np.atleast_1d(self.a)[wBound]
            ex = np.atleast_1d(self.ex)[wBound]
            ey = np.atleast_1d(self.ey)[wBound]
            ex2 = ex * ex
            ey2 = ey * ey
            e2 = ex2 + ey2
            exey = ex * ey
            eta = 1. + np.sqrt(1 - e2)
            beta = 1. / eta

            if lE is None:
                lEb = _ellipticalTrueToEccentricLongitudeMany(
                    lv[wBound], ex, ey
                )
            else:
                lEb = lE[wBound]

            cLe = np.cos(lEb)
            sLe = np.sin(lEb)
            exCeyS = ex[:, None] * cLe + ey[:, None] * sLe

            # avoid temporary arrays as much as possible.
            x = (1 - beta * ey2)[:, None] * cLe
            x += (beta * exey)[:, None] * sLe
            x -= ex[:, None]
            x *= a[:, None]
            y = (1 - beta * ex2)[:, None] * sLe
            y += (beta * exey)[:, None] * cLe
            y -= ey[:, None]
            y *= a[:, None]

            factor = np.sqrt(self.mu / a)[:, None] / (1. - exCeyS)
            xDot = factor * (-sLe + (beta * ey)[:, None] * exCeyS)
            yDot = factor * (cLe - (beta * ex)[:, None] * exCeyS)

            rb = np.einsum("ab,ac->abc", x, np.atleast_2d(self._pEq)[wBound])
            rb += np.einsum("ab,ac->abc", y, np.atleast_2d(self._qEq)[wBound])
            vb = np.einsum(
                "ab,ac->abc",
                xDot,
                np.atleast_2d(self._pEq)[wBound]
            )
            vb += np.einsum(
                "ab,ac->abc",
                yDot,
                np.atleast_2d(self._qEq)[wBound]
            )
            # checking if can avoid the array insert below.  The check is almost
            # free (speedwise), so usually a good idea.
            if np.all(wBound):
                return rb, vb
            r[wBound] = rb
            v[wBound] = vb

        if np.any(~wBound):
            # Copy Keplerian formula for the moment; optimize some other day...
            a = np.atleast_1d(self.a)[~wBound]
            ex = np.atleast_1d(self.ex)[~wBound]
            ey = np.atleast_1d(self.ey)[~wBound]
            e2 = ex * ex + ey * ey
            e = np.sqrt(e2)
            lonPa = np.arctan2(ey, ex)
            if lv is None:
                lvub = _hyperbolicTrueToEccentricLongitudeMany(
                    lE[~wBound], ex, ey
                )
            else:
                lvub = lv[~wBound]
            trueAnomaly = np.atleast_1d(lvub - lonPa[:, None])  # (n,m)

            sinV = np.sin(trueAnomaly)  # (n,m)
            cosV = np.cos(trueAnomaly)
            f = a * (1 - e * e)  # (n,)
            posFactor = f[:, None] / (1 + e[:, None] * cosV)  # (n,m)
            velFactor = np.sqrt(self.mu / f[:, None])  # (n,m)

            x = posFactor * cosV  # (n,m)
            y = posFactor * sinV
            xDot = -velFactor * sinV
            yDot = velFactor * (e[:, None] + cosV)

            # r = x*self._pK + y*self._qK
            # v = xDot*self._pK + yDot*self._qK

            s, c = np.sin(lonPa), np.cos(lonPa)  # (n,)
            R = np.array([[c, -s], [s, c]])  # (2, 2, n)
            rp = np.array([x, y])  # (2, n, m)
            rp = np.einsum("abc,bcd->acd", R, rp)
            rpDot = np.array([xDot, yDot])  # (2, n, m)
            rpDot = np.einsum("abc,bcd->acd", R, rpDot)  # (2, n, m)

            rub = np.einsum("nm,nt->nmt", rp[0], np.atleast_2d(self._pEq)[~wBound])
            rub += np.einsum("nm,nt->nmt", rp[1], np.atleast_2d(self._qEq)[~wBound])
            vub = np.einsum("nm,nt->nmt", rpDot[0], np.atleast_2d(self._pEq)[~wBound])
            vub += np.einsum("nm,nt->nmt", rpDot[1], np.atleast_2d(self._qEq)[~wBound])
            if np.all(~wBound):
                return rub, vub
            r[~wBound] = rub
            v[~wBound] = vub

        return r, v

    # Specialization for parallelizing over orbits
    @staticmethod
    def _rvFromEquinoctialMany(a, ex, ey, mu, pEq, qEq, lE):
        if np.any(a < 0):
            raise NotImplementedError(
                "Parallelization over hyperbolic orbits not implemented"
            )
        exey = ex * ey
        ex2 = ex * ex
        ey2 = ey * ey
        e2 = ex2 + ey2
        eta = 1. + np.sqrt(1 - e2)
        beta = 1. / eta

        cLe = np.cos(lE)
        sLe = np.sin(lE)
        exCeyS = ex[:, None] * cLe + ey[:, None] * sLe

        x = a[:, None] * (
            (1 - beta * ey2)[:, None] * cLe + (beta * exey)[:, None] * sLe - ex[:, None]
        )
        y = a[:, None] * (
            (1 - beta * ex2)[:, None] * sLe + (beta * exey)[:, None] * cLe - ey[:, None]
        )

        factor = np.sqrt(mu / a)[:, None] / (1. - exCeyS)
        xDot = factor * (-sLe + (beta * ey)[:, None] * exCeyS)
        yDot = factor * (cLe - (beta * ex)[:, None] * exCeyS)

        r = np.einsum("ab,ac->abc", x, pEq)
        r += np.einsum("ab,ac->abc", y, qEq)
        v = np.einsum("ab,ac->abc", xDot, pEq)
        v += np.einsum("ab,ac->abc", yDot, qEq)
        return r, v

    def _rvFromKeplerian(self, trueAnomaly):
        # self is (n,) or scalar
        # trueAnomaly is (n, m)
        # always return (n, m, 3); calling code will squeeze
        n = len(np.atleast_1d(self.a))
        m = trueAnomaly.shape[1]
        r = np.empty((n, m, 3))
        v = np.empty((n, m, 3))

        wBound = np.atleast_1d(self.a) > 0
        if np.any(wBound):
            a = np.atleast_1d(self.a)[wBound]
            e = np.atleast_1d(self.e)[wBound]
            uME2 = (1 - e) * (1 + e)
            s1Me2 = np.sqrt(uME2)
            E = _ellipticalTrueToEccentricAnomalyMany(
                trueAnomaly[wBound],
                e
            )
            cosE = np.cos(E)
            sinE = np.sin(E)

            # coordinates of position and velocity in the orbital plane
            x = a[:, None] * (cosE - e[:, None])
            y = a[:, None] * sinE * s1Me2[:, None]
            factor = np.sqrt(self.mu / a[:, None]) / (1 - e[:, None] * cosE)
            xDot = -sinE * factor
            yDot = cosE * s1Me2[:, None] * factor
            rb = np.einsum("ab,ac->abc", x, np.atleast_2d(self._pK)[wBound])
            rb += np.einsum("ab,ac->abc", y, np.atleast_2d(self._qK)[wBound])
            vb = np.einsum("ab,ac->abc", xDot, np.atleast_2d(self._pK)[wBound])
            vb += np.einsum("ab,ac->abc", yDot, np.atleast_2d(self._qK)[wBound])
            r[wBound] = rb
            v[wBound] = vb
        if np.any(~wBound):
            sinV = np.atleast_1d(np.sin(trueAnomaly[~wBound]))
            cosV = np.atleast_1d(np.cos(trueAnomaly[~wBound]))
            a = np.atleast_1d(self.a)[~wBound]
            e = np.atleast_1d(self.e)[~wBound]
            f = a * (1 - e * e)
            posFactor = f[:, None] / (1 + e[:, None] * cosV)
            velFactor = np.sqrt(self.mu / f[:, None])

            x = posFactor * cosV
            y = posFactor * sinV
            xDot = -velFactor * sinV
            yDot = velFactor * (e[:, None] + cosV)

            rub = np.einsum("ab,ac->abc", x, np.atleast_2d(self._pK)[~wBound])
            rub += np.einsum("ab,ac->abc", y, np.atleast_2d(self._qK)[~wBound])
            vub = np.einsum("ab,ac->abc", xDot, np.atleast_2d(self._pK)[~wBound])
            vub += np.einsum("ab,ac->abc", yDot, np.atleast_2d(self._qK)[~wBound])
            r[~wBound] = rub
            v[~wBound] = vub

        return r, v

    def _setEquinoctial(self):
        # set equinoctial elements from state vectors
        # Compute semimajor axis
        r = np.atleast_2d(self.r)
        v = np.atleast_2d(self.v)
        r2 = normSq(r)
        rnorm = np.sqrt(r2)
        v2 = normSq(v)
        rv2OverMu = rnorm * v2 / self.mu

        # negative for hyperbolic
        a = rnorm / (2 - rv2OverMu)
        muA = a * self.mu

        # Compute inclination vector
        w = np.cross(r, v)
        wnormed = normed(w)
        d = 1. / (1 + wnormed[:, 2])
        hx = -d * wnormed[:, 1]
        hy = d * wnormed[:, 0]

        # Compute true longitude argument
        cLv = (r[:, 0] - d * r[:, 2] * wnormed[:, 0]) / rnorm
        sLv = (r[:, 1] - d * r[:, 2] * wnormed[:, 1]) / rnorm
        lv = np.arctan2(sLv, cLv)

        # Compute eccentricity vector
        # Separate maths for bound/unbound orbits
        e = np.empty_like(a)
        ex = np.empty_like(a)
        ey = np.empty_like(a)
        trueAnomaly = np.empty_like(a)
        lonPa = np.empty_like(a)

        wBound = a > 0

        # bound orbits first
        eS = np.einsum("ab,ab->a", r, v) / np.sqrt(np.abs(muA))
        eC = rv2OverMu - 1
        if np.any(wBound):
            eSE = eS[wBound]
            eCE = eC[wBound]
            e2 = eCE * eCE + eSE * eSE
            e[wBound] = np.sqrt(e2)
            f = eCE - e2
            g = np.sqrt(1 - e2) * eSE
            ex[wBound] = a[wBound] * (f * cLv[wBound] + g * sLv[wBound])
            ex[wBound] /= rnorm[wBound]
            ey[wBound] = a[wBound] * (f * sLv[wBound] - g * cLv[wBound])
            ey[wBound] /= rnorm[wBound]
            lonPa[wBound] = np.arctan2(ey[wBound], ex[wBound])
            trueAnomaly[wBound] = _wrapToPi(lv[wBound] - lonPa[wBound])

        # now unbound orbits
        if np.any(~wBound):
            e[~wBound] = np.sqrt(1 - normSq(w[~wBound]) / muA[~wBound])
            eSH = eS[~wBound]
            eCH = eC[~wBound]
            H = 0.5 * np.log((eCH + eSH) / (eCH - eSH))
            trueAnomaly[~wBound] = _wrapToPi(
                _hyperbolicEccentricToTrueAnomalyMany(
                    H[:, None], e[~wBound]
                )[:, 0]
            )
            lonPa[~wBound] = _wrapToPi(lv[~wBound] - trueAnomaly[~wBound])
            ex[~wBound] = e[~wBound] * np.cos(lonPa[~wBound])
            ey[~wBound] = e[~wBound] * np.sin(lonPa[~wBound])

        # now populate properties and squeeze as necessary.
        if self.r.ndim == 1:
            a = a[0]
            e = e[0]
            hx = hx[0]
            hy = hy[0]
            ex = ex[0]
            ey = ey[0]
            lv = lv[0]
            trueAnomaly = trueAnomaly[0]
            lonPa = lonPa[0]
        self.a = a
        self.e = e
        self.hx = hx
        self.hy = hy
        self.ex = ex
        self.ey = ey
        self.lv = lv
        self.trueAnomaly = trueAnomaly
        self.lonPa = lonPa

    def _setKeplerian(self):
        # set keplerian elements from state vectors
        # Let _setEquinoctial take care of setting a, e, trueAnomaly
        self._setEquinoctial()
        r = np.atleast_2d(self.r)
        v = np.atleast_2d(self.v)
        w = normed(np.cross(r, v))
        i = unitAngle3(w, np.array([0, 0, 1.]))
        wr = np.cross(np.array([0, 0, 1.]), w)
        raan = np.arctan2(wr[:, 1], wr[:, 0])
        pa = _wrapToPi(np.atleast_1d(self.lonPa) - raan)
        if self.r.ndim == 1:
            i = i[0]
            raan = raan[0]
            pa = pa[0]
        self.i = i
        self.raan = raan
        self.pa = pa

    def _setPQEquinoctial(self):
        hx2 = self.hx * self.hx
        hy2 = self.hy * self.hy
        hxhy = self.hx * self.hy
        factH = 1. / (1 + hx2 + hy2)

        # Reference axes defining orbital plane
        self._pEq = (np.array([1 + hx2 - hy2, 2 * hxhy, -2 * self.hy]) * factH).T
        self._qEq = (np.array([2 * hxhy, 1 - hx2 + hy2, 2 * self.hx]) * factH).T

    def _setPQKeplerian(self):
        cosRaan = np.cos(self.raan)
        sinRaan = np.sin(self.raan)
        cosPa = np.cos(self.pa)
        sinPa = np.sin(self.pa)
        cosI = np.cos(self.i)
        sinI = np.sin(self.i)
        crcp = cosRaan * cosPa
        crsp = cosRaan * sinPa
        srcp = sinRaan * cosPa
        srsp = sinRaan * sinPa
        self._pK = np.array([
            crcp - cosI * srsp, srcp + cosI * crsp, sinI * sinPa
        ]).T
        self._qK = np.array([
            -crsp - cosI * srcp, -srsp + cosI * crcp, sinI * cosPa
        ]).T

    @_lazy_property
    def _pEq(self):
        self._setPQEquinoctial()
        return self._pEq

    @_lazy_property
    def _qEq(self):
        self._setPQEquinoctial()
        return self._qEq

    @_lazy_property
    def _pK(self):
        self._setPQKeplerian()
        return self._pK

    @_lazy_property
    def _qK(self):
        self._setPQKeplerian()
        return self._qK

    @_lazy_property
    def a(self):
        """Semimajor axis in meters.
        """
        self._setEquinoctial()
        return self.a

    @_lazy_property
    def hx(self):
        """First component of equinoctial inclination vector.
        """
        self._setEquinoctial()
        return self.hx

    @_lazy_property
    def hy(self):
        """Second component of equinoctial inclination vector.
        """
        self._setEquinoctial()
        return self.hy

    @_lazy_property
    def ex(self):
        """First component of equinoctial eccentricity vector.
        """
        self._setEquinoctial()
        return self.ex

    @_lazy_property
    def ey(self):
        """Second component of equinoctial eccentricity vector.
        """
        self._setEquinoctial()
        return self.ey

    @_lazy_property
    def lv(self):
        """True longitude in radians.
        """
        self._setEquinoctial()
        return self.lv

    @_lazy_property
    def lE(self):
        """Eccentric longitude in radians.
        """
        import numbers
        if isinstance(self.a, numbers.Real):
            if self.a > 0:
                return _ellipticalTrueToEccentricLongitude(
                    self.lv, self.ex, self.ey
                )
            else:
                return _hyperbolicTrueToEccentricLongitude(
                    self.lv, self.ex, self.ey
                )
        else:
            wBound = self.a > 0
            lE = np.empty_like(self.lv)
            if np.any(wBound):
                lE[wBound] = _ellipticalTrueToEccentricLongitude(
                    self.lv[wBound], self.ex[wBound], self.ey[wBound]
                )
            if np.any(~wBound):
                lE[~wBound] = _hyperbolicTrueToEccentricLongitude(
                    self.lv[~wBound], self.ex[~wBound], self.ey[~wBound]
                )
            return lE

    @_lazy_property
    def lM(self):
        """Mean longitude in radians.
        """
        import numbers
        if isinstance(self.a, numbers.Real):
            if self.a > 0:
                return _ellipticalEccentricToMeanLongitude(
                    self.lE, self.ex, self.ey
                )
            else:
                return _hyperbolicEccentricToMeanLongitude(
                    self.lE, self.ex, self.ey
                )
        else:
            wBound = self.a > 0
            lM = np.empty_like(self.lv)
            if np.any(wBound):
                lM[wBound] = _ellipticalEccentricToMeanLongitude(
                    self.lE[wBound], self.ex[wBound], self.ey[wBound]
                )
            if np.any(~wBound):
                lM[~wBound] = _hyperbolicEccentricToMeanLongitude(
                    self.lE[~wBound], self.ex[~wBound], self.ey[~wBound]
                )
            return lM

    @_lazy_property
    def e(self):
        """Eccentricity.
        """
        return np.sqrt(self.ex * self.ex + self.ey * self.ey)

    @_lazy_property
    def i(self):
        """Inclination in radians.
        """
        self._setKeplerian()
        return self.i

    @_lazy_property
    def pa(self):
        """Periapsis argument in radians.
        """
        self._setKeplerian()
        return self.pa

    @_lazy_property
    def lonPa(self):
        """Longitude of periapsis argument in radians.
        """
        self._setEquinoctial()
        return self.lonPa

    @_lazy_property
    def raan(self):
        """Right ascension of the ascending node in radians.
        """
        self._setKeplerian()
        return self.raan

    @_lazy_property
    def trueAnomaly(self):
        """True anomaly in radians.
        """
        self._setEquinoctial()
        return self.trueAnomaly

    @_lazy_property
    def eccentricAnomaly(self):
        """Eccentric anomaly in radians.
        """
        import numbers
        if isinstance(self.a, numbers.Real):
            if self.a > 0:
                return _ellipticalTrueToEccentricAnomaly(
                    self.trueAnomaly, self.e
                )
            else:
                return _hyperbolicTrueToEccentricAnomaly(
                    self.trueAnomaly, self.e
                )
        else:
            wBound = self.a > 0
            eccentricAnomaly = np.empty_like(self.trueAnomaly)
            if np.any(wBound):
                eccentricAnomaly[wBound] = _ellipticalTrueToEccentricAnomaly(
                    self.trueAnomaly[wBound], self.e[wBound]
                )
            if np.any(~wBound):
                eccentricAnomaly[~wBound] = _hyperbolicTrueToEccentricAnomaly(
                    self.trueAnomaly[~wBound], self.e[~wBound]
                )
            return eccentricAnomaly

    @_lazy_property
    def meanAnomaly(self):
        """Mean anomaly in radians.
        """
        import numbers
        if isinstance(self.a, numbers.Real):
            if self.a > 0:
                return _ellipticalEccentricToMeanAnomaly(
                    _ellipticalTrueToEccentricAnomaly(self.trueAnomaly, self.e),
                    self.e
                )
            else:
                return _hyperbolicEccentricToMeanAnomaly(
                    _hyperbolicTrueToEccentricAnomaly(self.trueAnomaly, self.e),
                    self.e
                )
        else:
            wBound = self.a > 0
            meanAnomaly = np.empty_like(self.trueAnomaly)
            if np.any(wBound):
                meanAnomaly[wBound] = _ellipticalEccentricToMeanAnomaly(
                    _ellipticalTrueToEccentricAnomaly(
                        self.trueAnomaly[wBound], self.e[wBound]
                    ),
                    self.e[wBound]
                )
            if np.any(~wBound):
                meanAnomaly[~wBound] = _hyperbolicEccentricToMeanAnomaly(
                    _hyperbolicTrueToEccentricAnomaly(
                        self.trueAnomaly[~wBound], self.e[~wBound]
                    ),
                    self.e[~wBound]
                )
            return meanAnomaly

    @_lazy_property
    def period(self):
        """Orbital period in seconds.
        """
        import numbers
        if isinstance(self.a, numbers.Real):
            return 2 * np.pi / self.meanMotion if self.a > 0 else np.inf
        else:
            wBound = self.a > 0
            period = np.empty_like(self.a)
            if np.any(wBound):
                period[wBound] = 2 * np.pi / self.meanMotion[wBound]
            if np.any(~wBound):
                period[~wBound] = np.inf
            return period

    @_lazy_property
    def meanMotion(self):
        """Mean motion in radians per second.
        """
        return np.sqrt(self.mu / np.abs(self.a**3))

    @property
    def equinoctialElements(self):
        """Equinoctial elements (a, hx, hy, ex, ey, lv).
        """
        return self.a, self.hx, self.hy, self.ex, self.ey, self.lv

    @property
    def keplerianElements(self):
        """Keplerian elements (a, e, i, pa, raan, trueAnomaly).
        """
        return self.a, self.e, self.i, self.pa, self.raan, self.trueAnomaly

    @_lazy_property
    def p(self):
        """Semi-latus rectum in meters.
        """
        return self.a * (1.0 - self.e**2)

    @_lazy_property
    def angularMomentum(self):
        """(Specific) angular momentum vector in m^2/s.
        """
        return np.cross(self.r, self.v)

    @_lazy_property
    def energy(self):
        """(Specific) orbital energy in m^2/s^2.
        """
        return -0.5 * self.mu / self.a

    @_lazy_property
    def LRL(self):
        """Laplace-Runge-Lenz vector in m^3/s^2.
        """
        return -self.mu * normed(self.r) - np.cross(self.angularMomentum, self.v)

    @_lazy_property
    def periapsis(self):
        """Periapsis coordinate of orbit in meters.
        """
        import numbers
        pdist = self.p / (1 + self.e)
        if isinstance(self.a, numbers.Real):
            return normed(self.LRL) * pdist
        else:
            return normed(self.LRL) * pdist[:, None]

    @_lazy_property
    def apoapsis(self):
        """Apoapsis coordinate of orbit in meters.
        """
        import numbers
        if isinstance(self.a, numbers.Real):
            if self.a < 0:
                return np.array([np.inf, np.inf, np.inf])
            adist = self.p / (1 - self.e)
            return -normed(self.LRL) * adist
        else:
            wBound = self.a > 0
            apoapsis = np.empty((len(self.a), 3))
            if np.any(wBound):
                adist = self.p[wBound] / (1 - self.e[wBound])
                apoapsis[wBound] = -normed(self.LRL)[wBound] * adist[:, None]
            if np.any(~wBound):
                apoapsis[~wBound] = np.array([np.inf, np.inf, np.inf])
            return apoapsis

    @_lazy_property
    def tle(self):
        """Two line element for this Orbit.
        """
        from .io import make_tle
        return make_tle(*self.kozaiMeanKeplerianElements, self.t)


class EarthObserver:
    """ An earth-bound observer.

    Parameters
    ----------
    lon : float
        Geodetic longitude in degrees (increasing to the East).
    lat : float
        Geodetic latitude in degrees.
    elevation : float, optional
        Elevation in meters.
    fast : bool, optional
        Use fast lookup tables for Earth Orientation parameters.  ~ meter and
        ~10 micron/sec accurate for dates between approximately 1973 and the
        present.  Less accurate outside this range.

    Attributes
    ----------
    lon
    lat
    elevation
    itrs : array_like (3,)
        ITRS coordinates in meters.

    Methods
    -------
    getRV(time)
        Get position and velocity at specified time(s) in the GCRF frame.
    """
    def __init__(self, lon, lat, elevation=0, fast=False):
        from astropy.coordinates import EarthLocation
        import astropy.units as u

        self.lon = lon
        self.lat = lat
        self.elevation = elevation
        self._location = EarthLocation(
            self.lon * u.deg, self.lat * u.deg, self.elevation * u.m
        )
        self.itrs = self._location.itrs.cartesian.xyz.to(u.m).value
        self.fast = fast

    def __repr__(self):
        out = "EarthObserver(lon={!r}, lat={!r}".format(self.lon, self.lat)
        if self.elevation != 0:
            out += ", elevation={!r}".format(self.elevation)
        out += ")"
        return out

    _dGHAdt = (
        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], dtype=float) * 1.002737909350795 * 2 * np.pi / 86400
    )

    def getRV(self, time):
        """ Get position and velocity at specified time(s).

        Parameters
        ----------
        time : array_like or astropy.time.Time (n,)
            If float (array), then should correspond to GPS seconds;
            i.e., seconds since 1980-01-06 00:00:00 UTC

        Returns
        -------
        r : array_like (n, 3)
            Position in meters.
        v : array_like (n, 3)
            Velocity in meters per second.

        Notes
        -----
        The returned position and velocity are in the GCRF frame.
        """
        import astropy.units as u

        if self.fast:
            import numbers
            if isinstance(time, astropy.time.Time):
                time = time.gps
            if isinstance(time, numbers.Real):
                scalar = True
                time = np.array([time])
            else:
                scalar = False
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
            outR = np.transpose(polar @ U, (0, 2, 1)) @ self.itrs
            outV = np.transpose(polar @ self._dGHAdt @ U, (0, 2, 1)) @ self.itrs

            if scalar:
                return outR[0], outV[0]
            return outR, outV
        else:
            if not isinstance(time, astropy.time.Time):
                time = astropy.time.Time(time, format='gps')
            r, v = self._location.get_gcrs_posvel(time)
            return r.xyz.to(u.m).T.value, v.xyz.to(u.m / u.s).T.value

    def sunAlt(self, time):
        """Get Sun altitude for observer at `time`.

        Parameters
        ----------
        time : array_like or astropy.time.Time (n,)
            If float (array), then should correspond to GPS seconds;
            i.e., seconds since 1980-01-06 00:00:00 UTC

        Returns
        -------
        alt : array_like(n,)
            Altitude of sun in radians.
        """
        if isinstance(time, astropy.time.Time):
            time = time.gps
        ro, _ = self.getRV(time)
        r_sun = sunPos(time, fast=False)
        dr = r_sun - ro
        return np.pi / 2 - unitAngle3(normed(ro), normed(dr))


class OrbitalObserver:
    """ An observer in orbit.

    Parameters
    ----------
    orbit : Orbit
        The orbit of the observer.
    propagator : Propagator, optional
        The propagator instance to use.

    Attributes
    ----------
    orbit
    propagator

    Methods
    -------
    getRV(time)
        Get position and velocity at specified time(s).
    """
    def __init__(self, orbit, propagator=_KeplerianPropagator()):
        self.orbit = orbit
        self.propagator = propagator

    def __repr__(self):
        out = "OrbitalObserver({!r}".format(self.orbit)
        if self.propagator != _KeplerianPropagator():
            out += ", propagator={!r}".format(self.propagator)
        out += ")"
        return out

    def getRV(self, time):
        """ Get position and velocity at specified time(s).

        Parameters
        ----------
        time : array_like or astropy.time.Time (n,)
            If float (array), then should correspond to GPS seconds;
            i.e., seconds since 1980-01-06 00:00:00 UTC

        Returns
        -------
        r : array_like (n, 3)
            Position in meters.
        v : array_like (n, 3)
            Velocity in meters per second.
        """
        from .compute import rv
        return rv(self.orbit, time, propagator=self.propagator)
