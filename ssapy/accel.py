"""
Classes for modeling accelerations.
"""

import numpy as np

from .constants import EARTH_MU, EARTH_RADIUS
from .utils import norm, sunPos, _gpsToTT, ntw_to_r
from .ellipsoid import Ellipsoid as _Ellipsoid

try:
    import erfa
except ImportError:
    # Let this raise
    import astropy._erfa as erfa


# Some utility functions
from functools import lru_cache


@lru_cache(maxsize=None)
def _fac(n):
    n = int(n)
    from math import factorial
    return factorial(n)


@lru_cache(maxsize=None)
def _invnorm(n, m):
    n = int(n)
    m = int(m)
    num = (1 if m == 0 else 2) * (2 * n + 1) * _fac(n - m)
    den = _fac(n + m)
    return np.sqrt(num / den)


class Accel:
    """Base class for accelerations."""
    def __init__(self, time_breakpoints=None):
        if time_breakpoints is None:
            self.time_breakpoints = [-np.inf, np.inf]
        else:
            self.time_breakpoints = time_breakpoints

    def __add__(self, rhs):
        return AccelSum((self, rhs))

    def __mul__(self, rhs):
        return AccelProd(self, rhs)

    def __sub__(self, rhs):
        return AccelSum([self, AccelProd(rhs, -1.0)])


class AccelSum(Accel):
    """Acceleration defined as the sum of other accelerations.

    Note, besides directly invoking this class, you can simply use the `+`
    operator on any two `Accel` subclasses too.

    Parameters
    ----------
    accels : list of Accel
        Accelerations to add
    """
    def __init__(self, accels):
        super().__init__()
        self.accels = tuple(accels)
        self.time_breakpoints = np.unique(
            np.concatenate([a.time_breakpoints for a in accels]))

    def __call__(self, r, v, t, **kwargs):
        """Evaluate acceleration at particular place/moment.

        Parameters
        ----------
        r : array_like, shape(3, )
            Position in meters.
        v : array_like, shape(3, )
            Velocity in meters per second.
        t : float
            Time as GPS seconds
        **kwargs : dict
            Other arguments to pass on to individual terms of sum.

        Returns
        -------
        accel : array_like, shape(3,)
            Acceleration in meters per second^2
        """
        out = self.accels[0](r, v, t, **kwargs)
        for f in self.accels[1:]:
            out += f(r, v, t, **kwargs)
        return out

    def __hash__(self):
        return hash(("AccelSum", self.accels))

    def __eq__(self, rhs):
        if not isinstance(rhs, AccelSum):
            return False
        return (self.accels == rhs.accels)


class AccelProd(Accel):
    """Acceleration defined as the product of an acceleration with a constant factor."""
    def __init__(self, accel, factor):
        super().__init__()
        self.accel = accel
        self.factor = factor
        self.time_breakpoints = accel.time_breakpoints

    def __call__(self, r, v, t, **kwargs):
        out = self.accel(r, v, t, **kwargs)
        out *= self.factor
        return out

    def __hash__(self):
        return hash(("AccelProd", self.accel, self.factor))

    def __eq__(self, rhs):
        if not isinstance(rhs, AccelProd):
            return False
        return (
            self.accel == rhs.accel and self.factor == rhs.factor
        )


class AccelKepler(Accel):
    """Keplerian acceleration.  I.e., force is proportional to ``1 / |r|^2``.

    Parameters
    ----------
    mu : float, optional
        Gravitational constant of central body in m^3/s^2.  (Default: Earth's
        gravitational constant in WGS84).
    """
    def __init__(self, mu=EARTH_MU):
        super().__init__()
        self.mu = mu

    def __call__(self, r, v, t, **kwargs):
        """Evaluate acceleration at particular place/moment.

        Parameters
        ----------
        r : array_like, shape(3, )
            Position in meters.
        v : array_like, shape(3, )
            Velocity in meters per second.  Unused.
        t : float
            Time as GPS seconds.  Unused

        Returns
        -------
        accel : array_like, shape(3,)
            Acceleration in meters per second^2
        """
        return -self.mu * r / norm(r)**3

    def __hash__(self):
        return hash(("AccelKepler", self.mu))

    def __eq__(self, rhs):
        if not isinstance(rhs, AccelKepler):
            return False
        return (self.mu == rhs.mu)


class AccelSolRad(Accel):
    """Acceleration due to solar radiation pressure.

    This is a relatively simple model in which the direction of the acceleration
    is directly away from the sun and the magnitude is modulated by a single
    solar radiation pressure coefficient `CR`.  The coefficient is 1.0 for
    purely absorbed light, and 2.0 for purely reflected light.  Rough typical
    values for a variety of different satellite components are

        ~ 1.2  for solar panels
        ~ 1.3  for a high gain antenna
        ~ 1.9  for a aluminum coated mylar solar sail.

    Additionally, this class models the Earth's shadow as a cylinder.  Either
    the satellite is inside the cylinder, in which case the acceleration from
    this term is 0, or the satellite is outside the cylinder, in which case the
    full magnitude of the acceleration is computed.

    More details can be found in Section 3.4 of Montenbruck and Gill.

    Parameters
    ----------
    defaultkw : dict
        default parameters for kwargs passed to __call__,
        (area, mass, CR)
    """
    def __init__(self, **defaultkw):
        super().__init__()
        self.defaultkw = defaultkw

    def __call__(self, r, v, t, **kwargs):
        """Evaluate acceleration at particular place/moment.

        Parameters
        ----------
        r : array_like, shape(3, )
            Position in meters in GCRF frame.
        v : array_like, shape(3, )
            Velocity in meters per second.  Unused.
        t : float
            Time as GPS seconds
        area : float
            Area in meters^2.
        mass : flat
            Mass in kg.
        CR : float
            Radiation pressure coefficient.

        Returns
        -------
        accel : array_like, shape(3,)
            Acceleration in meters per second^2
        """
        kw = dict()
        kw.update(self.defaultkw)
        kw.update(kwargs)
        rr = r - sunPos(t)
        P0 = 4.56e-6  # Solar rad pressure [N/m^2]   MG eqn (3.69)
        AU2 = 2.2379522708536898e22  # 1 AU squared [m^2]
        # MG (3.75)
        return P0 * kw['CR'] * kw['area'] / kw['mass'] * rr / norm(rr)**3 * AU2

    def __hash__(self):
        return hash((
            "AccelSolRad",
            frozenset(self.defaultkw.items())
        ))

    def __eq__(self, rhs):
        if not isinstance(rhs, AccelSolRad):
            return False
        return self.defaultkw == rhs.defaultkw


class AccelEarthRad(Accel):
    """Acceleration due to Earth radiation pressure.

    This is a very simple model in which the direction of the acceleration
    is directly away from the Earth and the magnitude is modulated by a single
    solar radiation pressure coefficient `CR`.  The coefficient is 1.0 for
    purely absorbed light, and 2.0 for purely reflected light.  Rough typical
    values for a variety of different satellite components are

        ~ 1.2  for solar panels
        ~ 1.3  for a high gain antenna
        ~ 1.9  for a aluminum coated mylar solar sail.

    The radiation pressure at the Earth's surface is given as
    (230 + 459*k) W/m^2, where 230 is from the thermal radiation of the
    earth, and 459 is the reflected sunlight.  k is the illuminated
    fraction of the earth as seen from the satellite,
    assuming the earth is point-like (i.e., neglecting that the
    satellite will see less than a full hemisphere for LEO objects).  The
    radiation pressure goes down like 1/r^2 as an object moves away from the
    earth.

    This is a simplification of the more complex model presented in MG 3.7.1,
    neglecting spatial variation in the emitted light and the different
    angles to different parts of the earth.

    Parameters
    ----------
    defaultkw : dict
        default parameters for kwargs passed to __call__,
        (area, mass, CR)
    """
    def __init__(self, **defaultkw):
        super().__init__()
        self.defaultkw = defaultkw

    def __call__(self, r, v, t, **kwargs):
        """Evaluate acceleration at particular place/moment.

        Parameters
        ----------
        r : array_like, shape(3, )
            Position in meters in GCRF frame.
        v : array_like, shape(3, )
            Velocity in meters per second.  Unused.
        t : float
            Time as GPS seconds
        area : float
            Area in meters^2.
        mass : flat
            Mass in kg.
        CR : float
            Radiation pressure coefficient.

        Returns
        -------
        accel : array_like, shape(3,)
            Acceleration in meters per second^2
        """
        kw = dict()
        kw.update(self.defaultkw)
        kw.update(kwargs)
        r_sun = sunPos(t)
        d_sun = norm(r_sun)
        r = np.asarray(r)
        normr = norm(r)
        if normr < 1:
            r = np.array([1, 0, 0])
            normr = 1
        if normr < EARTH_RADIUS:
            r = r * EARTH_RADIUS / normr
            normr = EARTH_RADIUS
        normsunsat = norm(r_sun - r)
        cosi = ((d_sun**2 + normr**2 - normsunsat**2) / (2 * d_sun * normr))
        k = (1 + cosi) / 2
        # Astronomical Algorithms, Chap 41, Jean Meeus.
        pressure = (459 * k + 230) / 299792458 * r * EARTH_RADIUS**2 / normr**3
        accel = pressure * kw['CR'] * kw['area'] / kw['mass']
        return accel

    def __hash__(self):
        return hash((
            "AccelEarthRad",
            frozenset(self.defaultkw.items())
        ))

    def __eq__(self, rhs):
        if not isinstance(rhs, AccelEarthRad):
            return False
        return self.defaultkw == rhs.defaultkw


class AccelDrag(Accel):
    """Acceleration due to atmospheric drag.

    This class uses the Harris-Priester density model, which includes diurnal
    variation in the atmospheric bulge, but omits longer period seasonal
    variations.

    The acceleration also depends on a drag coefficient, which is hard to
    determine a priori, but takes on typical values around ~2 to ~2.3 for most
    satellites.

    See Section 3.5 of Montenbruck and Gill for more details.

    Parameters
    ----------
    recalc_threshold : float, optional
        Number of seconds past which the code will recompute the
        precession/nutation matrix.  Default: 86400*30  (30 days)
    defaultkw : dict
        default parameters for kwargs passed to __call__,
        (area, mass, CR)
    """
    def __init__(self, recalc_threshold=86400 * 30, **defaultkw):
        from . import _ssapy

        self.recalc_threshold = recalc_threshold
        self._t = None
        super().__init__()
        self.defaultkw = defaultkw
        ellip = _Ellipsoid()
        self.atm = _ssapy.HarrisPriester(ellip, n=6.0)

    def __call__(self, r, v, t, _T=None, **kwargs):
        """Evaluate acceleration at particular place/moment.

        Parameters
        ----------
        r : array_like, shape(3, )
            Position in meters in GCRF frame.
        v : array_like, shape(3, )
            Velocity in meters per second in GCRF frame.
        t : float
            Time as GPS seconds
        area : float
            Area in meters^2.
        mass : flat
            Mass in kg.
        CD : float
            Drag coefficient.  Typical values are ~ 2 - 2.3

        Returns
        -------
        accel : array_like, shape(3,)
            Acceleration in meters per second^2
        """
        kw = dict()
        kw.update(self.defaultkw)
        kw.update(kwargs)
        mjd_tt = _gpsToTT(t)
        if _T is None:
            if self._t is None or np.abs(t - self._t) > self.recalc_threshold:
                self._t = t
                self._T = erfa.pnm80(2400000.5, mjd_tt)
            _T = self._T
        r_sun = sunPos(t)
        r_tod = _T @ r
        v_tod = _T @ v

        v_rel = v_tod - np.cross([0, 0, 7.2921159e-5], r_tod)  # MG (3.98)
        ra_sun = np.arctan2(r_sun[1], r_sun[0])
        dec_sun = np.arctan(r_sun[2] / np.hypot(r_sun[0], r_sun[1]))
        density = self.atm.density(
            *r_tod,
            ra_sun,
            dec_sun
        )
        if not np.isfinite(density):
            print(f"r_tod = {r_tod}")
            print(f"ra_sun = {ra_sun}")
            print(f"dec_sun = {dec_sun}")
            raise ValueError("non finite density")
        a_tod = -0.5 * kw['CD'] * kw['area'] / kw['mass'] * density * v_rel * norm(v_rel)
        return _T.T @ a_tod

    def __hash__(self):
        return hash((
            "AccelDrag",
            frozenset(self.defaultkw.items())
        ))

    def __eq__(self, rhs):
        if not isinstance(rhs, AccelDrag):
            return False
        return self.defaultkw == rhs.defaultkw


class AccelConstNTW(Accel):
    """Constant acceleration in NTW coordinates.

    Intended to enable maneuvers.  Semimajor axis changes are often done by
    accelerating in the in-track direction at perigee, while inclination
    change maneuvers are done by accelerating in the cross-track direction.
    So these maneuvers are both conveniently implemented in NTW space.

    This class supports a constant acceleration burn in a fixed direction in
    NTW space at a specified time and duration.

    Parameters
    ----------
    accelntw : array_like, shape(3,)
        Direction and magnitude of acceleration in NTW frame, m/s^2.
    time_breakpoints : array_like
        Times in GPS seconds when acceleration should be turned on and off.
        These alternate; first time is on, second time is off, third time is
        back on.  must be sorted.
    """
    def __init__(self, accelntw, time_breakpoints=None):
        if time_breakpoints is None:
            time_breakpoints = [-np.inf, np.inf]
        if np.any(np.diff(time_breakpoints) < 0):
            raise ValueError('acceleration times must be sorted!')
        super().__init__(np.array(time_breakpoints))
        self.accelntw = np.array(accelntw)

    def __call__(self, r, v, t, **kwargs):
        ind = np.searchsorted(self.time_breakpoints, t)
        off = (ind % 2) == 0
        if off:
            return np.zeros(3)
        else:
            return ntw_to_r(r, v, self.accelntw, relative=True)

    def __hash__(self):
        return hash((
            "AccelConstantNTW",
            tuple(self.time_breakpoints),
            tuple(self.accelntw)
        ))

    def __eq__(self, rhs):
        if not isinstance(rhs, AccelConstNTW):
            return False
        return (np.all(self.accelntw == rhs.accelntw) and np.all(self.time_breakpoints == rhs.time_breakpoints))
