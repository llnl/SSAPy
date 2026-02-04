"""
Classes for propagating orbits.
"""

from abc import ABC, abstractmethod
import numpy as np
from astropy.time import Time

from .utils import LRU_Cache, norm, teme_to_gcrf
from .constants import EARTH_RADIUS

# Set up a cache for propagation interpolants.  Useful in case one wants to
# propagate using the same Propagator and Orbit multiple times, such as when
# optimizing a position along an orbit to find a rise-time, set-time,
# highest-elevation, etc., or when computing topocentric quantities for one
# orbit and many observers.

# The cache will be a map (orbit, propagator) -> interpolant.  One slight
# wrinkle is that we need to be able to extend the domain of a previously cached
# interpolant.  An easy way to implement this is to have the return object be a
# mutable container (such as a python list) instead of the raw interpolant.  We
# can then mutate the interpolant without invalidating the cached container.


def _getInterpolantContainer(orbit, propagator):
    # The default contained interpolant is empty.  Calling code should mutate
    # the container to hold a useful interpolant.  Because the list is mutated
    # in-place, it remains in the cache in the same position.
    return []


_InterpolantCache = LRU_Cache(_getInterpolantContainer)


class Propagator(ABC):
    """ Abstract base class for orbit propagators.

    Note, the interface for propagators is through functions in ssapy.compute.
    """
    # Essentially no public interface.
    # Interface is instead through compute.rv

    @abstractmethod
    def _getRVOne(self, orbit, time):
        ...  # Subclasses must override

    # def _getRVMany(self, orbit, time):
    #     # Computes positions, velocities broadcasting both over orbits and
    #     # times.  Base class version is just a for-loop, but subclasses may
    #     # override if they can be more efficient (e.g., KeplerianPropagator).
    #     nOrbit = len(orbit)
    #     nTime = len(time)
    #     outR = np.empty((nOrbit, nTime, 3,), dtype=float)
    #     outV = np.empty_like(outR)
    #     for j, orb in enumerate(orbit):
    #         outR[j], outV[j] = self._getRVOne(orb, time)
    #     return outR, outV
    def _getRVMany(self, orbit, time):
        nOrbit = len(orbit)
        outR_list = []
        outV_list = []
        min_len = len(time)

        for j, orb in enumerate(orbit):
            rj, vj = self._getRVOne(orb, time)
            if len(rj) < min_len:
                min_len = len(rj)
            outR_list.append(rj)
            outV_list.append(vj)

        # Truncate all results and time array to minimum length returned
        outR = np.array([r[:min_len] for r in outR_list])
        outV = np.array([v[:min_len] for v in outV_list])

        return outR, outV

class KeplerianPropagator(Propagator):
    """ A basic Keplerian propagator for finding the position and velocity of an
    orbiting object at some future or past time.
    """
    def __repr__(self):
        return "KeplerianPropagator()"

    def _getRVOne(self, orbit, time):
        from .orbit import _ellipticalMeanToEccentricLongitude
        from .orbit import _hyperbolicMeanToEccentricLongitude
        dts = np.atleast_1d(time - orbit.t)
        lM = orbit.lM + orbit.meanMotion * dts
        if orbit.a > 0:
            lE = _ellipticalMeanToEccentricLongitude(lM, orbit.ex, orbit.ey)
        else:
            lE = _hyperbolicMeanToEccentricLongitude(lM, orbit.ex, orbit.ey)
        r, v = orbit._rvFromEquinoctial(lE=np.atleast_2d(lE))
        return r[0], v[0]

    def _getRVMany(self, orbit, time):
        from .orbit import _ellipticalMeanToEccentricLongitudeMany
        from .orbit import _hyperbolicMeanToEccentricLongitudeMany
        # expect a Vector orbit.
        dts = np.atleast_2d(time[None, :] - orbit.t[:, None])  # (nOrbit, nTime)
        lM = orbit.lM[:, None] + orbit.meanMotion[:, None] * dts
        wBound = np.atleast_1d(orbit.a) > 0
        lE = np.empty_like(lM)
        if np.any(wBound):
            ex, ey = orbit.ex[wBound], orbit.ey[wBound]
            lE[wBound] = _ellipticalMeanToEccentricLongitudeMany(
                lM[wBound], ex, ey
            )
        if np.any(~wBound):
            ex, ey = orbit.ex[~wBound], orbit.ey[~wBound]
            lE[~wBound] = _hyperbolicMeanToEccentricLongitudeMany(
                lM[~wBound], ex, ey
            )
        r, v = orbit._rvFromEquinoctial(lE=lE)
        return r, v

    def __hash__(self):
        return hash("KeplerianPropagator")

    def __eq__(self, rhs):
        return isinstance(rhs, KeplerianPropagator)


class SeriesPropagator(Propagator):
    """ Propagate using Taylor series expansion of Keplerian motion.  This is
    quite a bit faster than the full Keplerian propagator, but only valid for
    short time intervals.  Using order=2 for a few seconds of time though is
    definitely reasonable.

    Parameters
    ----------
    order : int, optional
        Order of series expansion.  1 = constant velocity, 2 = constant
        acceleration
    """
    def __init__(self, order=2):
        self.order = order

    def __repr__(self):
        if self.order == 2:
            return "SeriesPropagator()"
        return "SeriesPropagator({})".format(self.order)

    def _getRVOne(self, orbit, time):
        dts = np.atleast_1d(time - orbit.t)
        if self.order == 0:
            # Just constant position.  Weird, but allowed.
            rs = np.broadcast_to(orbit.r, (len(time), 3))
            # Or should I use 0 here?
            vs = np.broadcast_to(orbit.v, (len(time), 3))
            return rs, vs
        if self.order == 1:
            # Constant velocity
            rs = orbit.r + orbit.v * dts[:, None]
            vs = np.broadcast_to(orbit.v, (len(time), 3))
            return rs, vs
        if self.order == 2:
            # Constant acceleration
            accel = -orbit.mu * orbit.r / norm(orbit.r)**3
            rs = orbit.r + dts[:, None] * (orbit.v + 0.5 * dts[:, None] * accel)
            vs = orbit.v + dts[:, None] * accel
            return rs, vs
        if self.order == 3:
            accel = -orbit.mu * orbit.r / norm(orbit.r)**3
            jerk = -orbit.mu * (
                orbit.r * (-3 * np.dot(orbit.r, orbit.v) / norm(orbit.r)**5) + orbit.v / norm(orbit.r)**3
            )
            rs = orbit.r + dts[:, None] * (
                orbit.v + 0.5 * dts[:, None] * (
                    accel + dts[:, None] * jerk / 3.
                )
            )
            vs = orbit.v + dts[:, None] * (accel + 0.5 * dts[:, None] * jerk)
            return rs, vs
        else:
            # I think this is basically the f and g series idea in Escobal
            # (1965) section 3.9.1
            raise NotImplementedError("order>3 not yet implemented")

    def __hash__(self):
        return hash(("SeriesPropagator", self.order))

    def __eq__(self, rhs):
        if not isinstance(rhs, SeriesPropagator):
            return False
        return self.order == rhs.order


class SGP4Propagator(Propagator):
    """Propagate using simplified perturbation model SGP4.

    Parameters
    ----------
    t : float or astropy.time.Time, optional
        Reference time at which to compute frame transformation between GCRF
        and TEME.  SGP4 calculations occur in the TEME frame, but useful input
        and output is in the GCRF frame.  In principle, one could do the
        transformation at every instant in time for which the orbit is queried.
        However, the rate of change in the transformation is small, ~0.15 arcsec
        per day, so here we just use a single transformation.

        If float, then should correspond to GPS seconds;
        i.e., seconds since 1980-01-06 00:00:00 UTC

        If None, then use the time of the orbit being propagated.
    truncate : bool, optional
        Truncate elements to precision of TLE ASCII format?  This may be
        required in order to reproduce the results of running sgp4 directly from
        a TLE.
    """
    def __init__(self, t=None, truncate=False):
        if isinstance(t, Time):
            t = t.gps
        self.t = t
        self.truncate = truncate

    def __repr__(self):
        return "SGP4Propagator()"

    def _getRVOne(self, orbit, time):
        from sgp4.api import Satrec, WGS72
        from .orbit import _ellipticalEccentricToMeanAnomaly, _ellipticalTrueToEccentricAnomaly
        from .constants import WGS72_EARTH_MU
        from .io import make_tle

        if self.truncate:
            line1, line2 = make_tle(*orbit.kozaiMeanKeplerianElements, orbit.t)
            sat = Satrec.twoline2rv(line1, line2)
        else:
            a, e, i, pa, raan, trueAnomaly = orbit.kozaiMeanKeplerianElements
            meanAnomaly = _ellipticalEccentricToMeanAnomaly(
                _ellipticalTrueToEccentricAnomaly(
                    trueAnomaly % (2 * np.pi), e
                ),
                e
            )
            meanMotion = np.sqrt(WGS72_EARTH_MU / np.abs(a**3)) * 60.0  # rad/m
            tt = Time(orbit.t, format='gps').utc
            epoch = tt.mjd - 33281.0
            sat = Satrec()
            sat.sgp4init(
                WGS72,        # gravity model
                'i',          # 'a' = old AFSPC mode, 'i' = improved mode
                0,            # satnum: Satellite number
                epoch,        # epoch: days since 1949 December 31 00:00 UT
                0.0,          # bstar: drag coefficient (kg/m2er)
                0.0,          # ndot: ballistic coefficient (revs/day)
                0.0,          # nddot: second derivative of mean motion (revs/day^3)
                e,            # ecco: eccentricity
                pa,           # argpo: argument of perigee (radians)
                i,            # inclo: inclination (radians)
                meanAnomaly,  # mo: mean anomaly (radians)
                meanMotion,   # no_kozai: mean motion (radians/minute)
                raan,         # nodeo: right ascension of ascending node (radians)
            )

        rs, vs = [], []
        for t in time:
            e, r, v = sat.sgp4_tsince((t - orbit.t) / 60.0)
            rs.append(r)
            vs.append(v)
        rs = np.array(rs)
        vs = np.array(vs)
        tref = self.t if self.t is not None else orbit.t
        rot = teme_to_gcrf(tref)
        rs = np.dot(rot, rs.T).T
        vs = np.dot(rot, vs.T).T
        rs *= 1e3  # km -> m
        vs *= 1e3  # km/s -> m/s
        return rs, vs

    def __hash__(self):
        t = self.t.gps if isinstance(self.t, Time) else self.t
        return hash(("SGP4Propagator", t))

    def __eq__(self, rhs):
        if not isinstance(rhs, SGP4Propagator):
            return False
        return np.all(self.t == rhs.t)


def impact_event(t, s):
    r = s[0:3]
    return np.linalg.norm(r) - EARTH_RADIUS

impact_event.terminal = True
impact_event.direction = -1


class SciPyPropagator(Propagator):
    """Propagate using the scipy.integrate.solve_ivp ODE solver.

    Parameters
    ----------
    accel : ssapy.Accel
        Accel object containing the acceleration model by which to propagate.
    ode_kwargs : dict
        Keyword arguments to pass to `scipy.integrate.solve_ivp`.  Of particular
        interest may be the kwarg `rtol`, which usually yields reasonable
        results when set ~1e-7.  For best results, check for convergence.
    """
    def __init__(self, accel, ode_kwargs=None):
        self.accel = accel
        if ode_kwargs is None:
            ode_kwargs = {'rtol': 1e-7}
        self.ode_kwargs = ode_kwargs

    def __repr__(self):
        return "SciPyPropagator({!r}, {!r})".format(self.accel, self.ode_kwargs)

    @staticmethod
    def _concatenateOdeSolutions(sol0, sol1):
        from scipy.integrate._ivp.common import OdeSolution
        # Ensure solutions are ascending
        if not sol0.ascending:
            sol0 = OdeSolution(sol0.ts[::-1], sol0.interpolants[::-1])
        if not sol1.ascending:
            sol1 = OdeSolution(sol1.ts[::-1], sol1.interpolants[::-1])

        # Just eliminate degenerate solutions
        if sol0.ts.size == 2 and sol0.ts[0] == sol0.ts[1]:
            return sol1
        if sol1.ts.size == 2 and sol1.ts[0] == sol1.ts[1]:
            return sol0

        # Merge non-degenerate solutions
        assert sol0.ts[-1] == sol1.ts[0]
        ts = np.concatenate([sol0.ts[:-1], sol1.ts])
        interpolants = sum([s.interpolants for s in [sol0, sol1]], [])
        return OdeSolution(ts, interpolants)

    def _solve_piecewise_ivp(self, fp, t_span, sol):
        from scipy.integrate import solve_ivp
        # solve from t_span[0] to t_span[1]
        # we need to cut this into pieces at each time in
        # self.accel.time_breakpoints.
        tbreak = self.accel.time_breakpoints
        # make infinities finite but outside the range t_span
        tbreak = np.clip(tbreak, np.min(t_span) - 1, np.max(t_span) + 1)
        tbreakind = np.arange(len(tbreak))
        t_span_ind = np.interp(t_span, tbreak, tbreakind)
        tbreak = tbreak[(tbreakind > np.min(t_span_ind)) & (tbreakind < np.max(t_span_ind))]
        if t_span[1] < t_span[0]:
            tbreak = tbreak[::-1]
        alltimes = np.concatenate([[t_span[0]], tbreak, [t_span[1]]])
        for t0, t1 in zip(alltimes[:-1], alltimes[1:]):
            soln = solve_ivp(
                fp,
                [t0, t1],
                sol(t0),
                dense_output=True,
                events=impact_event,
                **self.ode_kwargs
            )
            if soln.t_events and soln.t_events[0].size > 0:
                print(f"Impact detected at t = {soln.t_events[0][0]:.2f} s")
            if not soln.success:
                raise ValueError(soln.message)
            if t1 > t0:
                sol = self._concatenateOdeSolutions(sol, soln.sol)
            else:
                sol = self._concatenateOdeSolutions(soln.sol, sol)
        return sol

    def _getRVOne(self, orbit, tQuery):
        from scipy.integrate._ivp.base import ConstantDenseOutput
        from scipy.integrate._ivp.common import OdeSolution
        # Pattern for ScipyPropagator interpolant is just:
        # OdeSolution
        container = _InterpolantCache(orbit, self)
        
        def fp(t, s):
            r = s[0:3]
            v = s[3:6]
            return np.hstack([v, self.accel(r, v, t, **orbit.propkw)])

        tmin, tmax = np.min(tQuery), np.max(tQuery)
        update = False
        if len(container) == 0:
            ts = np.array([orbit.t, orbit.t])
            interpolants = [ConstantDenseOutput(
                orbit.t,
                orbit.t,
                np.hstack([orbit.r, orbit.v])
            )]
            sol = OdeSolution(ts, interpolants)
            update = True
        else:
            sol = container[0]
        if tmin < sol.ts[0]:
            sol = self._solve_piecewise_ivp(
                fp,
                [sol.ts[0], tmin],
                sol)
            update = True
        if tmax > sol.ts[-1]:
            sol = self._solve_piecewise_ivp(
                fp,
                [sol.ts[-1], tmax],
                sol)
            update = True
        if update:
            container.clear()
            container.append(sol)
        
        tQuery = tQuery[tQuery <= sol.ts[-1]]
        if len(tQuery) == 0:
            return np.empty((0, 3)), np.empty((0, 3))
    
        out = sol(tQuery).T
        return out[:, 0:3], out[:, 3:6]

    def __hash__(self):
        return hash((
            "SciPyPropagator",
            self.accel,
            frozenset(self.ode_kwargs.items())
        ))

    def __eq__(self, rhs):
        if not isinstance(rhs, SciPyPropagator):
            return False
        return (
            self.accel == rhs.accel and self.ode_kwargs == rhs.ode_kwargs
        )


class RKPropagator(Propagator, ABC):
    """
    Abstract base class for Runge-Kutta-based orbit propagators.

    The `RKPropagator` class provides a framework for implementing orbit propagation 
    using Runge-Kutta methods. Subclasses must override specific methods to define 
    the propagation logic and interpolation behavior.

    Attributes:
        _minPoints (int): Minimum number of points required for interpolation. 
            This should be overridden by subclasses.

    Methods:
        _prop():
            Abstract method that subclasses must implement to define the propagation logic.
        _getRVOne(orbit, tQuery):
            Retrieves the position and velocity of an orbit at specified query times 
            using interpolation.

    Args:
        orbit (Orbit): The orbit object containing initial conditions and propagation settings.
        tQuery (numpy.ndarray): An array of times at which the position and velocity are queried.

    Returns:
        tuple: A tuple containing:
            - r (numpy.ndarray): Position vectors at the query times.
            - v (numpy.ndarray): Velocity vectors at the query times.

    Notes:
        - The `_getRVOne` method uses an interpolation cache to store computed states and times, 
          minimizing redundant computations.
        - The Runge-Kutta propagation logic must be implemented in the `_prop` method by subclasses.
        - Spline interpolation is used for smooth querying of position and velocity data.
        - The `container` object caches the interpolant data, including times, states, and spline objects.

    Example:
        Subclass implementation:
```python
        class RK4Propagator(RKPropagator):
            _minPoints = 4

            def _prop(self, times, states, h, t_target, propkw):
                # Implement RK4 propagation logic here
                ...
```
    Raises:
        NotImplementedError: If `_prop` is not overridden by a subclass.
    """
    _minPoints = None  # subclass should override

    @abstractmethod
    def _prop(self):
        ...  # Subclasses must override

    def _getRVOne(self, orbit, tQuery):
        from collections import deque
        from scipy.interpolate import make_interp_spline
        # Pattern for RK interpolant is:
        # times, states, h_pre, h_app, spline
        container = _InterpolantCache(orbit, self)

        tQuery = np.asarray(tQuery, dtype=np.float64)
        tmin, tmax = np.min(tQuery), np.max(tQuery)

        if len(container) == 0:
            times = deque([np.float64(orbit.t)])
            states = deque([np.asarray(np.hstack([orbit.r, orbit.v]), dtype=np.float64)])
            h_pre = -np.float64(self.h)
            h_app = np.float64(self.h)
            spline = None
        else:
            times, states, h_pre, h_app, spline = container

        remake_spline = False
        if times[0] >= tmin:
            h_pre = self._prop(times, states, h_pre, tmin, orbit.propkw)
            remake_spline = True
        if times[-1] <= tmax:
            h_app = self._prop(times, states, h_app, tmax, orbit.propkw)
            remake_spline = True

        # Always materialize arrays for trimming / lookup
        times_arr = np.asarray(times, dtype=np.float64)
        states_arr = np.asarray(states, dtype=np.float64)
        if times_arr.size == 1:
            # Only defined exactly at the single cached time
            tQuery = tQuery[tQuery == times_arr[0]]
            if tQuery.size == 0:
                return np.empty((0, 3)), np.empty((0, 3))
            out = np.repeat(states_arr, tQuery.size, axis=0)
            return out[:, 0:3], out[:, 3:6]

        if remake_spline or spline is None:
            k = min(3, len(times_arr) - 1)  # cubic, stable
            spline = make_interp_spline(times_arr, states_arr, k=k)
            container.clear()
            container.extend([times, states, h_pre, h_app, spline])
            times_arr = np.asarray(times, dtype=np.float64)
            states_arr = np.asarray(states, dtype=np.float64)

        tQuery = tQuery[(tQuery >= times_arr[0]) & (tQuery <= times_arr[-1])]
        if tQuery.size == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        # Exact knot lookup (prevents tiny spline-at-knot drift)
        idx = np.searchsorted(times_arr, tQuery)
        exact = np.zeros(tQuery.size, dtype=bool)
        ok = idx < times_arr.size
        exact[ok] = np.isclose(times_arr[idx[ok]], tQuery[ok], rtol=0.0, atol=1e-9)

        out = np.empty((tQuery.size, 6), dtype=np.float64)
        if np.all(exact):
            out[:] = states_arr[idx]
        else:
            out[exact] = states_arr[idx[exact]]
            out[~exact] = spline(tQuery[~exact])

        return out[:, 0:3], out[:, 3:6]


class RK4Propagator(RKPropagator):
    """Runge-Kutta 4th order numerical integrator.

    Parameters
    ----------
    accel : ssapy.Accel
        Accel object containing the acceleration model by which to propagate.
    h : float
        Step size in seconds.  Reasonable values are ~50s for LEO propagations
        over a ~day for ~meter accuracy, or 1000s for GEO propagations over a
        few days with ~meter accuracy.  For best results, check for convergence.
    """
    _minPoints = 3

    def __init__(self, accel, h):
        self.accel = accel
        self.h = h

    def __repr__(self):
        return "RK4Propagator({!r}, {!r})".format(self.accel, self.h)

    def _prop(self, times, states, h, tthresh, propkw):
        """Propagate in one direction.

        Parameters
        ----------
        times : collections.deque of float
            Known times.  Must be non-empty.  Will be mutated.
        states : collections.deque of ndarray
            Known states.  Must be non-empty.  Will be mutated.
        h : float
            time increment
        tthresh : float
            Stop when crossing this boundary
        propkw : dict
            orbit.propkw arguments

        Returns
        -------
        h : float
            final time increment
        """
        def fp(s, t):
            r = s[0:3]
            v = s[3:6]
            return np.hstack([v, self.accel(r, v, t, **propkw)])

        if h > 0:
            t = times[-1]
            state = states[-1]
            pred = lambda t: t <= tthresh
        else:
            t = times[0]
            state = states[0]
            pred = lambda t: t >= tthresh

        keepGoing = True
        while keepGoing:
            # test here so we always get 1 extra iteration...
            if not pred(t) and len(times) >= self._minPoints:
                keepGoing = False
            k1 = fp(state, t)
            k2 = fp(state + 0.5 * h * k1, t + 0.5 * h)
            k3 = fp(state + 0.5 * h * k2, t + 0.5 * h)
            k4 = fp(state + h * k3, t + h)
            state = state + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + h
            
            if h > 0:
                states.append(state)
                times.append(t)
            else:
                states.appendleft(state)
                times.appendleft(t)

            # EARTH COLLISION CHECK
            if np.linalg.norm(state[0:3]) <= EARTH_RADIUS:
                print("Collision with Earth detected. Propagation stopped at t =", t)
                break  # Do not add further states — exit

        return h

    def __hash__(self):
        return hash((
            "RK4Propagator",
            self.accel,
            self.h
        ))

    def __eq__(self, rhs):
        if not isinstance(rhs, RK4Propagator):
            return False
        return (
            self.accel == rhs.accel and self.h == rhs.h
        )


class RK8Propagator(RKPropagator):
    """Runge-Kutta 8th order numerical integrator.

    Parameters
    ----------
    accel : ssapy.Accel
        Accel object containing the acceleration model by which to propagate.
    h : float
        Step size in seconds.  ~70s yields accuracy of ~1e-6 meters at GEO over
        a couple of days.  ~20s yields accuracy of ~1e-5 meters at LEO over a
        few hours.
    """
    _minPoints = 7

    def __init__(self, accel, h):
        self.accel = accel
        self.h = h

    # Class level variables for Butcher tableau
    c = np.array([0, 1 / 18, 1 / 12, 1 / 8, 5 / 16, 3 / 8, 59 / 400, 93 / 200, 5490023248 / 9719169821, 13 / 20, 1201146811 / 1299019798, 1, 1], dtype=np.float64)
    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1 / 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1 / 48, 1 / 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1 / 32, 0, 3 / 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [5 / 16, 0, -75 / 64, 75 / 64, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [3 / 80, 0, 0, 3 / 16, 3 / 20, 0, 0, 0, 0, 0, 0, 0, 0],
                  [29443841 / 614563906, 0, 0, 77736538 / 692538347, -28693883 / 1125000000, 23124283 / 1800000000, 0, 0, 0, 0, 0, 0, 0],
                  [16016141 / 946692911, 0, 0, 61564180 / 158732637, 22789713 / 633445777, 545815736 / 2771057229, -180193667 / 1043307555, 0, 0, 0, 0, 0, 0],
                  [39632708 / 573591083, 0, 0, -433636366 / 683701615, -421739975 / 2616292301, 100302831 / 723423059, 790204164 / 839813087, 800635310 / 3783071287, 0, 0, 0, 0, 0],
                  [246121993 / 1340847787, 0, 0, -37695042795 / 15268766246, -309121744 / 1061227803, -12992083 / 490766935, 6005943493 / 2108947869, 393006217 / 1396673457, 123872331 / 1001029789, 0, 0, 0, 0],
                  [-1028468189 / 846180014, 0, 0, 8478235783 / 508512852, 1311729495 / 1432422823, -10304129995 / 1701304382, -48777925059 / 3047939560, 15336726248 / 1032824649, -45442868181 / 3398467696, 3065993473 / 597172653, 0, 0, 0],
                  [185892177 / 718116043, 0, 0, -3185094517 / 667107341, -477755414 / 1098053517, -703635378 / 230739211, 5731566787 / 1027545527, 5232866602 / 850066563, -4093664535 / 808688257, 3962137247 / 1805957418, 65686358 / 487910083, 0, 0],
                  [403863854 / 491063109, 0, 0, -5068492393 / 434740067, -411421997 / 543043805, 652783627 / 914296604, 11173962825 / 925320556, -13158990841 / 6184727034, 3936647629 / 1978049680, -160528059 / 685178525, 248638103 / 1413531060, 0, 0]], dtype=np.float64)
    b8 = np.array([14005451 / 335480064, 0, 0, 0, 0, -59238493 / 1068277825, 181606767 / 758867731, 561292985 / 797845732, -1041891430 / 1371343529, 760417239 / 1151165299, 118820643 / 751138087, -528747749 / 2220607170, 1 / 4], dtype=np.float64)

    def __repr__(self):
        return "RK8Propagator({!r}, {!r})".format(self.accel, self.h)

    def _prop(self, times, states, h, tthresh, propkw):
        """Propagate in one direction.

        Parameters
        ----------
        times : collections.deque of float
            Know times.  Must be non-empty.  Will be mutated.
        state : collections.deque of ndarray
            Known states.  Must be non-empty.  Will be mutated.
        h : float
            time increment
        tthresh : float
            Stop when crossing this boundary
        propkw : dict
            orbit.propkw arguments

        Returns
        -------
        h : float
            final time increment
        """
        # Make local to save typing, and dot lookups.
        c = self.c
        a = self.a
        b8 = self.b8

        # Can go forward or backward depending on sign of h
        def fp(s, t):
            r = s[0:3]
            v = s[3:6]
            return np.hstack([v, self.accel(r, v, t, **propkw)])

        if h > 0:
            t = times[-1]
            state = states[-1]
            pred = lambda t: t <= tthresh
        else:
            t = times[0]
            state = states[0]
            pred = lambda t: t >= tthresh
        keepGoing = True
        while keepGoing:
            # test here so we always get 1 extra iteration, which seems to
            # interpolate better
            if not pred(t) and len(times) >= self._minPoints:
                keepGoing = False
            k = np.zeros((13, 6), dtype=np.float64)
            for i in range(13):
                k[i] = h * fp(state + np.dot(a[i], k), t + c[i] * h)
            state = state + np.dot(b8, k)
            t = t + h
            if h > 0:
                states.append(state)
                times.append(t)
            else:
                states.appendleft(state)
                times.appendleft(t)

            # EARTH COLLISION CHECK
            if np.linalg.norm(state[0:3]) <= EARTH_RADIUS:
                print("Collision with Earth detected. Propagation stopped at t =", t)
                break  # Do not add further states — exit

        return h

    def __hash__(self):
        return hash((
            "RK8Propagator",
            self.accel,
            self.h
        ))

    def __eq__(self, rhs):
        if not isinstance(rhs, RK8Propagator):
            return False
        return (
            self.accel == rhs.accel and self.h == rhs.h
        )


class RK78Propagator(RK8Propagator):
    """Runge-Kutta 8th order numerical integrator with adaptive step size
    computed from embedded 7th order integrator error estimate.

    Parameters
    ----------
    accel : ssapy.Accel
        Accel object containing the acceleration model by which to propagate.
    h : float
        Initial step size in seconds.  A few 10s of seconds is usually a good
        starting point here; it'll automatically be adjusted by the algorithm.
    tol : float or array of float.
        Tolerance for a single integrator step.  Used to adaptively change the
        integrator step size.  Broadcasts to 6-dimensions.  A good target is
        usually ~[1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9] for cm accuracy at GEO
        over a few days, or around LEO over a few hours.
    """
    _minPoints = 7

    def __init__(self, accel, h, tol=(1e-6,) * 3 + (1e-9,) * 3):
        self.accel = accel
        self.h = h
        self.tol = tol

    # Inherit most class vars from RK8Propagator, but need b7 coefficients
    b7 = np.array([13451932 / 455176623, 0, 0, 0, 0, -808719846 / 976000145, 1757004468 / 5645159321, 656045339 / 265891186, -3867574721 / 1518517206, 465885868 / 322736535, 53011238 / 667516719, 2 / 45, 0])

    def __repr__(self):
        return "RK78Propagator({!r}, {!r}, {!r})".format(self.accel, self.h, self.tol)

    def _prop(self, times, states, h, tthresh, propkw):
        """Propagate in one direction.

        Parameters
        ----------
        times : collections.deque of float
            Know times.  Must be non-empty.  Will be mutated.
        state : collections.deque of ndarray
            Known states.  Must be non-empty.  Will be mutated.
        h : float
            time increment
        tthresh : float
            Stop when crossing this boundary
        propkw : dict
            orbit.propkw arguments

        Returns
        -------
        h : float
            final time increment
        """
        # Make local to save typing, and dot lookups.
        c = self.c
        a = self.a
        b8 = self.b8
        b7 = self.b7

        # Can go forward or backward depending on sign of h
        def fp(s, t):
            r = s[0:3]
            v = s[3:6]
            return np.hstack([v, self.accel(r, v, t, **propkw)])

        def step(h, t, state):
            while True:
                k = np.zeros((13, 6), dtype=np.float64)
                for i in range(13):
                    k[i] = h * fp(state + np.dot(a[i], k), t + c[i] * h)
                result7 = state + np.dot(b7, k)
                result8 = state + np.dot(b8, k)
                errmax = np.max(np.abs(result7 - result8) / self.tol)
                if errmax > (1.0):
                    h *= max(0.1, 0.9 * errmax**(-1 / 7))
                    continue
                else:
                    # reset size for next step and break
                    errmax = max(errmax, 1e-15)  # avoid underflow
                    hnext = h * min(5.0, 0.9 * errmax**(-1 / 8))
                    return h, result8, hnext

        if h > 0:
            t = times[-1]
            state = states[-1]
            pred = lambda t: t <= tthresh
        else:
            t = times[0]
            state = states[0]
            pred = lambda t: t >= tthresh

        keepGoing = True
        while keepGoing:
            # test here so we always get 1 extra iteration...
            if not pred(t) and len(times) >= self._minPoints:
                keepGoing = False
            h, state, h_next = step(h, t, state)
            t = t + h
            h = h_next
            if h > 0:
                states.append(state)
                times.append(t)
            else:
                states.appendleft(state)
                times.appendleft(t)

            # EARTH COLLISION CHECK
            if np.linalg.norm(state[0:3]) <= EARTH_RADIUS:
                print("Collision with Earth detected. Propagation stopped at t =", t)
                break  # Do not add further states — exit

        return h

    def __hash__(self):
        return hash((
            "RK78Propagator",
            self.accel,
            self.h,
            self.tol
        ))

    def __eq__(self, rhs):
        if not isinstance(rhs, RK78Propagator):
            return False
        return (
            self.accel == rhs.accel and self.h == rhs.h and self.tol == rhs.tol
        )


def default_numerical(*args, cls=None, accel=None, extra_accel=None):
    """Construct a numerical propagator with sensible default acceleration.

    Parameters
    ----------
    *args : list
        Arguments to Propagator
    cls : Propagator
        class to use.  Default of None means SciPyPropagator.
    accel : Accel
        acceleration model to use.  Default of None means Earth(4, 4), sun, moon.
    extra_accel : Accel or list of Accel, optional
        Additional accelerations to add to `accel`.

    Returns
    -------
    Instance of Propagator with desired Accel model.
    """
    from .accel import AccelKepler, AccelSum
    from .gravity import AccelHarmonic, AccelThirdBody
    from .body import get_body
    if accel is None:
        earth = get_body("earth")
        aK = AccelKepler(earth.mu)
        aH = AccelHarmonic(earth, 4, 4)
        aSun = AccelThirdBody(get_body("sun"))
        aMoon = AccelThirdBody(get_body("moon"))
        accellist = [aK, aH, aSun, aMoon]
        if extra_accel is not None:
            if not isinstance(extra_accel, list):
                extra_accel = [extra_accel]
            accellist += extra_accel
        accel = AccelSum(accellist)
    if cls is None:
        from functools import partial
        cls = partial(SciPyPropagator, ode_kwargs=dict(method='DOP853', rtol=1e-9, atol=(1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4)))
    return cls(accel, *args)
