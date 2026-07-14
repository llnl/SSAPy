"""
Path B: physically-propagated orbits seeded from a TLE.

A TLE is an SGP4 object: its mean elements and its B* drag term are defined
*inside* the SGP4 theory and are only self-consistent when propagated by SGP4
(see ``SGP4Propagator`` / ``Orbit.fromTLETuple`` for the exact-reproduction
path, "Path A").

This module supports the other case: taking a TLE as a starting point for
SSAPy's *numerical* force model (two-body + geopotential + third bodies +
Harris-Priester drag).  The result is a genuinely different -- and arguably
higher-fidelity -- dynamical product.  It will NOT equal direct sgp4 and must
not be validated against it; it is validated against truth (owner ephemerides,
precise orbits, or an orbit-determination residual).

Two things a TLE does not give you that the numerical drag model needs:

1. A physical ballistic coefficient.  ``AccelDrag`` wants ``CD``, ``area``,
   ``mass`` (it uses ``CD*area/mass``); a TLE carries only B*, which is scaled
   to SGP4's *reference* atmosphere and is not transferable to Harris-Priester.
   ``bstar_to_cd_a_over_m`` gives a rough SEED only.

2. Model consistency.  The defensible way to pin the physical coefficient is
   ``fit_drag`` -- estimate ``Cd*A/m`` (and optionally the epoch state) by
   least squares against a reference arc.
"""

import numpy as np

# SGP4 reference density used to map B* -> physical Cd*A/m.
# B* = (Cd*A/m) * rho0 / 2   =>   Cd*A/m = 2*B* / rho0.
# rho0 in kg/m^2/Earth-radius (Vallado, canonical->SI). This is only a seed:
# B* is a fitted SGP4 parameter, not a physical ballistic coefficient, and the
# SGP4 reference atmosphere differs from Harris-Priester used by AccelDrag.
SGP4_RHO0 = 0.1570  # kg / m^2 / Earth-radius


def bstar_to_cd_a_over_m(bstar, rho0=SGP4_RHO0):
    """Rough physical drag ratio Cd*A/m [m^2/kg] implied by an SGP4 B* term.

    Parameters
    ----------
    bstar : float
        SGP4 B* drag term (as carried on ``Satrec.bstar``), units 1/Earth-radii.
    rho0 : float, optional
        SGP4 reference density [kg/m^2/Earth-radius]. Default ``SGP4_RHO0``.

    Returns
    -------
    float
        Cd*A/m in m^2/kg.

    Notes
    -----
    Approximate and model-inconsistent -- use only as a starting guess for
    ``fit_drag``. Non-positive B* (common for high or maneuvering objects)
    yields a non-positive result and should be replaced by a physical prior.
    """
    return 2.0 * bstar / rho0


def propkw_from_cd_a_over_m(cd_a_over_m, cr_a_over_m=0.0):
    """Build a ``propkw`` dict encoding physical drag/SRP ratios.

    ``AccelDrag`` uses ``CD*area/mass`` and ``AccelSolRad``/``AccelEarthRad``
    use ``CR*area/mass``. By fixing ``area = mass = 1`` the ``CD`` and ``CR``
    entries carry the full m^2/kg ratios directly, so only the physically
    meaningful ratios need to be specified or fit.
    """
    return dict(area=1.0, mass=1.0,
                CD=float(cd_a_over_m), CR=float(cr_a_over_m))


def drag_propagator(harmonics=(4, 4), third_bodies=("sun", "moon"),
                    ode_kwargs=None):
    """Numerical propagator with two-body + geopotential + third bodies + drag.

    Parameters
    ----------
    harmonics : 2-tuple of int, optional
        (n, m) degree/order of the Earth geopotential. Default (4, 4).
    third_bodies : iterable of str, optional
        Third-body point-mass perturbers. Default sun and moon.
    ode_kwargs : dict, optional
        Overrides for the SciPy integrator (default DOP853, rtol 1e-9).

    Returns
    -------
    SciPyPropagator
    """
    from functools import partial
    from .accel import AccelKepler, AccelSum, AccelDrag
    from .gravity import AccelHarmonic, AccelThirdBody
    from .body import get_body
    from .propagator import SciPyPropagator

    earth = get_body("earth")
    accels = [AccelKepler(earth.mu), AccelHarmonic(earth, *harmonics)]
    for name in third_bodies:
        accels.append(AccelThirdBody(get_body(name)))
    accels.append(AccelDrag())
    accel = AccelSum(accels)

    if ode_kwargs is None:
        ode_kwargs = dict(method="DOP853", rtol=1e-9,
                          atol=(1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4))
    return partial(SciPyPropagator, ode_kwargs=ode_kwargs)(accel)


def numerical_from_tle(tle, cd_a_over_m=None, cr_a_over_m=0.0,
                       harmonics=(4, 4), propagator=None):
    """Build an ``Orbit`` (epoch state from the TLE) plus a drag propagator.

    Parameters
    ----------
    tle : 2-tuple of str
        Line1, Line2.
    cd_a_over_m : float, optional
        Physical Cd*A/m [m^2/kg]. If None, seeded from B* via
        ``bstar_to_cd_a_over_m`` (approximate -- prefer ``fit_drag``).
    cr_a_over_m : float, optional
        Physical Cr*A/m [m^2/kg] for SRP. Default 0 (drag only).
    harmonics : 2-tuple of int, optional
        Geopotential degree/order. Default (4, 4).
    propagator : Propagator, optional
        Prebuilt propagator; if None one is created via ``drag_propagator``.

    Returns
    -------
    orbit : Orbit
        Epoch state (GCRF) with ``propkw`` populated.
    propagator : Propagator
    """
    from .orbit import Orbit

    orbit = Orbit.fromTLETuple(tle)
    if cd_a_over_m is None:
        cd_a_over_m = bstar_to_cd_a_over_m(orbit._sat.bstar)
    orbit.propkw = propkw_from_cd_a_over_m(cd_a_over_m, cr_a_over_m)
    if propagator is None:
        propagator = drag_propagator(harmonics=harmonics)
    return orbit, propagator


def _sgp4_reference_arc(tle, times):
    """Reference GCRF positions from the TLE's own SGP4 over ``times``.

    Convenience truth stand-in for demos/tests. For a real product, pass owner
    ephemeris or a precise orbit as the reference instead -- fitting a physical
    model to SGP4 output is only a pipeline demonstration.
    """
    from .orbit import Orbit
    from .propagator import SGP4Propagator
    from .compute import rv
    orbit = Orbit.fromTLETuple(tle)               # Path A: native Satrec
    prop = SGP4Propagator()
    r = np.array([rv(orbit, t, propagator=prop)[0] for t in times])
    return orbit, r


def fit_drag(orbit, times, r_ref, propagator=None, harmonics=(4, 4),
             cd_a_over_m0=None, bounds=(1e-6, 1.0), verbose=False):
    """Fit physical Cd*A/m so numerical propagation matches a reference arc.

    This is the defensible way to give a TLE-seeded numerical orbit a physical
    drag coefficient: solve for the single scalar Cd*A/m [m^2/kg] that minimizes
    position residuals of the full numerical force model against ``r_ref``.

    Parameters
    ----------
    orbit : Orbit
        Epoch state (e.g. from ``Orbit.fromTLETuple``).
    times : array_like (m,)
        GPS seconds at which the reference is sampled.
    r_ref : array_like (m, 3)
        Reference GCRF positions [m] (owner ephemeris / precise orbit; or, for
        a demo, ``_sgp4_reference_arc`` output).
    propagator : Propagator, optional
        Drag propagator; built via ``drag_propagator`` if None.
    harmonics : 2-tuple of int, optional
        Geopotential degree/order for the built propagator. Default (4, 4).
    cd_a_over_m0 : float, optional
        Initial guess [m^2/kg]. If None and the orbit carries a native Satrec,
        seeded from B*; otherwise 0.02.
    bounds : 2-tuple, optional
        (lower, upper) bounds on Cd*A/m for the solver.
    verbose : bool, optional
        Print the residual per iteration.

    Returns
    -------
    dict
        ``cd_a_over_m`` (fit), ``propkw``, ``orbit`` (copy with propkw set),
        ``propagator``, ``rms_before`` and ``rms_after`` (RMS 3D position error
        [m] at the seed vs the fit), and ``max_after`` (max error [m]).
    """
    from scipy.optimize import least_squares
    from .compute import rv

    times = np.asarray(times, dtype=float)
    r_ref = np.asarray(r_ref, dtype=float)
    if propagator is None:
        propagator = drag_propagator(harmonics=harmonics)

    if cd_a_over_m0 is None:
        sat = getattr(orbit, "_sat", None)
        seed = bstar_to_cd_a_over_m(sat.bstar) if sat is not None else 0.02
        cd_a_over_m0 = seed if bounds[0] < seed < bounds[1] else 0.02

    def _propagate(cd):
        orbit.propkw = propkw_from_cd_a_over_m(cd)
        r = np.array([rv(orbit, t, propagator=propagator)[0] for t in times])
        return r

    def _rms3d(r):
        return float(np.sqrt(np.mean(np.sum((r - r_ref) ** 2, axis=1))))

    def resid(p):
        r = _propagate(p[0])
        if verbose:
            print(f"  Cd*A/m={p[0]:.5e}  rms(3D)={_rms3d(r):.1f} m")
        return (r - r_ref).ravel()

    rms_before = _rms3d(_propagate(cd_a_over_m0))

    opt = least_squares(resid, [cd_a_over_m0],
                        bounds=([bounds[0]], [bounds[1]]),
                        xtol=1e-12, ftol=1e-12)
    cd_fit = float(opt.x[0])

    r_fit = _propagate(cd_fit)
    err = np.linalg.norm(r_fit - r_ref, axis=1)
    rms_after = float(np.sqrt(np.mean(err ** 2)))

    propkw = propkw_from_cd_a_over_m(cd_fit)
    orbit.propkw = dict(propkw)
    return dict(cd_a_over_m=cd_fit, propkw=propkw, orbit=orbit,
                propagator=propagator, rms_before=rms_before,
                rms_after=rms_after, max_after=float(err.max()))
