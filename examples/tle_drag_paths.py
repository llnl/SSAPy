"""
Two ways to use a TLE in SSAPy.

Path A  -- exact SGP4 reproduction.
    ``Orbit.fromTLETuple`` now retains the native SGP4 record (B* and the
    mean-motion-rate terms), and ``SGP4Propagator`` propagates with it.  Output
    matches direct ``sgp4.api.Satrec`` to machine precision, including for
    drag-sensitive LEO objects such as the ISS.  This makes SSAPy a faithful
    TLE-native baseline -- but note it is *calling* SGP4, not replacing it.

Path B  -- physical numerical propagation seeded from the TLE.
    A TLE gives only B*, which is tied to SGP4's reference atmosphere and is not
    a physical ballistic coefficient.  ``ssapy.tle_drag`` seeds a physical
    Cd*A/m from B* (rough) and fits it by orbit determination against a
    reference arc so SSAPy's numerical force model (two-body + geopotential +
    Sun/Moon + Harris-Priester drag) tracks that arc.  The result is a
    *different* dynamical product: it does not equal SGP4 and must be validated
    against truth (owner ephemeris / precise orbit), not against SGP4.

Run:  python examples/tle_drag_paths.py
"""

import warnings
import numpy as np
from sgp4.api import Satrec
from astropy.time import Time
from astropy import units as u

import ssapy
from ssapy import Orbit
from ssapy.propagator import SGP4Propagator
from ssapy.compute import rv
from ssapy.utils import teme_to_gcrf
from ssapy.tle_drag import (bstar_to_cd_a_over_m, numerical_from_tle,
                            _sgp4_reference_arc, fit_drag)

warnings.filterwarnings("ignore")

ISS = ("1 25544U 98067A   24015.54791435  .00016717  00000-0  30074-3 0  9993",
       "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49514637123456")
GEO = ("1 41866U 16071A   24015.50000000 -.00000267  00000-0  00000+0 0  9990",
       "2 41866   0.0177  86.6394 0001679 191.2716 168.7788  1.00272058 26978")


def _direct_sgp4_gcrf(tle, dt_minutes):
    sat = Satrec.twoline2rv(*tle)
    t0 = Time(sat.jdsatepoch, format="jd") + sat.jdsatepochF * u.d
    gps = t0.gps + np.asarray(dt_minutes, dtype=float) * 60.0
    rs = np.array([np.array(sat.sgp4_tsince(dt)[1]) * 1e3 for dt in dt_minutes])
    rot = teme_to_gcrf(gps)                      # per-output-time (vectorized)
    return np.einsum("nij,nj->ni", rot, rs)


def demo_path_a():
    print("=" * 68)
    print("PATH A -- SSAPy SGP4Propagator vs direct sgp4 (should be ~0)")
    print("=" * 68)
    dts = [0.0, 60.0, 6 * 60.0, 12 * 60.0, 24 * 60.0]
    for name, tle in [("ISS (LEO ~420 km, high drag)", ISS),
                      ("GEO (negligible drag)", GEO)]:
        ref = _direct_sgp4_gcrf(tle, dts)
        orb = Orbit.fromTLETuple(tle)
        prop = SGP4Propagator()
        got = np.array([rv(orb, orb.t + dt * 60.0, propagator=prop)[0]
                        for dt in dts])
        err = np.linalg.norm(ref - got, axis=1)
        print(f"\n{name}:")
        for dt, e in zip(dts, err):
            print(f"   t0 + {dt/60:5.1f} h : {e:.3e} m")


def demo_path_b():
    print("\n" + "=" * 68)
    print("PATH B -- physical numerical propagation seeded/fit from a TLE")
    print("=" * 68)
    orb0, _ = _sgp4_reference_arc(ISS, [0.0])
    t0 = orb0.t
    times = t0 + np.arange(0, 24 * 60 + 1, 30) * 60.0     # 1 day, 30-min steps
    _, r_ref = _sgp4_reference_arc(ISS, times)            # truth stand-in

    seed = bstar_to_cd_a_over_m(orb0._sat.bstar)
    print(f"\nCd*A/m seed from B*         : {seed:.4e} m^2/kg  (approximate)")

    orb, prop = numerical_from_tle(ISS)                   # unfit, B* seed
    r_seed = np.array([rv(orb, t, propagator=prop)[0] for t in times])
    rms_seed = np.sqrt(np.mean(np.sum((r_seed - r_ref) ** 2, axis=1)))
    print(f"numerical(seed) vs SGP4 arc : RMS(3D) {rms_seed:8.1f} m")

    res = fit_drag(orb0, times, r_ref)
    print(f"fit Cd*A/m                  : {res['cd_a_over_m']:.4e} m^2/kg")
    print(f"numerical(fit)  vs SGP4 arc : RMS(3D) {res['rms_after']:8.1f} m"
          f"   max {res['max_after']:8.1f} m")
    print("\nThe fit residual is a floor set by the SGP4-vs-numerical model")
    print("difference (Harris-Priester atmosphere, full force model), not by")
    print("integration error -- so Path B is a distinct product, not an SGP4")
    print("reproduction.  Validate it against real ephemeris, where it may")
    print("outperform SGP4.")


if __name__ == "__main__":
    demo_path_a()
    demo_path_b()
