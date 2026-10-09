from types import SimpleNamespace

import numpy as np
from astropy.coordinates import TETE, get_sun
from astropy.time import Time

from ssapy.accel import AccelDrag


def test_drag_density_sees_the_sun_of_date():
    # Harris-Priester's diurnal bulge follows the Sun's right ascension and
    # declination in the true-of-date frame the density is evaluated in.
    # The values AccelDrag passes must match astropy's true-equator,
    # true-equinox (TETE) Sun to 0.05 deg (fast sunPos is good to 0.6 arcmin);
    # the GCRF values it passed before were 0.37 deg off in RA in 2026.
    t = Time("2026-10-08T00:00:00", scale="utc")
    drag = AccelDrag()
    seen = {}

    def capture(x, y, z, ra_sun, dec_sun):
        seen["ra"], seen["dec"] = ra_sun, dec_sun
        return 1e-12

    drag.atm = SimpleNamespace(density=capture)  # the C++ model's methods are read-only
    drag(np.array([6778e3, 0.0, 0.0]), np.array([0.0, 7670.0, 0.0]), t.gps, area=1.0, mass=100.0, CD=2.2)
    sun = get_sun(t).transform_to(TETE(obstime=t))
    d_ra = (np.degrees(seen["ra"]) - sun.ra.deg + 180.0) % 360.0 - 180.0
    assert abs(d_ra) < 0.05
    assert abs(np.degrees(seen["dec"]) - sun.dec.deg) < 0.05
