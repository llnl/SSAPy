"""
A collection of physical constants.

.. data:: WGS84_EARTH_MU

    Earth gravitational constant from WGS84 model [m³/s²]

.. data:: WGS72_EARTH_MU

    Earth gravitational constant from WGS72 model [m³/s²]

.. data:: WGS84_EARTH_OMEGA

    Earth angular velocity from WGS84 model [rad/s]

.. data:: WGS72_EARTH_OMEGA

    Earth angular velocity from WGS72 model [rad/s]

.. data:: WGS84_EARTH_RADIUS

    Earth radius at equator from WGS84 model [m]

.. data:: WGS72_EARTH_RADIUS

    Earth radius at equator from WGS72 model [m]

.. data:: WGS84_EARTH_FLATTENING

    Earth flattening f = (a - b) / a where a and b are the major
    and minor axes respectively, from WGS84 model [unitless]

.. data:: WGS72_EARTH_FLATTENING

    Earth flattening f = (a - b) / a where a and b are the major
    and minor axes respectively, from WGS72 model [unitless]

.. data:: WGS84_EARTH_POLAR_RADIUS

    Earth polar radius, calculated by multiplying the equatorial
    radius by (1 - flattening), from WGS84 model [m]

.. data:: WGS72_EARTH_POLAR_RADIUS

    Earth polar radius, calculated by multiplying the equatorial
    radius by (1 - flattening), from WGS72 model [m]

.. data:: RGEO

    GEO-synchronous radius [m]

.. data:: VGEO

    GEO-synchronous velocity [m/s]

.. data:: RGEOALT

    GEO-synchronous altitude [m]

.. data:: VLEO

    Approximate orbital velocity for low earth orbit (altitude of 500km) [m/s]

.. data:: LD

    Lunar semi-major axis [m]

Gravitational constants
-----------------------

.. data:: SUN_MU

    Sun gravitational constant, from IAU 1976 model [m³/s²]

.. data:: MOON_MU

    Moon gravitational constant, from the DE200 ephemeris [m³/s²]

.. data:: MERCURY_MU

    Mercury gravitational constant [m³/s²]

.. data:: VENUS_MU

    Venus gravitational constant [m³/s²]

.. data:: EARTH_MU

    Earth gravitational constant, from WGS84 model [m³/s²]

.. data:: MARS_MU

    Mars gravitational constant [m³/s²]

.. data:: JUPITER_MU

    Jupiter gravitational constant [m³/s²]

.. data:: SATURN_MU

    Saturn gravitational constant [m³/s²]

.. data:: URANUS_MU

    Uranus gravitational constant [m³/s²]

.. data:: NEPTUNE_MU

    Neptune gravitational constant [m³/s²]

Mass
----

.. data:: SUN_MASS

    Sun mass, from the DE405 ephemeris [kg]

.. data:: MOON_MASS

    Moon mass, from the DE405 ephemeris [kg]

.. data:: MERCURY_MASS

    Mercury mass, from the DE405 ephemeris [kg]

.. data:: VENUS_MASS

    Venus mass, from the DE405 ephemeris [kg]

.. data:: EARTH_MASS

    Earth mass, from the DE405 ephemeris [kg]

.. data:: MARS_MASS

    Mars mass, from the DE405 ephemeris [kg]

.. data:: JUPITER_MASS

    Jupiter mass, from the DE405 ephemeris [kg]

.. data:: SATURN_MASS

    Saturn mass, from the DE405 ephemeris [kg]

.. data:: URANUS_MASS

    Uranus mass, from the DE405 ephemeris [kg]

.. data:: NEPTUNE_MASS

    Neptune mass, from the DE405 ephemeris [kg]

Radius
------

.. data:: MOON_RADIUS

    Moon radius (source: 10.2138/rmg.2006.60.3) [m]

.. data:: MERCURY_RADIUS

    Mercury radius [m]

.. data:: VENUS_RADIUS

    Venus radius [m]

.. data:: EARTH_RADIUS

    Earth radius [m]

.. data:: MARS_RADIUS

    Mars radius [m]

.. data:: JUPITER_RADIUS

    Jupiter radius [m]

.. data:: SATURN_RADIUS

    Saturn radius [m]

.. data:: URANUS_RADIUS

    Uranus radius [m]

.. data:: NEPTUNE_RADIUS

    Neptune radius [m]
"""

import numpy as np

# GM
WGS84_EARTH_MU = 3.986004418e14  # [m^3/s^2]
WGS72_EARTH_MU = 3.986005e14
# angular velocity
WGS84_EARTH_OMEGA = 72.92115147e-6  # [rad/s]
WGS72_EARTH_OMEGA = WGS84_EARTH_OMEGA
# radius at equator
WGS84_EARTH_RADIUS = 6.378137e6  # [m]
WGS72_EARTH_RADIUS = 6.378135e6  # [m]

# flattening f = (a-b)/a with a,b the major,minor axes
WGS84_EARTH_FLATTENING = 1 / 298.257223563
WGS72_EARTH_FLATTENING = 1 / 298.26
# polar radius can be derived from above; [m]
WGS84_EARTH_POLAR_RADIUS = WGS84_EARTH_RADIUS * (1 - WGS84_EARTH_FLATTENING)
WGS72_EARTH_POLAR_RADIUS = WGS72_EARTH_RADIUS * (1 - WGS72_EARTH_FLATTENING)

# GEO-sync radius and velocity are derived.
RGEO = np.cbrt(WGS84_EARTH_MU / WGS84_EARTH_OMEGA**2)  # [m]
VGEO = RGEO * WGS84_EARTH_OMEGA  # [m/s]
RGEOALT = RGEO - WGS84_EARTH_RADIUS  # [m] altitude of GEO
# Rough value:
VLEO = np.sqrt(WGS84_EARTH_MU / (WGS84_EARTH_RADIUS + 500e3))  # [m/s]

# Note JGM3 values from Montenbruck & Gill code are
# reference_radius = 6378.1363e3
# earth_mu = 398600.4415e+9

# VALUES FROM WIKI UNLESS STATED
SUN_MU = 1.32712438e+20  # [m^3/s^2] IAU 1976
MOON_MU = 398600.4415e+9 / 81.300587  # [m^3/s^2] DE200
MERCURY_MU = 2.2032e13
VENUS_MU = 3.24859e14
EARTH_MU = WGS84_EARTH_MU
MARS_MU = 4.282837e13
JUPITER_MU = 1.26686534e17
SATURN_MU = 3.7931187e16
URANUS_MU = 5.793939e15
NEPTUNE_MU = 6.836529e15

# MASS [kg] Values from the DE405 ephemeris
SUN_MASS = 1.98847e+30
MOON_MASS = 7.348e22
MERCURY_MASS = 3.301e23
VENUS_MASS = 4.8675e24
EARTH_MASS = 5.9722e24
MARS_MASS = 6.417e23
JUPITER_MASS = 1.899e27
SATURN_MASS = 5.685e26
URANUS_MASS = 8.682e25
NEPTUNE_MASS = 1.024e26

# RADIUS - MEAN RADIUS FROM WIKI UNLESS STATED
MOON_RADIUS = 1738.1e3  # 10.2138/rmg.2006.60.3
MERCURY_RADIUS = 2439.4e3
VENUS_RADIUS = 6052e3
EARTH_RADIUS = WGS84_EARTH_RADIUS
MARS_RADIUS = 3389.5e3
JUPITER_RADIUS = 69911e3
SATURN_RADIUS = 58232e3
URANUS_RADIUS = 25362e3
NEPTUNE_RADIUS = 24622e3

# Distance from Earth to Moon
LD = 384399000  # lunar semi-major axis in meters
