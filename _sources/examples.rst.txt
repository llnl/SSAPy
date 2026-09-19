SSAPy by Example
================

The examples below use base SSAPy APIs that are part of the current package.
Higher-level plotting workflows and apparent-magnitude calculations are
maintained in `SSAPy-Toolkit <https://github.com/LLNL/SSAPy-Toolkit>`_.

Define an epoch and a simple geosynchronous orbit:

.. code-block:: python

    import numpy as np
    import astropy.units as u
    from astropy.time import Time

    import ssapy

    t0 = Time("2024-01-01T00:00:00", scale="utc")

    orbit = ssapy.Orbit.fromKeplerianElements(
        ssapy.constants.RGEO,  # semi-major axis [m]
        0.001,                 # eccentricity
        np.radians(45.0),      # inclination [rad]
        0.0,                   # argument of periapsis [rad]
        0.0,                   # right ascension of ascending node [rad]
        np.pi,                 # true anomaly [rad]
        t=t0,
    )

Propagate the orbit at user-selected times:

.. code-block:: python

    times = t0 + np.linspace(0.0, 6.0, 7) * u.hour
    propagator = ssapy.KeplerianPropagator()

    r_gcrf, v_gcrf = ssapy.rv(orbit, times, propagator=propagator)
    print(r_gcrf.shape, v_gcrf.shape)

Compute a ground track. The geodetic form returns longitude, latitude, and
height; the Cartesian form returns International Terrestrial Reference Frame
(ITRF) coordinates.

.. code-block:: python

    lon, lat, height = ssapy.groundTrack(
        orbit, times, propagator=propagator, format="geodetic"
    )

    x_itrf, y_itrf, z_itrf = ssapy.groundTrack(
        r_gcrf, times, format="cartesian"
    )

Compute observer geometry from an Earth-based site:

.. code-block:: python

    observer = ssapy.EarthObserver(lon=-121.76, lat=37.68, elevation=120.0)

    ra, dec, slant_range = ssapy.radec(
        orbit, times, observer=observer, propagator=propagator
    )
    alt, az = ssapy.altaz(
        orbit, times, observer=observer, propagator=propagator
    )

    print(np.degrees(ra[0]), np.degrees(dec[0]), slant_range[0])
    print(np.degrees(alt[0]), np.degrees(az[0]))

Convert between Geocentric Celestial Reference Frame (GCRF) and True Equator
Mean Equinox (TEME) Cartesian coordinates:

.. code-block:: python

    gcrf_to_teme = ssapy.utils.gcrf_to_teme(times)
    teme_to_gcrf = ssapy.utils.teme_to_gcrf(times)

    r_teme = np.einsum("tij,tj->ti", gcrf_to_teme, r_gcrf)
    r_roundtrip = np.einsum("tij,tj->ti", teme_to_gcrf, r_teme)

    print(np.max(np.linalg.norm(r_roundtrip - r_gcrf, axis=1)))

Base SSAPy also includes lower-level coordinate utilities such as
``ssapy.utils.lb_to_unit``, ``ssapy.utils.unit_to_lb``,
``ssapy.utils.rv_to_ntw``, ``ssapy.utils.ntw_to_r``,
``ssapy.compute.radecRateObsToRV``, and ``ssapy.compute.rvObsToRaDecRate``.
