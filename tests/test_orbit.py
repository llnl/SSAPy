import unittest
import numbers
import numpy as np
from astropy.time import Time
import astropy.units as u
import pytest

import ssapy
from ssapy.orbit import _ellipticalEccentricToTrueAnomaly, _ellipticalTrueToEccentricAnomaly
from ssapy.orbit import _hyperbolicEccentricToTrueAnomaly, _hyperbolicTrueToEccentricAnomaly
from ssapy.orbit import _ellipticalEccentricToMeanAnomaly, _ellipticalMeanToEccentricAnomaly
from ssapy.orbit import _hyperbolicEccentricToMeanAnomaly, _hyperbolicMeanToEccentricAnomaly
from ssapy.orbit import _ellipticalEccentricToTrueLongitude, _ellipticalTrueToEccentricLongitude
from ssapy.orbit import _hyperbolicEccentricToTrueLongitude, _hyperbolicTrueToEccentricLongitude
from ssapy.orbit import _ellipticalEccentricToMeanLongitude, _ellipticalMeanToEccentricLongitude
from ssapy.orbit import _hyperbolicEccentricToMeanLongitude, _hyperbolicMeanToEccentricLongitude
from ssapy.utils import normed, norm, teme_to_gcrf
from . import ssapy_test_helpers
from .ssapy_test_helpers import timer, checkAngle, checkSphere, sample_orbit, sample_LEO_orbit, sample_GEO_orbit


try:
    import orekit
    orekit.initVM()
    from orekit.pyhelpers import setup_orekit_curdir
    setup_orekit_curdir()
    from org.orekit.time import TimeScalesFactory
    ts = TimeScalesFactory.getUTC()
except:
    has_orekit = False
else:
    has_orekit = True
print("Has orekit? ", has_orekit)

# some versions of numpy-with-mkl fail the following test, and subsequently fail
# other tests down below at the default tolerance.  So check first here and
# degrade tolerance if required.
try:
    n = 102
    l = np.linspace(0, 2*np.pi, n*n).reshape((n, n))
    cl = np.cos(l)
    for i in range(n):
        np.testing.assert_array_equal(cl[i], np.cos(l[i]))
except AssertionError:
    print("Using loose tolerances")
    r_atol = 1e-5
    v_atol = 1e-9
    dc_atol = 1e-11
    mp_atol = 1e-4  # ugh!
else:
    print("Using tight tolerances")
    r_atol = 1e-15
    v_atol = 1e-15
    dc_atol = 1e-15
    mp_atol = 1e-15


@timer
def test_anomaly_conversion():
    np.random.seed(5)
    # Test eccentric -> true -> eccentric
    eccentric_anomalies = np.random.uniform(0, 2*np.pi, size=100)

    for e in np.random.uniform(0, 1, size=100):
        true_anomalies = _ellipticalEccentricToTrueAnomaly(eccentric_anomalies, e)
        eccentric_anomalies2 = _ellipticalTrueToEccentricAnomaly(true_anomalies, e)
        checkAngle(eccentric_anomalies, eccentric_anomalies2)

    for e in np.random.uniform(1, 2, size=100):
        true_anomalies = _hyperbolicEccentricToTrueAnomaly(eccentric_anomalies, e)
        eccentric_anomalies2 = _hyperbolicTrueToEccentricAnomaly(true_anomalies, e)
        # I don't think hyperbolic anomalies are really angles in the way that elliptical
        # anomalies are, so don't assume close mod 2*pi is sufficient.
        np.testing.assert_allclose(eccentric_anomalies, eccentric_anomalies2, rtol=0, atol=1e-10)

    # Test eccentric -> mean -> eccentric
    for e in np.random.uniform(0, 1, size=100):
        mean_anomalies = _ellipticalEccentricToMeanAnomaly(eccentric_anomalies, e)
        eccentric_anomalies2 = _ellipticalMeanToEccentricAnomaly(mean_anomalies, e)
        checkAngle(eccentric_anomalies, eccentric_anomalies2, atol=1e-11)

    for e in np.random.uniform(1, 2, size=100):
        mean_anomalies = _hyperbolicEccentricToMeanAnomaly(eccentric_anomalies, e)
        # Don't worry about converting large mean anomalies back to eccentric.
        # I haven't checked yet, but I'm guess this means the object is pretty far from the
        # hyperbola vertex.
        w = np.abs(mean_anomalies) < 40
        eccentric_anomalies2 = _hyperbolicMeanToEccentricAnomaly(mean_anomalies[w], e)
        np.testing.assert_allclose(eccentric_anomalies[w], eccentric_anomalies2, rtol=1e-3, atol=1e-8)


@timer
def test_longitude_conversion():
    np.random.seed(57)

    # Test eccentric -> true -> eccentric
    eccentric_longitudes = np.random.uniform(0, 2*np.pi, size=100)

    # elliptic
    for e, phase in zip(np.random.uniform(0, 1, size=100), np.random.uniform(0, 2*np.pi, size=100)):
        ex = e * np.cos(phase)
        ey = e * np.sin(phase)
        true_longitudes = _ellipticalEccentricToTrueLongitude(eccentric_longitudes, ex, ey)
        eccentric_longitudes2 = _ellipticalTrueToEccentricLongitude(true_longitudes, ex, ey)
        checkAngle(eccentric_longitudes, eccentric_longitudes2, atol=1e-13)

    # hyperbolic
    for e, phase in zip(np.random.uniform(1, 2, size=100), np.random.uniform(0, 2*np.pi, size=100)):
        ex = e * np.cos(phase)
        ey = e * np.sin(phase)
        true_longitudes = _hyperbolicEccentricToTrueLongitude(eccentric_longitudes, ex, ey)
        eccentric_longitudes2 = _hyperbolicTrueToEccentricLongitude(true_longitudes, ex, ey)
        # Don't use checkAngle for hyperbolic longitudes
        np.testing.assert_allclose(eccentric_longitudes, eccentric_longitudes2, rtol=0, atol=1e-11)

    # Test eccentric -> mean -> eccentric
    for e, phase in zip(np.random.uniform(0, 1, size=100), np.random.uniform(0, 2*np.pi, size=100)):
        ex = e * np.cos(phase)
        ey = e * np.sin(phase)
        mean_longitudes = _ellipticalEccentricToMeanLongitude(eccentric_longitudes, ex, ey)
        eccentric_longitudes2 = _ellipticalMeanToEccentricLongitude(mean_longitudes, ex, ey)
        checkAngle(eccentric_longitudes, eccentric_longitudes2, atol=1e-11)

    # hyperbolic
    for e, phase in zip(np.random.uniform(1, 2, size=100), np.random.uniform(0, 2*np.pi, size=100)):
        ex = e * np.cos(phase)
        ey = e * np.sin(phase)
        mean_longitudes = _hyperbolicEccentricToMeanLongitude(eccentric_longitudes, ex, ey)
        # Don't worry about converting large mean longitudes back to eccentric.
        # I haven't checked yet, but I'm guess this means the object is pretty far from the
        # hyperbola vertex.
        w = np.abs(mean_longitudes) < 40
        eccentric_longitudes2 = _hyperbolicMeanToEccentricLongitude(mean_longitudes[w], ex, ey)
        np.testing.assert_allclose(eccentric_longitudes[w], eccentric_longitudes2, rtol=1e-3, atol=1e-8)


@timer
def test_orbit_ctor():
    np.random.seed(577)

    rs = []
    vs = []
    eqElts = []
    kElts = []
    for _ in range(1000):
        orbit = sample_GEO_orbit(t=0.0)
        rs.append(orbit.r)
        vs.append(orbit.v)

        eqElts.append(orbit.equinoctialElements)
        kElts.append(orbit.keplerianElements)
        orbit2 = ssapy.Orbit.fromEquinoctialElements(*orbit.equinoctialElements, orbit.t)
        orbit3 = ssapy.Orbit.fromKeplerianElements(*orbit.keplerianElements, orbit.t)
        np.testing.assert_equal(orbit.equinoctialElements, orbit2.equinoctialElements)
        np.testing.assert_equal(orbit.keplerianElements, orbit3.keplerianElements)

        np.testing.assert_allclose(orbit.keplerianElements, orbit2.keplerianElements)
        np.testing.assert_allclose(orbit.equinoctialElements, orbit3.equinoctialElements)

        # Check equinoctial element definitions
        for orb in [orbit, orbit2, orbit3]:
            lonPa = orb.pa + orb.raan
            np.testing.assert_allclose(orb.ex, orb.e*np.cos(lonPa))
            np.testing.assert_allclose(orb.ey, orb.e*np.sin(lonPa))
            np.testing.assert_allclose(orb.hx, np.tan(orb.i/2)*np.cos(orb.raan))
            np.testing.assert_allclose(orb.hy, np.tan(orb.i/2)*np.sin(orb.raan))
            checkAngle(orb.lv, orb.trueAnomaly + lonPa, atol=1e-13)
            np.testing.assert_allclose(orbit.r, orb.r, atol=1e-4, rtol=0)
            np.testing.assert_allclose(orbit.v, orb.v, atol=1e-8, rtol=0)
            np.testing.assert_equal(0, orb.t)

        orbit = sample_LEO_orbit(t=0.0)
        rs.append(orbit.r)
        vs.append(orbit.v)

        eqElts.append(orbit.equinoctialElements)
        kElts.append(orbit.keplerianElements)
        orbit2 = ssapy.Orbit.fromEquinoctialElements(*orbit.equinoctialElements, orbit.t)
        orbit3 = ssapy.Orbit.fromKeplerianElements(*orbit.keplerianElements, orbit.t)
        np.testing.assert_equal(orbit.equinoctialElements, orbit2.equinoctialElements)
        np.testing.assert_equal(orbit.keplerianElements, orbit3.keplerianElements)

        np.testing.assert_allclose(orbit.keplerianElements, orbit2.keplerianElements)
        np.testing.assert_allclose(orbit.equinoctialElements, orbit3.equinoctialElements)

        # Check equinoctial element definitions
        for orb in [orbit, orbit2, orbit3]:
            lonPa = orb.pa + orb.raan
            np.testing.assert_allclose(orb.ex, orb.e*np.cos(lonPa))
            np.testing.assert_allclose(orb.ey, orb.e*np.sin(lonPa))
            np.testing.assert_allclose(orb.hx, np.tan(orb.i/2)*np.cos(orb.raan))
            np.testing.assert_allclose(orb.hy, np.tan(orb.i/2)*np.sin(orb.raan))
            checkAngle(orb.lv, orb.trueAnomaly + lonPa, atol=1e-13)
            np.testing.assert_allclose(orbit.r, orb.r, atol=1e-4, rtol=0)
            np.testing.assert_allclose(orbit.v, orb.v, atol=1e-8, rtol=0)
            np.testing.assert_equal(0, orb.t)

    # Construct an "Orbit" with array arguments
    rs = np.array(rs)
    vs = np.array(vs)
    orbits = ssapy.Orbit(rs, vs, 0.0)
    np.testing.assert_allclose(
        np.array(eqElts),
        np.array(orbits.equinoctialElements).T,
        rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        np.array(kElts),
        np.array(orbits.keplerianElements).T,
        rtol=0, atol=1e-15
    )
    orbits2 = ssapy.Orbit.fromEquinoctialElements(*orbits.equinoctialElements, 0)
    np.testing.assert_allclose(
        np.array(orbits.r),
        np.array(orbits2.r),
        rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        np.array(orbits.v),
        np.array(orbits2.v),
        rtol=0, atol=1e-8
    )
    orbits3 = ssapy.Orbit.fromKeplerianElements(*orbits.keplerianElements, 0)
    np.testing.assert_allclose(
        np.array(orbits.r),
        np.array(orbits3.r),
        rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        np.array(orbits.v),
        np.array(orbits3.v),
        rtol=0, atol=1e-8
    )


@timer
def test_orbit_hyper_ctor():
    np.random.seed(5772)

    rs = []
    vs = []
    eqElts = []
    kElts = []
    for _ in range(1000):
        # Pick a distance near GEO
        r = np.random.uniform(4e7, 5e7)
        # Pick velocity large enough to ensure a hyperbolic orbit
        vmin = np.sqrt(2*ssapy.constants.WGS84_EARTH_MU/r)
        v = np.random.uniform(vmin, vmin*2)
        # Pick a random direction (not uniform on sphere)
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        r = np.array([x, y, z])
        rs.append(r)
        # Repeat for velocity
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        vx = v * np.cos(theta) * np.sin(phi)
        vy = v * np.sin(theta) * np.sin(phi)
        vz = v * np.cos(phi)
        v = np.array([vx, vy, vz])
        vs.append(v)

        orbit = ssapy.Orbit(r, v, 0.0)
        eqElts.append(orbit.equinoctialElements)
        kElts.append(orbit.keplerianElements)
        assert orbit.a < 0
        assert orbit.e > 1

        orbit2 = ssapy.Orbit.fromEquinoctialElements(*orbit.equinoctialElements, orbit.t)
        orbit3 = ssapy.Orbit.fromKeplerianElements(*orbit.keplerianElements, orbit.t)
        np.testing.assert_equal(orbit.equinoctialElements, orbit2.equinoctialElements)
        np.testing.assert_equal(orbit.keplerianElements, orbit3.keplerianElements)

        np.testing.assert_allclose(orbit.keplerianElements, orbit2.keplerianElements)
        np.testing.assert_allclose(orbit.equinoctialElements, orbit3.equinoctialElements)

        # Check equinoctial element definitions
        for orb in [orbit, orbit2, orbit3]:
            checkAngle(orb.lonPa, orb.pa+orb.raan, atol=1e-13)
            np.testing.assert_allclose(orb.ex, orb.e*np.cos(orb.lonPa))
            np.testing.assert_allclose(orb.ey, orb.e*np.sin(orb.lonPa))
            np.testing.assert_allclose(orb.hx, np.tan(orb.i/2)*np.cos(orb.raan))
            np.testing.assert_allclose(orb.hy, np.tan(orb.i/2)*np.sin(orb.raan))
            checkAngle(orb.lv, orb.trueAnomaly+orb.lonPa, atol=1e-14)
            np.testing.assert_allclose(r, orb.r, atol=1e-5, rtol=0)
            np.testing.assert_allclose(v, orb.v, atol=1e-9, rtol=0)
            np.testing.assert_equal(0, orb.t)

    rs = np.array(rs)
    vs = np.array(vs)
    orbits = ssapy.Orbit(rs, vs, 0.0)
    np.testing.assert_allclose(
        np.array(eqElts),
        np.array(orbits.equinoctialElements).T,
        rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        np.array(kElts),
        np.array(orbits.keplerianElements).T,
        rtol=0, atol=1e-15
    )
    orbits2 = ssapy.Orbit.fromEquinoctialElements(*orbits.equinoctialElements, 0)
    np.testing.assert_allclose(
        np.array(orbits.r),
        np.array(orbits2.r),
        rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        np.array(orbits.v),
        np.array(orbits2.v),
        rtol=0, atol=1e-8
    )
    orbits3 = ssapy.Orbit.fromKeplerianElements(*orbits.keplerianElements, 0)
    np.testing.assert_allclose(
        np.array(orbits.r),
        np.array(orbits3.r),
        rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        np.array(orbits.v),
        np.array(orbits3.v),
        rtol=0, atol=1e-8
    )


@timer
def test_orbit_rv():
    np.random.seed(57721)

    for _ in range(1000):
        # Test near GEO
        orbit = sample_GEO_orbit(t=0.0)
        true_anomalies = np.random.uniform(-2*np.pi, 2*np.pi, size=1)
        true_longitudes = true_anomalies + orbit.pa + orbit.raan
        r1, v1 = orbit._rvFromKeplerian(np.atleast_2d(true_anomalies))
        r2, v2 = orbit._rvFromEquinoctial(lv=np.atleast_2d(true_longitudes))
        np.testing.assert_allclose(r1, r2, atol=1e-4, rtol=1e-13)
        np.testing.assert_allclose(v1, v2, atol=1e-6, rtol=1e-11)

        # Repeat near LEO
        orbit = sample_LEO_orbit(t=0.0)
        true_anomalies = np.random.uniform(-2*np.pi, 2*np.pi, size=1)
        true_longitudes = true_anomalies + orbit.pa + orbit.raan
        r1, v1 = orbit._rvFromKeplerian(np.atleast_2d(true_anomalies))
        r2, v2 = orbit._rvFromEquinoctial(lv=np.atleast_2d(true_longitudes))
        np.testing.assert_allclose(r1, r2, atol=1e-4, rtol=1e-13)
        np.testing.assert_allclose(v1, v2, atol=1e-6, rtol=1e-11)


@unittest.skipIf(not has_orekit, 'Unable to import orekit')
@timer
def test_orekit():
    from org.orekit.propagation.analytical import KeplerianPropagator as KP
    from org.orekit.frames import FramesFactory, TopocentricFrame
    from org.orekit.utils import PVCoordinates, Constants, IERSConventions
    from org.orekit.orbits import CartesianOrbit, KeplerianOrbit
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.time import TimeScalesFactory, AbsoluteDate
    from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint

    J2000 = FramesFactory.getEME2000()

    def arrToVec3(arr):
        return Vector3D(float(arr[0]), float(arr[1]), float(arr[2]))
    def vec3ToArr(vec3):
        return np.array([vec3.x, vec3.y, vec3.z])

    np.random.seed(577215)

    t0 = Time('J2000.0')

    for i in range(100):

        orbit = sample_GEO_orbit(t0)
        true_anomalies = np.random.uniform(-2*np.pi, 2*np.pi, size=1)

        # Now construct an orekit orbit and compare
        oreOrbit = CartesianOrbit(
            PVCoordinates(arrToVec3(r), arrToVec3(v)),
            J2000,
            AbsoluteDate(t0.utc.isot, TimeScalesFactory.getUTC()),
            ssapy.constants.WGS84_EARTH_MU
        )

        np.testing.assert_allclose(orbit.period, oreOrbit.keplerianPeriod)
        np.testing.assert_allclose(orbit.meanMotion, oreOrbit.keplerianMeanMotion)
        np.testing.assert_allclose(orbit.a, oreOrbit.a)
        np.testing.assert_allclose(orbit.e, oreOrbit.e)
        np.testing.assert_allclose(orbit.ex, oreOrbit.equinoctialEx)
        np.testing.assert_allclose(orbit.ey, oreOrbit.equinoctialEy)
        np.testing.assert_allclose(orbit.hx, oreOrbit.hx)
        np.testing.assert_allclose(orbit.hy, oreOrbit.hy)
        np.testing.assert_allclose(orbit.i, oreOrbit.i)
        np.testing.assert_allclose(orbit.lE, oreOrbit.lE)
        np.testing.assert_allclose(orbit.lM, oreOrbit.lM)
        np.testing.assert_allclose(orbit.lv, oreOrbit.lv)

        korbit = KeplerianOrbit(oreOrbit)
        checkAngle(orbit.eccentricAnomaly, korbit.eccentricAnomaly, atol=1e-12)
        checkAngle(orbit.meanAnomaly, korbit.meanAnomaly, atol=1e-12)
        checkAngle(orbit.trueAnomaly, korbit.trueAnomaly, atol=1e-12)
        checkAngle(orbit.pa, korbit.perigeeArgument, atol=1e-12)
        checkAngle(orbit.raan, korbit.rightAscensionOfAscendingNode, atol=1e-12)

        ts = t0 + np.random.uniform(-10, 10, size=100)*u.d

        kprop = KP(oreOrbit)
        oreRV = []
        for t in ts:
            orv = kprop.getPVCoordinates(
                AbsoluteDate(t.utc.isot, TimeScalesFactory.getUTC()),
                J2000
            )
            oreRV.append([vec3ToArr(orv.position), vec3ToArr(orv.velocity)])
        oreRV = np.array(oreRV).transpose((1,0,2))

        r, v = ssapy.rv(orbit, ts)

        # I'm a bit sad these aren't better...
        np.testing.assert_allclose(oreRV[0], r, rtol=3e-7, atol=30)
        np.testing.assert_allclose(oreRV[1], v, rtol=3e-5, atol=3e-2)

        # Do these sparingly since astropy is kind of slow for generating EarthLocation RV
        if i < 5:
            # Make on observer to compare radec and altaz
            obsLon = np.random.uniform(0, 360)
            obsLat = np.random.uniform(-90, 90)
            obsEl = np.random.uniform(2000, 4000)
            observer = ssapy.EarthObserver(obsLon, obsLat, obsEl)

            # And similar for orekit
            oreEarth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                                        Constants.WGS84_EARTH_FLATTENING,
                                        FramesFactory.getITRF(IERSConventions.IERS_2010, True))
            oreFrame = TopocentricFrame(
                oreEarth,
                GeodeticPoint(
                    float(np.deg2rad(obsLat)),
                    float(np.deg2rad(obsLon)),
                    float(obsEl)
                ),
                ""
            )
            frameR = []
            frameV = []
            for t in ts:
                frv = oreFrame.getPVCoordinates(
                    AbsoluteDate(t.utc.isot, TimeScalesFactory.getUTC()),
                    J2000
                )
                frameR.append(vec3ToArr(frv.position))
                frameV.append(vec3ToArr(frv.velocity))
            frameR = np.array(frameR)
            frameV = np.array(frameV)
            ssaFrameR, ssaFrameV = observer.getRV(ts)
            np.testing.assert_allclose(frameR, ssaFrameR, rtol=0, atol=1)
            np.testing.assert_allclose(frameV, ssaFrameV, rtol=0, atol=1e-3)

            dr = oreRV[0] - frameR
            oreRa = np.arctan2(dr[...,1], dr[...,0])
            oreDec = np.arcsin(dr[...,2]/norm(dr))

            # Compare radec
            ra, dec, _ = ssapy.radec(orbit, ts, observer=observer)
            checkSphere(ra, dec, oreRa, oreDec, atol=1.0/206265)

            # Now check altaz
            alt, az = ssapy.altaz(orbit, ts, observer)
            oreAlt = []
            oreAz = []
            for t, oreRV1 in zip(ts, oreRV[0]):
                oreT = AbsoluteDate(t.utc.isot, TimeScalesFactory.getUTC())
                oreAlt.append(oreFrame.getElevation(
                    arrToVec3(oreRV1), J2000, oreT
                ))
                oreAz.append(oreFrame.getAzimuth(
                    arrToVec3(oreRV1), J2000, oreT
                ))
            oreAlt = np.array(oreAlt)
            oreAz = np.array(oreAz)
            checkSphere(az, alt, oreAz, oreAlt, atol=20.0/206265)


@timer
def test_earth_observer():
    # That code runs and gives the right shape output
    lon = 100.0 # deg
    lat = 45.0 # deg
    elevation = 300.0 # m
    observer = ssapy.EarthObserver(lon, lat, elevation)
    times = Time("J2018") + np.linspace(-40, 0, 1000)*u.year
    positions, velocities = observer.getRV(times)
    assert positions.shape == (1000, 3)
    assert velocities.shape == (1000, 3)

    # Check we get the same with float time
    positions2, velocities2 = observer.getRV(times.gps)
    np.testing.assert_allclose(positions, positions2, rtol=0, atol=1e-3)
    np.testing.assert_allclose(velocities, velocities2, rtol=0, atol=1e-3)

    # Check that fast observer rv works reasonably well
    fastObserver = ssapy.EarthObserver(lon, lat, elevation, fast=True)
    positions3, velocities3 = fastObserver.getRV(times)
    np.testing.assert_allclose(positions, positions3, rtol=0, atol=1.0)
    np.testing.assert_allclose(velocities, velocities3, rtol=0, atol=3e-4)

    # Check that light time correction make some difference
    orbit = ssapy.Orbit.fromKeplerianElements(1e8, 0.9, 0.5, 0.5, 0.5, 0.5, Time("J2018"))
    times = Time("J2018") + np.linspace(-12, 12, 400)*u.hour
    ra, dec, slantRange = ssapy.radec(orbit, times, observer=observer)
    ra2, dec2, slantRange2 = ssapy.radec(orbit, times, observer=observer, obsAngleCorrection="linear")
    assert not np.allclose(ra, ra2, rtol=0, atol=1e-5)
    assert not np.allclose(dec, dec2, rtol=0, atol=1e-5)
    assert not np.allclose(slantRange, slantRange2, rtol=0, atol=100)

    # We should get very similar results for linear and exact light time correction algorithms
    ra3, dec3, slantRange3 = ssapy.radec(orbit, times, observer=observer, obsAngleCorrection="exact")
    np.testing.assert_allclose(ra2, ra3, rtol=0, atol=1e-9)
    np.testing.assert_allclose(dec2, dec3, rtol=0, atol=1e-9)
    np.testing.assert_allclose(slantRange2, slantRange3, rtol=0, atol=1e-2)


@timer
def test_orbital_observer():
    # Just checking that it runs for now, and gives the right shape output
    orbit = ssapy.Orbit.fromKeplerianElements(1e8, 0.9, 0.5, 0.5, 0.5, 0.5, Time("J2000"))
    observer = ssapy.OrbitalObserver(orbit)

    times = Time("J2018") + np.linspace(-40, 0, 1000)*u.year
    positions, velocities = observer.getRV(times)
    assert positions.shape == (1000, 3)
    assert velocities.shape == (1000, 3)

    orbit = ssapy.Orbit.fromKeplerianElements(-5e9, 1.02, 0.5, 0.5, 0.5, 0.5, Time("J2000"))
    observer = ssapy.OrbitalObserver(orbit)

    times = Time("J2018") + np.linspace(-40, 0, 1000)*u.year
    positions, velocities = observer.getRV(times)
    assert positions.shape == (1000, 3)
    assert velocities.shape == (1000, 3)

    positions2, velocities2 = observer.getRV(times.gps)
    np.testing.assert_allclose(positions, positions2, rtol=0, atol=1e-6)
    np.testing.assert_allclose(velocities, velocities2, rtol=0, atol=1e-6)

@timer
def test_rv():
    np.random.seed(5772156)
    NORBIT = 30
    NTIME = 300

    orbits = []
    for _ in range(NORBIT):
        orbit = sample_GEO_orbit(t=Time("J2000"))
        orbits.append(orbit)

    for prop in [
        ssapy.KeplerianPropagator(), ssapy.SeriesPropagator(0),
        ssapy.SeriesPropagator(1), ssapy.SeriesPropagator(2)
    ]:
        print("testing propagator: ", prop)
        times = Time("J2000") + np.linspace(-2, 2, NTIME)*u.year
        r, v = ssapy.rv(orbits, times, propagator=prop)
        assert r.shape == v.shape == (NORBIT, NTIME, 3)

        r1, v1 = ssapy.rv(orbits, times[17], propagator=prop)
        assert r1.shape == v1.shape == (NORBIT, 3)
        np.testing.assert_allclose(r1, r[:,17,:], rtol=0, atol=r_atol)
        np.testing.assert_allclose(v1, v[:,17,:], rtol=0, atol=v_atol)

        # Try scalar orbit
        r2, v2 = ssapy.rv(orbits[11], times, propagator=prop)
        assert r2.shape == v2.shape == (NTIME, 3)
        np.testing.assert_allclose(r2, r[11], rtol=0, atol=r_atol)
        np.testing.assert_allclose(v2, v[11], rtol=0, atol=v_atol)

        # Try orbit and time both scalar
        r3, v3 = ssapy.rv(orbits[3], times[4], propagator=prop)
        assert r3.shape == v3.shape == (3,)
        np.testing.assert_allclose(r3, r[3, 4], rtol=0, atol=r_atol)
        np.testing.assert_allclose(v3, v[3, 4], rtol=0, atol=v_atol)

    # Check that for small dt, series propagation is reasonably accurate
    times = Time("J2000") + np.linspace(-2, 2, NTIME)*u.s
    r, v = ssapy.rv(orbits, times)
    r0, v0 = ssapy.rv(orbits, times, propagator=ssapy.SeriesPropagator(0))
    # velocities are ~km/s, so order=0 only good to 1e4 or so.
    # acceleration is small though, so good to 1e0.
    np.testing.assert_allclose(r, r0, rtol=0, atol=1e4)
    np.testing.assert_allclose(v, v0, rtol=0, atol=1e0)

    # Constant velocity, OTOH, should be pretty good over few seconds
    r1, v1 = ssapy.rv(orbits, times, propagator=ssapy.SeriesPropagator(1))
    np.testing.assert_allclose(r, r1, rtol=0, atol=1e0)
    np.testing.assert_allclose(v, v1, rtol=0, atol=1e0)

    # Constant acceleration should do even better
    r2, v2 = ssapy.rv(orbits, times, propagator=ssapy.SeriesPropagator(2))
    np.testing.assert_allclose(r, r2, rtol=0, atol=1e-4)
    np.testing.assert_allclose(v, v2, rtol=0, atol=1e-4)

    # Constant jerk does even better
    r2, v2 = ssapy.rv(orbits, times, propagator=ssapy.SeriesPropagator(3))
    np.testing.assert_allclose(r, r2, rtol=0, atol=2e-5)
    np.testing.assert_allclose(v, v2, rtol=0, atol=2e-5)

    # Constant acceleration should even work reasonably well over a few minutes.
    times = Time("J2000") + np.linspace(-2, 2, NTIME)*u.min
    r, v = ssapy.rv(orbits, times)
    r2, v2 = ssapy.rv(orbits, times, propagator=ssapy.SeriesPropagator(2))
    np.testing.assert_allclose(r, r2, rtol=0, atol=1e1)
    np.testing.assert_allclose(v, v2, rtol=0, atol=1e0)

    # Constant jerk does even better
    times = Time("J2000") + np.linspace(-2, 2, NTIME)*u.min
    r, v = ssapy.rv(orbits, times)
    r2, v2 = ssapy.rv(orbits, times, propagator=ssapy.SeriesPropagator(3))
    np.testing.assert_allclose(r, r2, rtol=0, atol=1e-1)
    np.testing.assert_allclose(v, v2, rtol=0, atol=1e-2)


@timer
# def test_groundTrack():
#     np.random.seed(5772156)
#     NORBIT = 30
#     NTIME = 300

#     orbits = []
#     for _ in range(NORBIT):
#         orbit = sample_GEO_orbit(t=Time("J2000"))
#         orbits.append(orbit)

#     for prop in [
#         ssapy.KeplerianPropagator(), ssapy.SeriesPropagator(0),
#         ssapy.SeriesPropagator(1), ssapy.SeriesPropagator(2)
#     ]:
#         print("testing propagator: ", prop)
#         times = Time("J2000") + np.linspace(-2, 2, NTIME)*u.year
#         lat, lon, h = ssapy.groundTrack(orbits, times, propagator=prop)
#         assert lat.shape == lon.shape == h.shape == (NORBIT, NTIME)

#         lat1, lon1, h1 = ssapy.groundTrack(orbits, times[17], propagator=prop)
#         assert lat1.shape == lon1.shape == h1.shape == (NORBIT,)
#         np.testing.assert_allclose(lat1, lat[:,17], rtol=0, atol=v_atol)
#         np.testing.assert_allclose(lon1, lon[:,17], rtol=0, atol=v_atol)
#         np.testing.assert_allclose(h1, h[:,17], rtol=0, atol=r_atol)

#         # Try scalar orbit
#         lat2, lon2, h2 = ssapy.groundTrack(orbits[11], times, propagator=prop)
#         assert lat2.shape == lon2.shape == h2.shape == (NTIME,)
#         np.testing.assert_allclose(lat2, lat[11], rtol=0, atol=r_atol)
#         np.testing.assert_allclose(lon2, lon[11], rtol=0, atol=v_atol)
#         np.testing.assert_allclose(h2, h[11], rtol=0, atol=v_atol)

#         # Try orbit and time both scalar
#         lat3, lon3, h3 = ssapy.groundTrack(orbits[3], times[4], propagator=prop)
#         assert isinstance(lat3, numbers.Real)
#         assert isinstance(lon3, numbers.Real)
#         assert isinstance(h3, numbers.Real)
#         np.testing.assert_allclose(lat3, lat[3, 4], rtol=0, atol=r_atol)
#         np.testing.assert_allclose(lon3, lon[3, 4], rtol=0, atol=r_atol)
#         np.testing.assert_allclose(h3, h[3, 4], rtol=0, atol=r_atol)

#     # One that I verified by hand
#     time = np.linspace(0.0, 3600*7.0, 2)
#     orbit = ssapy.Orbit.fromKeplerianElements(
#         a=ssapy.constants.WGS72_EARTH_RADIUS+500e3,
#         e=0.001,
#         i=np.deg2rad(50),
#         pa=1.0,
#         raan=1.0,
#         trueAnomaly=0.0,
#         t=0.0
#     )
#     r, v = ssapy.rv(orbit, time)
#     lon, lat, height = ssapy.groundTrack(orbit, time)

#     # Verify that at time=time[0], Spain corresponds to sat position, and at time=time[-1], Madagascar does.
#     spain = ssapy.EarthObserver(lon=-3.7, lat=40.4)
#     madagascar = ssapy.EarthObserver(lon=46.9, lat=-18.8)

#     np.testing.assert_allclose(
#         normed(spain.getRV(time[0])[0]),
#         normed(r[0]),
#         atol=0.15
#     )

#     np.testing.assert_allclose(
#         normed(madagascar.getRV(time[-1])[0]),
#         normed(r[-1]),
#         atol=0.15
#     )

@timer
def test_dircos():
    np.random.seed(57721566)
    NORBIT = 30
    NTIME = 300

    # Just checking broadcasting behavior for the moment
    orbits = []
    for _ in range(NORBIT):
        orbit = sample_GEO_orbit(t=Time("J2000"))
        orbits.append(orbit)

    times = Time("J2000") + np.linspace(-2, 2, NTIME)*u.year

    obsPos = np.random.uniform(-1e7, 1e7, size=(NTIME, 3))
    obsVel = np.random.uniform(-1e3, 1e3, size=(NTIME, 3))

    dc = ssapy.dircos(orbits, times, obsPos)
    np.testing.assert_allclose(norm(dc), 1)
    assert dc.shape == (NORBIT, NTIME, 3)
    # Light time correction
    dc2 = ssapy.dircos(orbits, times, obsPos, obsVel, obsAngleCorrection="linear")
    dc3 = ssapy.dircos(orbits, times, obsPos, obsVel, obsAngleCorrection="exact")
    np.testing.assert_allclose(dc2, dc3, rtol=0, atol=1e-7)

    # Check len(stuff) = 1 and scalar inputs
    dc2 = ssapy.dircos(orbits[0:1], times, obsPos)
    dc3 = ssapy.dircos(orbits[0], times, obsPos)
    assert dc2.shape == (1, NTIME, 3)
    assert dc3.shape == (NTIME, 3)
    np.testing.assert_allclose(dc[0], np.squeeze(dc2), rtol=0, atol=dc_atol)
    np.testing.assert_allclose(dc[0], dc3, rtol=0, atol=dc_atol)
    # Light time correction for len(stuff) = 1...
    dc2 = ssapy.dircos(orbits[0:1], times, obsPos, obsVel, obsAngleCorrection="linear")
    dc3 = ssapy.dircos(orbits[0:1], times, obsPos, obsVel, obsAngleCorrection="exact")
    np.testing.assert_allclose(dc2, dc3, rtol=0, atol=1e-7)

    dc2 = ssapy.dircos(orbits, times[0:1], obsPos)
    dc3 = ssapy.dircos(orbits, times[0], obsPos)
    assert dc2.shape == (NORBIT, NTIME, 3)  # len(obsPos) forces len(times) back up to NTIME
    assert dc3.shape == (NORBIT, NTIME, 3)  # scalar also gets broadcasted
    # whole array isn't equal because now we're reusing a single time, but can still check
    # first item
    np.testing.assert_allclose(dc[:,0], dc2[:,0], rtol=0, atol=dc_atol)
    np.testing.assert_allclose(dc2, dc3, rtol=0, atol=dc_atol)
    # check light time correction for len(stuff) = 1...
    dc2 = ssapy.dircos(orbits, times[0], obsPos, obsVel, obsAngleCorrection="linear")
    dc3 = ssapy.dircos(orbits, times[0], obsPos, obsVel, obsAngleCorrection="exact")
    np.testing.assert_allclose(dc2, dc3, rtol=0, atol=1e-7)

    dc2 = ssapy.dircos(orbits, times, obsPos[0:1])
    dc3 = ssapy.dircos(orbits, times, obsPos[0])
    assert dc2.shape == (NORBIT, NTIME, 3)
    assert dc3.shape == (NORBIT, NTIME, 3)
    np.testing.assert_allclose(dc[:,0], dc2[:,0], rtol=0, atol=dc_atol)
    np.testing.assert_allclose(dc2, dc3, rtol=0, atol=dc_atol)
    # check light time correction for len(stuff) = 1...
    dc2 = ssapy.dircos(orbits, times, obsPos[0], obsVel[0], obsAngleCorrection="linear")
    dc3 = ssapy.dircos(orbits, times, obsPos[0], obsVel[0], obsAngleCorrection="exact")
    np.testing.assert_allclose(dc2, dc3, rtol=0, atol=1e-7)

    dc2 = ssapy.dircos(orbits, times[0:1], obsPos[0:1])
    dc3 = ssapy.dircos(orbits, times[0], obsPos[0])
    assert dc2.shape == (NORBIT, 1, 3)
    assert dc3.shape == (NORBIT, 3)
    np.testing.assert_allclose(dc[:,0], np.squeeze(dc2[:,0]), rtol=0, atol=dc_atol)
    np.testing.assert_allclose(np.squeeze(dc2), dc3, rtol=0, atol=dc_atol)

    dc2 = ssapy.dircos(orbits, times[0:1], obsPos[0])
    dc3 = ssapy.dircos(orbits, times[0], obsPos[0:1])
    assert dc2.shape == (NORBIT, 1, 3)
    assert dc3.shape == (NORBIT, 1, 3)
    np.testing.assert_allclose(dc[:,0], np.squeeze(dc2[:,0]), rtol=0, atol=dc_atol)
    np.testing.assert_allclose(dc2, dc3, rtol=0, atol=dc_atol)

    # Now check various combinations supplying observers instead of obsPos
    observers = [ssapy.OrbitalObserver(o) for o in orbits[:5]]
    orbits = orbits[5:]

    with np.testing.assert_raises(ValueError):
        dc = ssapy.dircos(orbits, times, observer=observers)

    times = times[:5]
    dc = ssapy.dircos(orbits, times, observer=observers)
    np.testing.assert_allclose(norm(dc), 1)
    assert dc.shape == (25, 5, 3)

    dc2 = ssapy.dircos(orbits[0:1], times, observer=observers)
    dc3 = ssapy.dircos(orbits[0], times, observer=observers)
    assert dc2.shape == (1, 5, 3)
    assert dc3.shape == (5, 3)
    np.testing.assert_allclose(dc[0], np.squeeze(dc2), rtol=0, atol=dc_atol)
    np.testing.assert_allclose(dc[0], dc3, rtol=0, atol=dc_atol)

    dc2 = ssapy.dircos(orbits, times[0:1], observer=observers)
    dc3 = ssapy.dircos(orbits, times[0], observer=observers)
    assert dc2.shape == (25, 5, 3)  # len(observers) forces len(times) back up to 5
    assert dc3.shape == (25, 5, 3)  # scalar also gets broadcasted
    # whole array isn't equal because now we're reusing a single time, but can still check
    # first item
    np.testing.assert_allclose(dc[:,0], dc2[:,0], rtol=0, atol=dc_atol)
    np.testing.assert_allclose(dc2, dc3, rtol=0, atol=dc_atol)

    dc2 = ssapy.dircos(orbits, times, observer=observers[0:1])
    dc3 = ssapy.dircos(orbits, times, observer=observers[0])
    assert dc2.shape == (25, 5, 3)
    assert dc3.shape == (25, 5, 3)
    np.testing.assert_allclose(dc[:,0], dc2[:,0], rtol=0, atol=dc_atol)
    np.testing.assert_allclose(dc2, dc3, rtol=0, atol=dc_atol)

    dc2 = ssapy.dircos(orbits, times[0:1], observer=observers[0:1])
    dc3 = ssapy.dircos(orbits, times[0], observer=observers[0])
    assert dc2.shape == (25, 1, 3)
    assert dc3.shape == (25, 3)
    np.testing.assert_allclose(dc[:,0], np.squeeze(dc2[:,0]), rtol=0, atol=dc_atol)
    np.testing.assert_allclose(np.squeeze(dc2), dc3, rtol=0, atol=dc_atol)

    dc2 = ssapy.dircos(orbits, times[0:1], observer=observers[0])
    dc3 = ssapy.dircos(orbits, times[0], observer=observers[0:1])
    assert dc2.shape == (25, 1, 3)
    assert dc3.shape == (25, 1, 3)
    np.testing.assert_allclose(dc[:,0], np.squeeze(dc2[:,0]), rtol=0, atol=dc_atol)
    np.testing.assert_allclose(dc2, dc3, rtol=0, atol=dc_atol)


@timer
def test_radec():
    np.random.seed(577215664)
    NORBIT = 30
    NTIME = 300

    # Just checking broadcasting behavior for the moment
    orbits = []
    for _ in range(30):
        orbit = sample_GEO_orbit(t=Time("J2000"))
        orbits.append(orbit)

    times = Time("J2000") + np.linspace(-2, 2, NTIME)*u.year

    obsPos = np.random.uniform(-1e7, 1e7, size=(NTIME, 3))
    obsVel = np.random.uniform(-1e3, 1e3, size=(NTIME, 3))

    ra, dec, slantRange = ssapy.radec(orbits, times, obsPos)
    assert ra.shape == dec.shape == slantRange.shape == (30, NTIME)

    # Light time correction
    ra2, dec2, slantRange2 = ssapy.radec(orbits, times, obsPos, obsVel, obsAngleCorrection="linear")
    ra3, dec3, slantRange3 = ssapy.radec(orbits, times, obsPos, obsVel, obsAngleCorrection="exact")
    np.testing.assert_allclose(ra2, ra3, rtol=0, atol=1e-8)
    np.testing.assert_allclose(dec2, dec3, rtol=0, atol=1e-8)
    np.testing.assert_allclose(slantRange2, slantRange3, rtol=0, atol=1e-1)

    # Check len(stuff) = 1 and scalar inputs
    ra2, dec2, slantRange2 = ssapy.radec(orbits[0:1], times, obsPos)
    ra3, dec3, slantRange3 = ssapy.radec(orbits[0], times, obsPos)
    assert ra2.shape == dec2.shape == slantRange2.shape == (1, NTIME)
    assert ra3.shape == dec3.shape == slantRange3.shape == (NTIME,)
    checkAngle(ra[0], np.squeeze(ra2))
    checkAngle(dec[0], np.squeeze(dec2))
    checkAngle(ra[0], ra3)
    checkAngle(dec[0], dec3)
    # Not sure why this isn't better, but 100 nm reproducibility seems good enough...
    np.testing.assert_allclose(slantRange[0], np.squeeze(slantRange2), rtol=0, atol=1e-7)
    np.testing.assert_allclose(slantRange[0], slantRange3, rtol=0, atol=1e-7)

    ra2, dec2, slantRange2 = ssapy.radec(orbits, times[0:1], obsPos)
    ra3, dec3, slantRange3 = ssapy.radec(orbits, times[0], obsPos)
    assert ra2.shape == dec2.shape == slantRange2.shape == (30, NTIME)  # len(obsPos) forces len(times) back up to NTIME
    assert ra3.shape == dec3.shape == slantRange3.shape == (30, NTIME)  # scalar also gets broadcasted
    # whole array isn't equal because now we're reusing a single time, but can still check
    # first item
    checkAngle(ra[:,0], ra2[:,0])
    checkAngle(dec[:,0], dec2[:,0])
    checkAngle(ra2, ra3)
    checkAngle(dec2, dec3)
    np.testing.assert_equal(slantRange[:,0], slantRange2[:,0])
    np.testing.assert_equal(slantRange2, slantRange3)

    ra2, dec2, slantRange2 = ssapy.radec(orbits, times, obsPos[0:1])
    ra3, dec3, slantRange3 = ssapy.radec(orbits, times, obsPos[0])
    assert ra2.shape == dec2.shape == slantRange2.shape == (30, NTIME)
    assert ra3.shape == dec3.shape == slantRange3.shape == (30, NTIME)
    checkAngle(ra[:,0], ra2[:,0])
    checkAngle(dec[:,0], dec2[:,0])
    checkAngle(ra2, ra3)
    checkAngle(dec2, dec3)
    np.testing.assert_equal(slantRange[:,0], slantRange2[:,0])
    np.testing.assert_equal(slantRange2, slantRange3)

    ra2, dec2, slantRange2 = ssapy.radec(orbits, times[0:1], obsPos[0:1])
    ra3, dec3, slantRange3 = ssapy.radec(orbits, times[0], obsPos[0])
    assert ra2.shape == dec2.shape == slantRange2.shape == (30, 1)
    assert ra3.shape == dec3.shape == slantRange3.shape == (30,)
    checkAngle(ra[:,0], np.squeeze(ra2[:,0]))
    checkAngle(dec[:,0], np.squeeze(dec2[:,0]))
    checkAngle(np.squeeze(ra2), ra3)
    checkAngle(np.squeeze(ra2), ra3)
    np.testing.assert_allclose(slantRange[:,0], np.squeeze(slantRange2[:,0]), rtol=0, atol=r_atol)
    np.testing.assert_equal(np.squeeze(slantRange2), slantRange3)

    ra2, dec2, slantRange2 = ssapy.radec(orbits, times[0:1], obsPos[0])
    ra3, dec3, slantRange3 = ssapy.radec(orbits, times[0], obsPos[0:1])
    assert ra2.shape == dec2.shape == slantRange2.shape == (30, 1)
    assert ra3.shape == dec3.shape == slantRange3.shape == (30, 1)
    checkAngle(ra[:,0], np.squeeze(ra2[:,0]))
    checkAngle(dec[:,0], np.squeeze(dec2[:,0]))
    checkAngle(ra2, ra3)
    checkAngle(dec2, dec3)
    np.testing.assert_allclose(slantRange[:,0], np.squeeze(slantRange2[:,0]), rtol=0, atol=r_atol)
    np.testing.assert_equal(slantRange2, slantRange3)

    # Now check various combinations supplying observers instead of obsPos
    observers = [ssapy.OrbitalObserver(o) for o in orbits[:5]]
    orbits = orbits[5:]

    with np.testing.assert_raises(ValueError):
        ra, dec, slantRange = ssapy.radec(orbits, times, observer=observers)

    times = times[:5]
    ra, dec, slantRange = ssapy.radec(orbits, times, observer=observers)
    assert ra.shape == dec.shape == slantRange.shape == (25, 5)

    ra2, dec2, slantRange2 = ssapy.radec(orbits[0:1], times, observer=observers)
    ra3, dec3, slantRange3 = ssapy.radec(orbits[0], times, observer=observers)
    assert ra2.shape == dec2.shape == slantRange2.shape == (1, 5)
    assert ra3.shape == dec3.shape == slantRange3.shape == (5,)
    checkAngle(ra[0], np.squeeze(ra2))
    checkAngle(dec[0], np.squeeze(dec2))
    checkAngle(ra[0], ra3)
    checkAngle(dec[0], dec3)
    np.testing.assert_equal(slantRange[0], np.squeeze(slantRange2))
    np.testing.assert_equal(slantRange[0], slantRange3)

    ra2, dec2, slantRange2 = ssapy.radec(orbits, times[0:1], observer=observers)
    ra3, dec3, slantRange3 = ssapy.radec(orbits, times[0], observer=observers)
    assert ra2.shape == dec2.shape == slantRange2.shape == (25, 5)  # len(observers) forces len(times) back up to 5
    assert ra3.shape == dec3.shape == slantRange3.shape == (25, 5)  # scalar also gets broadcasted
    # whole array isn't equal because now we're reusing a single time, but can still check
    # first item
    checkAngle(ra[:,0], ra2[:,0])
    checkAngle(dec[:,0], dec2[:,0])
    checkAngle(ra2, ra3)
    checkAngle(dec2, dec3)
    np.testing.assert_equal(slantRange[:,0], slantRange2[:,0])
    np.testing.assert_equal(slantRange2, slantRange3)

    ra2, dec2, slantRange2 = ssapy.radec(orbits, times, observer=observers[0:1])
    ra3, dec3, slantRange3 = ssapy.radec(orbits, times, observer=observers[0])
    assert ra2.shape == dec2.shape == slantRange2.shape == (25, 5)
    assert ra3.shape == dec3.shape == slantRange3.shape == (25, 5)
    checkAngle(ra[:,0], ra2[:,0])
    checkAngle(dec[:,0], dec2[:,0])
    checkAngle(ra2, ra3)
    checkAngle(dec2, dec3)
    np.testing.assert_equal(slantRange[:,0], slantRange2[:,0])
    np.testing.assert_equal(slantRange2, slantRange3)

    ra2, dec2, slantRange2 = ssapy.radec(orbits, times[0:1], observer=observers[0:1])
    ra3, dec3, slantRange3 = ssapy.radec(orbits, times[0], observer=observers[0])
    assert ra2.shape == dec2.shape == slantRange2.shape == (25, 1)
    assert ra3.shape == dec3.shape == slantRange3.shape == (25,)
    checkAngle(ra[:,0], np.squeeze(ra2[:,0]))
    checkAngle(dec[:,0], np.squeeze(dec2[:,0]))
    checkAngle(np.squeeze(ra2), ra3)
    checkAngle(np.squeeze(dec2), dec3)
    np.testing.assert_equal(slantRange[:,0], np.squeeze(slantRange2[:,0]))
    np.testing.assert_equal(np.squeeze(slantRange2), slantRange3)

    ra2, dec2, slantRange2 = ssapy.radec(orbits, times[0:1], observer=observers[0])
    ra3, dec3, slantRange3 = ssapy.radec(orbits, times[0], observer=observers[0:1])
    assert ra2.shape == dec2.shape == slantRange2.shape == (25, 1)
    assert ra3.shape == dec3.shape == slantRange3.shape == (25, 1)
    checkAngle(ra[:,0], np.squeeze(ra2[:,0]))
    checkAngle(dec[:,0], np.squeeze(dec2[:,0]))
    checkAngle(ra2, ra3)
    checkAngle(dec2, dec3)
    np.testing.assert_equal(slantRange[:,0], np.squeeze(slantRange2[:,0]))
    np.testing.assert_equal(slantRange2, slantRange3)


@timer
def test_radecRate():
    np.random.seed(5772156649 % 2**32)
    NORBIT = 30
    NTIME = 300

    # Check broadcasting behavior
    orbits = []
    for _ in range(NORBIT):
        while True:
            # For sampling orbits, pick a distance between LEO and GEO
            orbit = sample_orbit(t=Time("J2000"), r_low=7e6, r_high=5e7, v_low=2.7e3, v_high=8.3e3)
            # Exclude orbit if it intersects earth or is unbound
            if norm(orbit.periapsis) < 6.4e6:
                continue
            if orbit.energy > 0:
                continue
            break

        orbits.append(orbit)

    times = Time("J2000") + np.linspace(-2, 2, NTIME)*u.year

    obsPos = np.random.uniform(-1e7, 1e7, size=(NTIME, 3))
    obsVel = np.random.uniform(-1e2, 1e2, size=(NTIME, 3))

    _, _, _, dradt, ddecdt, slantRangeRate = ssapy.radec(
        orbits, times, obsPos, obsVel, rate=True)
    assert dradt.shape == ddecdt.shape == slantRangeRate.shape == (NORBIT, NTIME)

    # Light time correction
    _, _, _, dradt2, ddecdt2, slantRangeRate2 = ssapy.radec(
        orbits, times, obsPos, obsVel, obsAngleCorrection="linear", rate=True)
    _, _, _, dradt3, ddecdt3, slantRangeRate3 = ssapy.radec(
        orbits, times, obsPos, obsVel, obsAngleCorrection="exact", rate=True)
    np.testing.assert_allclose(dradt2, dradt3, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ddecdt2, ddecdt3, rtol=0, atol=1e-6)
    np.testing.assert_allclose(slantRangeRate2, slantRangeRate3, rtol=0, atol=1)

    # Check len(stuff) = 1 and scalar inputs
    _, _, _, dradt2, ddecdt2, slantRangeRate2 = ssapy.radec(
        orbits[0:1], times, obsPos, obsVel, rate=True)
    _, _, _, dradt3, ddecdt3, slantRangeRate3 = ssapy.radec(
        orbits[0], times, obsPos, obsVel, rate=True)
    assert dradt2.shape == ddecdt2.shape == slantRangeRate2.shape == (1, NTIME)
    assert dradt3.shape == ddecdt3.shape == slantRangeRate3.shape == (NTIME,)
    checkAngle(dradt[0], np.squeeze(dradt2))
    checkAngle(ddecdt[0], np.squeeze(ddecdt2))
    checkAngle(dradt[0], dradt3)
    checkAngle(ddecdt[0], ddecdt3)
    # Not sure why this isn't better, but 1 nm reproducibility seems good enough...
    np.testing.assert_allclose(slantRangeRate[0], np.squeeze(slantRangeRate2), rtol=0, atol=1e-9)
    np.testing.assert_allclose(slantRangeRate[0], slantRangeRate3, rtol=0, atol=1e-9)

    _, _, _, dradt2, ddecdt2, slantRangeRate2 = ssapy.radec(
        orbits, times[0:1], obsPos, obsVel, rate=True)
    _, _, _, dradt3, ddecdt3, slantRangeRate3 = ssapy.radec(
        orbits, times[0], obsPos, obsVel, rate=True)
    assert dradt2.shape == ddecdt2.shape == slantRangeRate2.shape == (NORBIT, NTIME)  # len(obsPos) forces len(times) back up to NTIME
    assert dradt3.shape == ddecdt3.shape == slantRangeRate3.shape == (NORBIT, NTIME)  # scalar also gets broadcasted
    # whole array isn't equal because now we're reusing a single time, but can still check
    # first item
    checkAngle(dradt[:,0], dradt2[:,0])
    checkAngle(ddecdt[:,0], ddecdt2[:,0])
    checkAngle(dradt2, dradt3)
    checkAngle(ddecdt2, ddecdt3)
    np.testing.assert_equal(slantRangeRate[:,0], slantRangeRate2[:,0])
    np.testing.assert_equal(slantRangeRate2, slantRangeRate3)

    _, _, _, dradt2, ddecdt2, slantRangeRate2 = ssapy.radec(
        orbits, times, obsPos[0:1], obsVel[0:1], rate=True)
    _, _, _, dradt3, ddecdt3, slantRangeRate3 = ssapy.radec(
        orbits, times, obsPos[0], obsVel[0], rate=True)
    assert dradt2.shape == ddecdt2.shape == slantRangeRate2.shape == (NORBIT, NTIME)
    assert dradt3.shape == ddecdt3.shape == slantRangeRate3.shape == (NORBIT, NTIME)
    checkAngle(dradt[:,0], dradt2[:,0])
    checkAngle(ddecdt[:,0], ddecdt2[:,0])
    checkAngle(dradt2, dradt3)
    checkAngle(ddecdt2, ddecdt3)
    np.testing.assert_equal(slantRangeRate[:,0], slantRangeRate2[:,0])
    np.testing.assert_equal(slantRangeRate2, slantRangeRate3)

    _, _, _, dradt2, ddecdt2, slantRangeRate2 = ssapy.radec(
        orbits, times[0:1], obsPos[0:1], obsVel[0:1], rate=True)
    _, _, _, dradt3, ddecdt3, slantRangeRate3 = ssapy.radec(
        orbits, times[0], obsPos[0], obsVel[0], rate=True)
    assert dradt2.shape == ddecdt2.shape == slantRangeRate2.shape == (NORBIT, 1)
    assert dradt3.shape == ddecdt3.shape == slantRangeRate3.shape == (NORBIT,)
    checkAngle(dradt[:,0], np.squeeze(dradt2[:,0]))
    checkAngle(ddecdt[:,0], np.squeeze(ddecdt2[:,0]))
    checkAngle(np.squeeze(dradt2), dradt3)
    checkAngle(np.squeeze(dradt2), dradt3)
    np.testing.assert_allclose(slantRangeRate[:,0], np.squeeze(slantRangeRate2[:,0]), rtol=0, atol=1e-9)
    np.testing.assert_equal(np.squeeze(slantRangeRate2), slantRangeRate3)

    _, _, _, dradt2, ddecdt2, slantRangeRate2 = ssapy.radec(
        orbits, times[0:1], obsPos[0], obsVel[0], rate=True)
    _, _, _, dradt3, ddecdt3, slantRangeRate3 = ssapy.radec(
        orbits, times[0], obsPos[0:1], obsVel[0:1], rate=True)
    assert dradt2.shape == ddecdt2.shape == slantRangeRate2.shape == (NORBIT, 1)
    assert dradt3.shape == ddecdt3.shape == slantRangeRate3.shape == (NORBIT, 1)
    checkAngle(dradt[:,0], np.squeeze(dradt2[:,0]))
    checkAngle(ddecdt[:,0], np.squeeze(ddecdt2[:,0]))
    checkAngle(dradt2, dradt3)
    checkAngle(ddecdt2, ddecdt3)
    np.testing.assert_allclose(slantRangeRate[:,0], np.squeeze(slantRangeRate2[:,0]), rtol=0, atol=1e-9)
    np.testing.assert_equal(slantRangeRate2, slantRangeRate3)

    # Now check various combinations supplying observers instead of obsPos
    observers = [ssapy.OrbitalObserver(o) for o in orbits[:5]]
    orbits = orbits[5:]

    with np.testing.assert_raises(ValueError):
        _, _, _, dradt, ddecdt, slantRangeRate = ssapy.radec(
            orbits, times, observer=observers, rate=True)

    times = times[:5]
    _, _, _, dradt, ddecdt, slantRangeRate = ssapy.radec(
        orbits, times, observer=observers, rate=True)
    assert dradt.shape == ddecdt.shape == slantRangeRate.shape == (NORBIT-5, 5)

    _, _, _, dradt2, ddecdt2, slantRangeRate2 = ssapy.radec(
        orbits[0:1], times, observer=observers, rate=True)
    _, _, _, dradt3, ddecdt3, slantRangeRate3 = ssapy.radec(
        orbits[0], times, observer=observers, rate=True)
    assert dradt2.shape == ddecdt2.shape == slantRangeRate2.shape == (1, 5)
    assert dradt3.shape == ddecdt3.shape == slantRangeRate3.shape == (5,)
    checkAngle(dradt[0], np.squeeze(dradt2))
    checkAngle(ddecdt[0], np.squeeze(ddecdt2))
    checkAngle(dradt[0], dradt3)
    checkAngle(ddecdt[0], ddecdt3)
    np.testing.assert_equal(slantRangeRate[0], np.squeeze(slantRangeRate2))
    np.testing.assert_equal(slantRangeRate[0], slantRangeRate3)

    _, _, _, dradt2, ddecdt2, slantRangeRate2 = ssapy.radec(
        orbits, times[0:1], observer=observers, rate=True)
    _, _, _, dradt3, ddecdt3, slantRangeRate3 = ssapy.radec(
        orbits, times[0], observer=observers, rate=True)
    assert dradt2.shape == ddecdt2.shape == slantRangeRate2.shape == (NORBIT-5, 5)  # len(observers) forces len(times) back up to 5
    assert dradt3.shape == ddecdt3.shape == slantRangeRate3.shape == (NORBIT-5, 5)  # scalar also gets broadcasted
    # whole array isn't equal because now we're reusing a single time, but can still check
    # first item
    checkAngle(dradt[:,0], dradt2[:,0])
    checkAngle(ddecdt[:,0], ddecdt2[:,0])
    checkAngle(dradt2, dradt3)
    checkAngle(ddecdt2, ddecdt3)
    np.testing.assert_equal(slantRangeRate[:,0], slantRangeRate2[:,0])
    np.testing.assert_equal(slantRangeRate2, slantRangeRate3)

    _, _, _, dradt2, ddecdt2, slantRangeRate2 = ssapy.radec(
        orbits, times, observer=observers[0:1], rate=True)
    _, _, _, dradt3, ddecdt3, slantRangeRate3 = ssapy.radec(
        orbits, times, observer=observers[0], rate=True)
    assert dradt2.shape == ddecdt2.shape == slantRangeRate2.shape == (NORBIT-5, 5)
    assert dradt3.shape == ddecdt3.shape == slantRangeRate3.shape == (NORBIT-5, 5)
    checkAngle(dradt[:,0], dradt2[:,0])
    checkAngle(ddecdt[:,0], ddecdt2[:,0])
    checkAngle(dradt2, dradt3)
    checkAngle(ddecdt2, ddecdt3)
    np.testing.assert_equal(slantRangeRate[:,0], slantRangeRate2[:,0])
    np.testing.assert_equal(slantRangeRate2, slantRangeRate3)

    _, _, _, dradt2, ddecdt2, slantRangeRate2 = ssapy.radec(
        orbits, times[0:1], observer=observers[0:1], rate=True)
    _, _, _, dradt3, ddecdt3, slantRangeRate3 = ssapy.radec(
        orbits, times[0], observer=observers[0], rate=True)
    assert dradt2.shape == ddecdt2.shape == slantRangeRate2.shape == (NORBIT-5, 1)
    assert dradt3.shape == ddecdt3.shape == slantRangeRate3.shape == (NORBIT-5,)
    checkAngle(dradt[:,0], np.squeeze(dradt2[:,0]))
    checkAngle(ddecdt[:,0], np.squeeze(ddecdt2[:,0]))
    checkAngle(np.squeeze(dradt2), dradt3)
    checkAngle(np.squeeze(ddecdt2), ddecdt3)
    np.testing.assert_equal(slantRangeRate[:,0], np.squeeze(slantRangeRate2[:,0]))
    np.testing.assert_equal(np.squeeze(slantRangeRate2), slantRangeRate3)

    _, _, _, dradt2, ddecdt2, slantRangeRate2 = ssapy.radec(
        orbits, times[0:1], observer=observers[0], rate=True)
    _, _, _, dradt3, ddecdt3, slantRangeRate3 = ssapy.radec(
        orbits, times[0], observer=observers[0:1], rate=True)
    assert dradt2.shape == ddecdt2.shape == slantRangeRate2.shape == (NORBIT-5, 1)
    assert dradt3.shape == ddecdt3.shape == slantRangeRate3.shape == (NORBIT-5, 1)
    checkAngle(dradt[:,0], np.squeeze(dradt2[:,0]))
    checkAngle(ddecdt[:,0], np.squeeze(ddecdt2[:,0]))
    checkAngle(dradt2, dradt3)
    checkAngle(ddecdt2, ddecdt3)
    np.testing.assert_equal(slantRangeRate[:,0], np.squeeze(slantRangeRate2[:,0]))
    np.testing.assert_equal(slantRangeRate2, slantRangeRate3)

    # Check that rates match that for finite difference with small delta-t
    ra0, dec0, range0 = ssapy.radec(orbits, times, observer=observers)
    _, _, _, dradt, ddecdt, rangeRate0 = ssapy.radec(
        orbits, times, observer=observers, rate=True)
    dts = np.ones(len(times))*0.01
    ra1, dec1, range1 = ssapy.radec(orbits, times+dts*u.s, observer=observers)

    tol = np.deg2rad(0.005/3600)  # 5 mas per second
    checkAngle(np.cos(dec0)*(ra1-ra0)/dts, dradt, atol=tol)
    checkAngle((dec1-dec0)/dts, ddecdt, atol=tol)
    np.testing.assert_allclose((range1-range0)/dts, rangeRate0, rtol=0, atol=0.1)


@timer
def test_altaz():
    np.random.seed(57721566490 % 2**32)
    NORBIT = 30
    NTIME = 300

    # Just checking broadcasting behavior for the moment
    orbits = []
    for _ in range(NORBIT):
        orbit = sample_GEO_orbit(t=Time("J2000"))
        orbits.append(orbit)

    times = Time("J2000") + np.linspace(-2, 2, NTIME)*u.year

    lon = np.random.uniform(0, 2*np.pi)
    lat = np.random.uniform(-np.pi/2, np.pi/2)
    elevation = np.random.uniform(2000, 5000)
    observer = ssapy.EarthObserver(lon, lat, elevation)

    alt, az = ssapy.altaz(orbits, times, observer)

    # Light time correction
    alt2, az2 = ssapy.altaz(orbits, times, observer, obsAngleCorrection="linear")
    alt3, az3 = ssapy.altaz(orbits, times, observer, obsAngleCorrection="exact")
    np.testing.assert_allclose(alt2, alt3, rtol=0, atol=1e-7)
    np.testing.assert_allclose(az2, az3, rtol=0, atol=1e-7)

    assert alt.shape == az.shape == (NORBIT, NTIME)
    # Check len(stuff) = 1 and scalar inputs
    alt2, az2 = ssapy.altaz(orbits[0:1], times, observer)
    alt3, az3 = ssapy.altaz(orbits[0], times, observer)
    assert alt2.shape == az2.shape == (1, NTIME)
    assert alt3.shape == az3.shape == (NTIME,)
    checkAngle(alt[0], np.squeeze(alt2), atol=1e-10)
    checkAngle(az[0], np.squeeze(az2), atol=1e-10)
    checkAngle(alt[0], alt3, atol=1e-10)
    checkAngle(az[0], az3, atol=1e-10)

    alt2, az2 = ssapy.altaz(orbits, times[0:1], observer)
    alt3, az3 = ssapy.altaz(orbits, times[0], observer)
    assert alt2.shape == az2.shape == (NORBIT, 1)
    assert alt3.shape == az3.shape ==(NORBIT,)
    checkAngle(alt[:,0], alt2[:,0], atol=1e-10)
    checkAngle(az[:,0], az2[:,0], atol=1e-10)
    checkAngle(np.squeeze(alt2), alt3, atol=1e-10)
    checkAngle(np.squeeze(az2), az3, atol=1e-10)

    alt2, az2 = ssapy.altaz(orbits[0], times[0], observer)
    checkAngle(alt[0,0], alt2, atol=1e-10)
    checkAngle(az[0,0], az2, atol=1e-10)

    # Explicitly check vectorization
    for _ in range(10):
        i = np.random.choice(len(orbits))
        j = np.random.choice(len(times))
        alt2, az2 = ssapy.altaz(orbits[i], times[j], observer)
        checkAngle(alt[i, j], alt2, rtol=0, atol=1e-10)
        checkAngle(az[i, j], az2, rtol=0, atol=1e-10)


# Used in multiprocessing test
class f:
    def __init__(self, time):
        self.time = time
    def __call__(self, x):
        return ssapy.rv(x, self.time)


@timer
def test_multiprocessing():
    from multiprocessing import Pool
    np.random.seed(577215664901 % 2**32)
    NORBIT = 30
    NTIME = 300

    orbits = []
    for _ in range(NORBIT):
        orbit = sample_GEO_orbit(t=Time("J2000"))
        orbits.append(orbit)

    time = Time("J2000") + np.linspace(-2, 2, NTIME)*u.year
    ff = f(time)

    with Pool(4) as pool:
        rv1 = pool.map(ff, orbits)
    rv2 = ssapy.rv(orbits, time)

    np.testing.assert_allclose(
        np.transpose(rv1, (1,0,2,3)),
        rv2,
        rtol=0,
        atol=mp_atol
    )


@timer
# def test_kozai():
#     np.random.seed(5772156649015 % 2**32)
#     # Start with 100 ~GEO orbits
#     # import tqdm
#     # for _ in tqdm.tqdm(range(10_000)):
#     for _ in range(100):
#         # Random point near GEO sphere:
#         a = np.random.uniform(-1e3, 1e3) + ssapy.constants.RGEO
#         u = np.random.uniform(0, 2*np.pi)
#         v = np.random.uniform(-1, 1)
#         r = np.array([np.sqrt(1-v*v)*np.cos(u), np.sqrt(1-v*v)*np.sin(u), v])
#         # Orthogonal velocity of correct magnitude.
#         # Generate another point on the unit sphere then subtract component along r
#         u = np.random.uniform(0, 2*np.pi)
#         v = np.random.uniform(-1, 1)
#         n = np.array([np.sqrt(1-v*v)*np.cos(u), np.sqrt(1-v*v)*np.sin(u), v])
#         n -= np.dot(r, n) * r
#         r *= a
#         v = normed(n) * ssapy.constants.VGEO + np.random.uniform(-5, 5, size=3)

#         orbit = ssapy.Orbit(r, v, 0.0, mu=ssapy.constants.WGS72_EARTH_MU)
#         # Test round trip
#         elements = orbit.kozaiMeanKeplerianElements
#         newOrbit = ssapy.Orbit.fromKozaiMeanKeplerianElements(*elements, t=0.0)
#         np.testing.assert_allclose(orbit.r, newOrbit.r, rtol=0, atol=1e-6)
#         np.testing.assert_allclose(orbit.v, newOrbit.v, rtol=0, atol=1e-10)
#         # How far off are we over 1/3 period ?
#         r0, v0 = ssapy.rv(orbit, orbit.period/3)
#         r1, v1 = ssapy.rv(newOrbit, orbit.period/3)
#         np.testing.assert_allclose(r0, r1, rtol=1e-6, atol=1e-5)
#         np.testing.assert_allclose(v0, v1, rtol=1e-6, atol=1e-2)

#     # 100 ~LEO orbits
#     # import tqdm
#     # for i in tqdm.tqdm(range(10_000)):
#     for _ in range(100):
#         perigee = 0
#         while perigee < (ssapy.constants.WGS84_EARTH_RADIUS + 300e3):
#             # Random point near LEO sphere:
#             a = np.random.uniform(500e3, 1000e3) + ssapy.constants.WGS84_EARTH_RADIUS
#             u = np.random.uniform(0, 2*np.pi)
#             v = np.random.uniform(-1, 1)
#             r = np.array([np.sqrt(1-v*v)*np.cos(u), np.sqrt(1-v*v)*np.sin(u), v])
#             # Orthogonal velocity of correct magnitude.
#             # Generate another point on the unit sphere then subtract component along r
#             u = np.random.uniform(0, 2*np.pi)
#             v = np.random.uniform(-1, 1)
#             n = np.array([np.sqrt(1-v*v)*np.cos(u), np.sqrt(1-v*v)*np.sin(u), v])
#             n -= np.dot(r, n) * r
#             r *= a
#             v = normed(n) * ssapy.constants.VLEO + np.random.uniform(-5, 5, size=3)

#             orbit = ssapy.Orbit(r, v, 0.0, mu=ssapy.constants.WGS72_EARTH_MU)
#             perigee = norm(orbit.periapsis)
#         # Test round trip
#         elements = orbit.kozaiMeanKeplerianElements
#         newOrbit = ssapy.Orbit.fromKozaiMeanKeplerianElements(*elements, t=0.0)
#         np.testing.assert_allclose(orbit.r, newOrbit.r, rtol=0, atol=1e-6)
#         np.testing.assert_allclose(orbit.v, newOrbit.v, rtol=0, atol=1e-10)
#         # How far off are we over 1/3 period ?
#         r0, v0 = ssapy.rv(orbit, orbit.period/3)
#         r1, v1 = ssapy.rv(newOrbit, orbit.period/3)
#         np.testing.assert_allclose(r0, r1, rtol=0, atol=1e-5)
#         np.testing.assert_allclose(v0, v1, rtol=0, atol=1e-9)

#     # 100 less-constrained random orbits
#     # import tqdm
#     # for _ in tqdm.tqdm(range(10_000)):
#     for _ in range(100):
#         perigee = 0
#         energy = 1
#         eccentricity = 1
#         while (
#             perigee < (ssapy.constants.WGS84_EARTH_RADIUS + 300e3)
#             or energy > 0
#             or eccentricity > 0.8
#         ):
#             # Random radius between LEO, 2xGEO
#             a = np.random.uniform(
#                 ssapy.constants.WGS84_EARTH_RADIUS + 500e3,
#                 ssapy.constants.RGEO * 2
#             )
#             u = np.random.uniform(0, 2*np.pi)
#             v = np.random.uniform(-1, 1)
#             r = np.array([np.sqrt(1-v*v)*np.cos(u), np.sqrt(1-v*v)*np.sin(u), v])
#             r *= a
#             # randomish velocity
#             u = np.random.uniform(0, 2*np.pi)
#             v = np.random.uniform(-1, 1)
#             n = np.array([np.sqrt(1-v*v)*np.cos(u), np.sqrt(1-v*v)*np.sin(u), v])
#             v = n * np.random.uniform(3e3, 10e3, size=3)

#             orbit = ssapy.Orbit(r, v, 0.0, mu=ssapy.constants.WGS72_EARTH_MU)
#             perigee = norm(orbit.periapsis)
#             energy = orbit.energy
#             eccentricity = orbit.e
#         # Test round trip
#         elements = orbit.kozaiMeanKeplerianElements
#         newOrbit = ssapy.Orbit.fromKozaiMeanKeplerianElements(*elements, t=0.0)
#         np.testing.assert_allclose(orbit.r, newOrbit.r, rtol=0, atol=1e-6)
#         np.testing.assert_allclose(orbit.v, newOrbit.v, rtol=0, atol=1e-10)
#         # How far off are we over 1/3 period ?
#         r0, v0 = ssapy.rv(orbit, orbit.period/3)
#         r1, v1 = ssapy.rv(newOrbit, orbit.period/3)
#         np.testing.assert_allclose(r0, r1, rtol=0, atol=1e-5)
#         np.testing.assert_allclose(v0, v1, rtol=0, atol=1e-9)


@timer
def test_sgp4_vector():
    np.random.seed(57721566490153 % 2**32)
    for _ in range(10):
        while True:
            orbit = sample_GEO_orbit(t=0.0)
            # Skip orbit if perigee is close to Earth's surface
            if np.sqrt(np.sum(orbit.periapsis**2)) > 1e7:
                break

        times = np.random.uniform(-orbit.period, orbit.period, size=10)
        r, v = ssapy.rv(orbit, times, propagator=ssapy.SGP4Propagator())
        # Test vectorization consistency
        for i, t in enumerate(times):
            r_, v_ = ssapy.rv(orbit, t, propagator=ssapy.SGP4Propagator())
            np.testing.assert_allclose(r[i], r_, rtol=0, atol=1e-6)
            np.testing.assert_allclose(v[i], v_, rtol=0, atol=1e-9)

        # Compare against Kepler:
        rk, vk = ssapy.rv(orbit, times)
        np.testing.assert_allclose(r, rk, rtol=0, atol=3e5)  # 300 km
        np.testing.assert_allclose(v, vk, rtol=0, atol=20)  # 20 m/s


@timer
def test_sgp4():
    import os
    from sgp4.api import Satrec

    for i in range(1,3):
        tle_file = os.path.join(os.path.dirname(__file__), "data", f"aeroTLE_{i}.txt")
        with open(tle_file, 'r') as fd:
            line1, line2 = fd.readlines()
        orbit = ssapy.Orbit.fromTLETuple((line1, line2))
        r, v = ssapy.rv(orbit, orbit.t, propagator=ssapy.SGP4Propagator())
        np.testing.assert_allclose(
            orbit.r, r,
            rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            orbit.v, v,
            rtol=1e-6, atol=1e-4
        )
        # Check TLE production
        # We don't expect a perfect TLE -> orbit -> TLE roundtrip here, since we
        # don't use ballistic/drag coefficients for instance, and we makeup
        # nonsense NORAD IDs for the TLE, but we should be able to roundtrip
        # orbit -> TLE -> orbit.
        line1a, line2a = orbit.tle
        orbit2 = ssapy.Orbit.fromTLETuple((line1a, line2a))
        assert orbit == orbit2

        # Load Aero data to test against
        b3_file = os.path.join(os.path.dirname(__file__), "data", f"aeroB3_{i}.obs")
        b3 = ssapy.io.parseB3(b3_file)
        # Need to parse B3 time columns and rotate ra/dec from TEME->J2000
        times = []
        for obs in b3:
            year, day = obs['year'], obs['day']
            hour, minute, second = obs['hour'], obs['minute'], obs['second']
            timestr = f"{year}:{day:03d}:{hour:02d}:{minute:02d}:{second:06.3f}"
            times.append(Time(timestr))
        times = Time(times)

        rot = teme_to_gcrf(times[0])
        x = np.cos(np.deg2rad(b3['azimuthAngle'])) * np.cos(np.deg2rad(b3['polarAngle']))
        y = np.sin(np.deg2rad(b3['azimuthAngle'])) * np.cos(np.deg2rad(b3['polarAngle']))
        z = np.sin(np.deg2rad(b3['polarAngle']))
        x, y, z = np.dot(rot, [x, y, z])
        b3ra = np.arctan2(y, x)
        b3dec = np.arcsin(z, np.sqrt(x*x + y*y + z*z))

        # First test directly against the sgp4 python package
        sat = Satrec.twoline2rv(line1, line2)
        rsgp4, vsgp4 = [], []
        for t in times:
            e, r, v = sat.sgp4_tsince((t.gps-orbit.t)/60)
            rsgp4.append(r)
            vsgp4.append(v)
        rsgp4 = np.array(rsgp4)*1e3
        vsgp4 = np.array(vsgp4)*1e3
        rot = teme_to_gcrf(orbit.t)
        rsgp4 = np.dot(rot, rsgp4.T).T
        vsgp4 = np.dot(rot, vsgp4.T).T

        r, v = ssapy.rv(orbit, times, propagator=ssapy.SGP4Propagator(orbit.t, truncate=True))
        np.testing.assert_allclose(rsgp4, r, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(vsgp4, v, rtol=1e-15, atol=1e-15)

        observer = ssapy.EarthObserver(lon=253.635647, lat=33.739450, elevation=2380.0)
        ra, dec, _ = ssapy.radec(orbit, times, observer=observer, propagator=ssapy.SGP4Propagator())
        checkSphere(ra, dec, b3ra, b3dec, atol=np.deg2rad(20.0/3600), verbose=True)

        if has_orekit:
            from org.orekit.frames import FramesFactory, TopocentricFrame
            from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint
            from org.orekit.time import TimeScalesFactory, AbsoluteDate, DateComponents, TimeComponents
            from org.orekit.utils import IERSConventions, Constants
            from org.orekit.propagation.analytical.tle import TLE, TLEPropagator

            tle = TLE(line1, line2)
            ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
            earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                                     Constants.WGS84_EARTH_FLATTENING,
                                     ITRF)

            from math import radians
            longitude = radians(253.635647)
            latitude  = radians(33.73945)
            altitude  = 2380.0
            station = GeodeticPoint(latitude, longitude, altitude)
            station_frame = TopocentricFrame(earth, station, "SST")
            inertial_frame = FramesFactory.getEME2000()
            propagator = TLEPropagator.selectExtrapolator(tle)
            utc = TimeScalesFactory.getUTC()

            oreRa = []
            oreDec = []
            for t in times:
                oreT = AbsoluteDate(t.isot, utc)
                pv = propagator.getPVCoordinates(oreT, inertial_frame)
                pos = pv.getPosition()
                x, y, z = pos.getX(), pos.getY(), pos.getZ()
                pv = station_frame.getPVCoordinates(oreT, inertial_frame)
                rSta = pv.getPosition()
                rx, ry, rz = rSta.getX(), rSta.getY(), rSta.getZ()
                dx = x-rx
                dy = y-ry
                dz = z-rz
                oreRa.append(np.arctan2(dy, dx))
                oreDec.append(np.arcsin(dz/np.sqrt(dx**2+dy**2+dz**2)))
            oreRa = np.array(oreRa)
            oreDec = np.array(oreDec)
            checkSphere(ra, dec, oreRa, oreDec, atol=np.deg2rad(1.5/3600), verbose=True)


@timer
def test_light_time_correction():
    np.random.seed(57721)
    # Construct a light-time delay situation from the ground up, and then
    # try to reverse it using ssapy.dircos
    for _ in range(100):
        t_emit = 0.0
        t_detect = np.random.uniform(0, 1)
        dircos = normed(np.random.uniform(-1, 1, size=3))
        orbit = ssapy.Orbit.fromKeplerianElements(
            np.random.uniform(1e7, 1e8),   # a
            np.random.uniform(0.01, 0.9),  # e
            np.random.uniform(0.01, 1.4),  # i
            np.random.uniform(0.01, 0.1),
            np.random.uniform(0.01, 0.1),
            np.random.uniform(0.01, 0.1),
            np.random.uniform(-100, 100)
        )
        r_emit, _ = ssapy.rv(orbit, t_emit)
        r_detect, _ = ssapy.rv(orbit, t_detect)
        obsPos = r_emit + (-dircos)*299792458*(t_detect-t_emit)
        dc_None = ssapy.dircos(orbit, t_detect, obsPos=obsPos)
        dc_linear = ssapy.dircos(orbit, t_detect, obsPos=obsPos, obsVel=0, obsAngleCorrection="linear")
        dc_exact = ssapy.dircos(orbit, t_detect, obsPos=obsPos, obsVel=0, obsAngleCorrection="exact")
        np.testing.assert_allclose(dircos, dc_None, rtol=0, atol=1e-3)
        np.testing.assert_allclose(dircos, dc_linear, rtol=0, atol=1e-7)
        np.testing.assert_allclose(dircos, dc_exact, rtol=0, atol=1e-14)


@timer
def test_find_passes():
    # Just checking that things run.  No checking of values.

    rng = np.random.default_rng(1234)
    t0 = Time("2010-01-01T00:00:00")
    orbits = []
    for _ in range(4):
        orbits.append(
            ssapy.Orbit.fromKeplerianElements(
                ssapy.constants.WGS84_EARTH_RADIUS + rng.uniform(400e3, 1000e3),
                rng.uniform(0.001, 0.002),
                np.deg2rad(60.0 + rng.uniform(-5.0, 5.0)),
                rng.uniform(0.0, 2*np.pi),
                rng.uniform(0.0, 2*np.pi),
                rng.uniform(0.0, 2*np.pi),
                t0,
                propkw = {
                    'area':rng.uniform(1.0, 2.0),
                    'mass':rng.uniform(100.0, 500.0),
                    'CD':rng.uniform(1.8, 2.2),
                    'CR':rng.uniform(1.1, 1.5),
                }
            )
        )

    observers = []
    for _ in range(4):
        observers.append(ssapy.EarthObserver(
            lon=rng.uniform(0.0, 360.0),
            lat=rng.uniform(30.0, 70.0),
            elevation=rng.uniform(100.0, 3000.0),
            fast=True
        ))

    prop = ssapy.propagator.default_numerical()

    for orbit in orbits:
        passes = ssapy.compute.find_passes(
            orbit,
            observers,
            t0, 1*u.d, 5*u.min,
            propagator=prop,
            horizon=np.deg2rad(10.0)
        )
        for observer, times in passes.items():
            for time in times[:2]:  # no need to refine all...
                ssapy.compute.refine_pass(
                    orbit, observer, time,
                    propagator=prop,
                    horizon=np.deg2rad(10.0)
                )

    orbits = []
    for _ in range(3):
        orbits.append(
            ssapy.Orbit.fromKeplerianElements(
                ssapy.constants.RGEO,
                rng.uniform(0.001, 0.002),
                np.deg2rad(0.1),
                rng.uniform(0.0, 2*np.pi),
                rng.uniform(0.0, 2*np.pi),
                rng.uniform(0.0, 2*np.pi),
                t0,
                propkw = {
                    'area':rng.uniform(1.0, 2.0),
                    'mass':rng.uniform(100.0, 500.0),
                    'CD':rng.uniform(1.8, 2.2),
                    'CR':rng.uniform(1.1, 1.5),
                }
            )
        )

    for orbit in orbits:
        passes = ssapy.compute.find_passes(
            orbit,
            observers,
            t0, 1*u.d, 5*u.min,
            propagator=prop,
            horizon=np.deg2rad(18.0)
        )
        for observer, times in passes.items():
            for time in times:  # no need to refine all...
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    refinement = ssapy.compute.refine_pass(
                        orbit, observer, time,
                        propagator=prop,
                        horizon=np.deg2rad(18.0)
                    )
                # it's almost certain GEO sat won't rise or set in search window
                np.testing.assert_allclose(
                    refinement['duration'].value,
                    2880.0  # minutes in 2 days, the default search window
                )


def test_musun():
    # Travis & Nate discovered mu wasn't always being propagated.  Here's an previously
    # failing example.

    a = u.AU.to(u.m)
    mu = ssapy.constants.SUN_MU
    e = 0.001
    i = 0.001
    pa = 0.001
    raan = 0.001
    trueAnomaly = 0.001
    t = Time("J2000")
    orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, t, mu=mu)

    r, v = ssapy.rv(orbit, orbit.t + orbit.period)
    np.testing.assert_allclose(orbit.r, r)
    np.testing.assert_allclose(orbit.v, v)


def test_MG_2_3():
    """Exercise 2.3 from Montenbruck and Gill
    Tests conversion of position and velocity vectors to Keplerian elements
    """
    # Provided position and velocity vectors
    r_ref = np.array([10000.0, 40000.0, -5000.0]) * 1000  # [m]
    v_ref = np.array([-1.5, 1.0, -0.1]) * 1000  # [m/s]

    # Instantiate Orbit object with position and velocity
    t = Time("2020-01-01 00:00:00.000000")
    orbit = ssapy.Orbit(r_ref, v_ref, t=t)

    # Compute Keplerian elements for this orbit
    orbit._setKeplerian()

    # Reference element values provided in Montenbruck and Gill
    a_ref = 25015.181 * 1000  # semi-major axis [m]
    e_ref = 0.7079772  # eccentricity
    i_ref = np.deg2rad(6.971)  # inclination
    raan_ref = np.deg2rad(173.290)  # right ascension of ascending node
    pa_ref = np.deg2rad(91.553)  # argument of perigee
    meanAnomaly_ref = np.deg2rad(144.225)  # mean anomaly

    # Check that reference elements and computed elements are close
    np.testing.assert_allclose(
        a_ref, orbit.a, atol=1e-5, err_msg="Semi-major axis test failed"
    )
    np.testing.assert_allclose(
        e_ref, orbit.e, atol=1e-5, err_msg="Eccentricity test failed"
    )
    np.testing.assert_allclose(
        i_ref, orbit.i, atol=1e-5, err_msg="Inclination test failed"
    )
    np.testing.assert_allclose(
        raan_ref,
        orbit.raan,
        atol=1e-5,
        err_msg="Right ascension of ascending node test failed",
    )
    np.testing.assert_allclose(
        pa_ref, orbit.pa, atol=1e-5, err_msg="Argument of perigee test failed"
    )
    np.testing.assert_allclose(
        meanAnomaly_ref,
        orbit.meanAnomaly,
        atol=1e-5,
        err_msg="Mean anomaly test failed",
    )


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--prof", action='store_true')
    parser.add_argument("--prof_out", type=str, default=None)
    parser.add_argument("--prof_png", type=str, default=None)
    args = parser.parse_args()

    if args.prof:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

    test_anomaly_conversion()
    test_longitude_conversion()
    test_orbit_ctor()
    test_orbit_hyper_ctor()
    test_orbit_rv()
    if has_orekit:
        test_orekit()
    test_earth_observer()
    test_orbital_observer()
    test_rv()
    test_groundTrack()
    test_dircos()
    test_radec()
    test_radecRate()
    test_altaz()
    test_multiprocessing()
    test_kozai()
    test_sgp4_vector()
    test_sgp4()
    test_light_time_correction()
    test_find_passes()
    test_musun()

    if args.prof:
        import pstats
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('cumtime')
        ps.print_stats(30)
        ps = pstats.Stats(pr).sort_stats('tottime')
        ps.print_stats(30)
        if args.prof_out:
            pr.dump_stats(args.prof_out)
            if args.prof_png:
                import subprocess
                cmd = "gprof2dot -f pstats {} -n1 -e1 | dot -Tpng -o {}".format(args.prof_out, args.prof_png)
                subprocess.run(cmd, shell=True)
    
