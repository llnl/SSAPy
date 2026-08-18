"""
Classes representing celestial bodies.
"""

import erfa
import numpy as np
from .utils import _gpsToTT, iers_interp as _iers_interp
from .constants import EARTH_MU, EARTH_RADIUS, MOON_MU, SUN_MU, MERCURY_MU, VENUS_MU, MARS_MU, JUPITER_MU, SATURN_MU, URANUS_MU, NEPTUNE_MU, MERCURY_RADIUS, VENUS_RADIUS, MARS_RADIUS, JUPITER_RADIUS, SATURN_RADIUS, URANUS_RADIUS, NEPTUNE_RADIUS
from .gravity import HarmonicCoefficients as _HarmonicCoefficients


def _close_if_possible(obj):
    close = getattr(obj, "close", None)
    if close is not None:
        close()


class _KernelBacked:
    def close(self):
        kernel = getattr(self, "kernel", None)
        if kernel is not None:
            _close_if_possible(kernel)
            self.kernel = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class EarthOrientation:
    """Orientation of earth in GCRF.  This is a callable class that returns the
    orientation matrix at a given time.

    Parameters
    ----------
    recalc_threshold : float
        Threshold for recomputing the orientation matrix. Default is 30 days.

    """
    def __init__(self, recalc_threshold=86400 * 30):
        self.recalc_threshold = recalc_threshold
        self._t = None

    def __call__(self, t, _E=None):
        """Return the orientation matrix at time t.

        Parameters
        ----------
        t : float
            Time in GPS seconds.

        Returns
        -------
        E : `numpy.ndarray`
            Orientation matrix at time t.
        """
        if _E is None:
            mjd_tt = _gpsToTT(t)
            if self._t is None or np.abs(t - self._t) > self.recalc_threshold:
                self._t = t
                self._dut1, _, _ = _iers_interp(t)
                self._T = erfa.pnm80(2400000.5, mjd_tt)
            gst = erfa.gst94(2400000.5, mjd_tt + self._dut1)
            _E = erfa.rxr(erfa.rv2m([0, 0, gst]), self._T)
        return _E


class MoonOrientation(_KernelBacked):
    """Orientation of moon in GCRF.  This is a callable class that returns the
    orientation matrix at a given time.
    """
    def __init__(self):
        import os
        from jplephem.pck import PCK
        from . import datadir

        fn = os.path.join(datadir, "moon_pa_de440_200625.bpc")
        self.kernel = PCK.open(fn)

    def __call__(self, t, _E=None):
        """Return the orientation matrix at time t.

        Parameters
        ----------
        t : float
            Time in GPS seconds.

        Returns
        -------
        E : `numpy.ndarray`
            Orientation matrix at time t.
        """
        mjd_tt = _gpsToTT(t)
        value, _ = self.kernel.segments[0].compute(2400000.5, mjd_tt)

        return (self._Rz(-value[0]) @ self._Rx(-value[1]) @ self._Rz(-value[2])).T

    @staticmethod
    def _Rx(alpha):
        ca, sa = np.cos(alpha), np.sin(alpha)
        out = np.eye(3)
        out[1, 1] = ca
        out[1, 2] = sa
        out[2, 1] = -sa
        out[2, 2] = ca
        return out

    @staticmethod
    def _Rz(alpha):
        ca, sa = np.cos(alpha), np.sin(alpha)
        out = np.eye(3)
        out[0, 0] = ca
        out[0, 1] = sa
        out[1, 0] = -sa
        out[1, 1] = ca
        return out


class MoonPosition(_KernelBacked):
    """Position of moon in GCRF.  This is a callable class that returns the
    position vector at a given time.
    """
    def __init__(self):
        import os
        from jplephem.spk import SPK
        from . import datadir

        fn = os.path.join(datadir, "de430.bsp")  # https://naif.jpl.nasa.gov/pub/naif/LUCY/kernels/spk/de430s.bsp.lbl
        self.kernel = SPK.open(fn)

    def __call__(self, t):
        """Return the position vector at time t.

        Parameters
        ----------
        t : float
            Time in GPS seconds.

        Returns
        -------
        pos : `numpy.ndarray`
            Position vector at time t in meters.
        """
        mjd_tt = _gpsToTT(t)
        pos = self.kernel[3, 301].compute(2400000.5, mjd_tt)  # Earth-moon barycenter -> moon
        pos -= self.kernel[3, 399].compute(2400000.5, mjd_tt)  # Earth-moon barycenter -> earth
        return pos * 1e3


class SunPosition(_KernelBacked):
    """Position of sun in GCRF.  This is a callable class that returns the
    position vector at a given time.
    """
    def __init__(self):
        import os
        from jplephem.spk import SPK
        from . import datadir

        fn = os.path.join(datadir, "de430.bsp")  # https://naif.jpl.nasa.gov/pub/naif/LUCY/kernels/spk/de430s.bsp.lbl
        self.kernel = SPK.open(fn)

    def __call__(self, t):
        """Return the position vector at time t.

        Parameters
        ----------
        t : float
            Time in GPS seconds.

        Returns
        -------
        pos : `numpy.ndarray`
            Position vector at time t in meters.
        """
        mjd_tt = _gpsToTT(t)

        pos = self.kernel[0, 10].compute(2400000.5, mjd_tt)  # SS bary -> sun
        pos -= self.kernel[0, 3].compute(2400000.5, mjd_tt)  # SS bary -> Earth-moon bary
        pos -= self.kernel[3, 399].compute(2400000.5, mjd_tt)  # Earth-moon bary -> Earth
        return pos * 1e3


class PlanetPosition(_KernelBacked):
    """Position of a planet in GCRF.  This is a callable class that returns the
    position vector at a given time.
    """
    def __init__(self, planet_index):
        import os
        from jplephem.spk import SPK
        from . import datadir

        fn = os.path.join(datadir, "de430.bsp")
        self.kernel = SPK.open(fn)
        self.planet_index = planet_index

    def __call__(self, t):
        """Return the position vector at time t.

        Parameters
        ----------
        t : float
            Time in GPS seconds.

        Returns
        -------
        pos : `numpy.ndarray`
            Position vector at time t in meters.
        """
        mjd_tt = _gpsToTT(t)
        pos = self.kernel[0, self.planet_index].compute(2400000.5, mjd_tt)  # SS bary -> Jupiter
        pos -= self.kernel[0, 3].compute(2400000.5, mjd_tt)  # SS bary -> Earth-moon bary
        pos -= self.kernel[3, 399].compute(2400000.5, mjd_tt)  # Earth-moon bary -> Earth
        return pos * 1e3


class Body:
    """A celestial body.

    Parameters
    ----------
    mu : `float`
        Gravitational parameter of the body in m^3/s^2.
    radius : `float`
        Radius of the body in meters.
    position : callable, optional
        A callable that returns the position vector of the body in GCRF at a
        given time.  [default: zero vector]
    orientation : callable, optional
        A callable that returns the orientation matrix of the body in GCRF at a
        given time.  [default: identity matrix]
    harmonics : `HarmonicCoefficients`, optional
        Harmonic coefficients for the body.  [default: None]
    """
    def __init__(
        self,
        mu,
        radius,
        position=lambda t: np.zeros(3),
        orientation=lambda t: np.eye(3),
        harmonics=None
    ):
        self.mu = mu
        self.radius = radius
        self.position = position
        self.orientation = orientation
        self.harmonics = harmonics

    def close(self):
        _close_if_possible(self.position)
        if self.orientation is not self.position:
            _close_if_possible(self.orientation)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def get_body(name, model=None):
    """
    Get a Body object for a named body.

    Parameters
    ----------
    name : str
        Name of the body. Must be one of "earth", "moon", "sun", or other supported planets.
    model : str, optional only available for Earth
        Name of the Earth harmonic model to use. Default is EGM84. options: EGM96, EGM2008.

    Returns
    -------
    body : Body
        Body object for the named body.
    """
    if name.lower() == "earth":
        if model is None:
            return Body(
                mu=EARTH_MU,
                radius=EARTH_RADIUS,
                orientation=EarthOrientation(),
                harmonics=_HarmonicCoefficients.fromEGM("EGM84")
            )
        else:
            if model == "1984" or model == "84":
                model = "EGM84"
            elif model == "1996" or model == "96":
                model = "EGM96"
            elif model == "2008" or model == "08":
                model = "EGM2008"
            return Body(
                mu=EARTH_MU,
                radius=EARTH_RADIUS,
                orientation=EarthOrientation(),
                harmonics=_HarmonicCoefficients.fromEGM(model)
            )
    elif name.lower() == "moon":
        return Body(
            mu=MOON_MU,
            radius=1737.4e3,
            position=MoonPosition(),
            orientation=MoonOrientation(),
            harmonics=_HarmonicCoefficients.fromTAB("gggrx_1200a_sha.tab")
        )
    elif name.lower() == "sun":
        return Body(
            mu=SUN_MU,
            radius=695700000.0,
            position=SunPosition(),
        )
    elif name.lower() == "mercury":
        return Body(
            mu=MERCURY_MU,
            radius=MERCURY_RADIUS,
            position=PlanetPosition(planet_index=1),
        )
    elif name.lower() == "venus":
        return Body(
            mu=VENUS_MU,
            radius=VENUS_RADIUS,
            position=PlanetPosition(planet_index=2),
        )
    elif name.lower() == "mars":
        return Body(
            mu=MARS_MU,
            radius=MARS_RADIUS,
            position=PlanetPosition(planet_index=4),
        )
    elif name.lower() == "jupiter":
        return Body(
            mu=JUPITER_MU,
            radius=JUPITER_RADIUS,
            position=PlanetPosition(planet_index=5),
        )
    elif name.lower() == "saturn":
        return Body(
            mu=SATURN_MU,
            radius=SATURN_RADIUS,
            position=PlanetPosition(planet_index=6),
        )
    elif name.lower() == "uranus":
        return Body(
            mu=URANUS_MU,
            radius=URANUS_RADIUS,
            position=PlanetPosition(planet_index=7),
        )
    elif name.lower() == "neptune":
        return Body(
            mu=NEPTUNE_MU,
            radius=NEPTUNE_RADIUS,
            position=PlanetPosition(planet_index=8),
        )
    else:
        raise ValueError(f"Unknown body {name}")
