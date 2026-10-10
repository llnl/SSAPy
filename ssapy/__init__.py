import os

__version__ = "1.1.12"



def _datadir():
    """Best-effort directory for installed split SSATK data packages."""
    try:
        from importlib.resources import files
        for package in ("ssapy_data_core", "ssapy_data_gravity", "ssapy_data_lunar", "ssapy_data_lunar_gravity"):
            try:
                return os.fspath(files(package) / "data")
            except (ImportError, TypeError):
                continue
    except (ImportError, TypeError):
        # Split data packages are optional at import time; find_file reports
        # the missing package when a resource is requested.
        return os.path.join(os.path.dirname(__file__), "data")


datadir = _datadir()

from . import _ssapy
from . import ephemeris  # noqa: F401  (ssapy.ephemeris.set_planetary_ephemeris)
from .orbit import Orbit, EarthObserver, OrbitalObserver
from .propagator import (
    KeplerianPropagator, SeriesPropagator, RK4Propagator, SGP4Propagator,
    SciPyPropagator, RK78Propagator, RK8Propagator, 
    LeapfrogPropagator, Leapfrog4Propagator
)
from .compute import rv, dircos, radec, altaz, quickAltAz, radecRate, groundTrack
from .accel import Accel, AccelKepler, AccelSum, AccelEarthRad, AccelSolRad, AccelDrag, AccelConstNTW
from .linker import ModelSelectorParams, BinarySelectorParams, Linker
from .orbit_solver import TwoPosOrbitSolver, GaussTwoPosOrbitSolver
from .orbit_solver import DanchickTwoPosOrbitSolver, SheferTwoPosOrbitSolver
from .orbit_solver import ThreeAngleOrbitSolver
from .particles import Particles
from .rvsampler import GEOProjectionInitializer, DistanceProjectionInitializer
from .rvsampler import circular_guess
from .rvsampler import GaussianRVInitializer, DirectInitializer
from .rvsampler import RVProbability, EmceeSampler, MVNormalProposal
from .rvsampler import RVSigmaProposal, MHSampler, LMOptimizer
from .ellipsoid import Ellipsoid
from .body import (
    EarthOrientation, MoonOrientation, MoonPosition, Body, get_body
)
from .gravity import HarmonicCoefficients, AccelThirdBody, AccelHarmonic

from . import constants
from . import io
from . import utils

from astropy.time import Time, TimeDelta
import astropy.units as u
from datetime import timedelta
