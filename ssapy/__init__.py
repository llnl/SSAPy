import os

__version__ = "1.1.10"



def _datadir():
    """Directory holding SSAPy's data files: ssapy/ in llnl-ssapy-data."""
    try:
        from importlib.resources import files
        return os.fspath(files("ssapy_data") / "data" / "ssapy")
    except (ImportError, TypeError):
        # llnl-ssapy-data is a required dependency; without it find_file
        # raises a FileNotFoundError that says how to install it.
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
