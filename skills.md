# SSAPy Skill Guide for AI Agents

This guide helps AI coding agents use or modify LLNL SSAPy when the package is
installed from PyPI or cloned from GitHub.

## Package Identity

- PyPI package: `llnl-ssapy`
- Import package: `ssapy`
- GitHub repository: `https://github.com/LLNL/SSAPy`
- Documentation: `https://software.llnl.gov/SSAPy/` or `https://LLNL.github.io/SSAPy/`
- Companion toolkit: `ssapy-toolkit`, imported as `ssapy_toolkit`
- Companion data package: `llnl-ssapy-data`, imported as `ssapy_data`
- Minimum Python version: Python 3.8

## Install and Import

Install from PyPI for normal use:

```bash
python -m pip install llnl-ssapy
```

Install from a clone for development:

```bash
git clone https://github.com/LLNL/SSAPy.git
cd SSAPy
python -m pip install -e .[dev,docs]
```

Basic import check:

```python
import ssapy

print(ssapy.__version__)
print(ssapy.datadir)
```

## Package Role

SSAPy is the base astrodynamics and orbit-determination engine. Prefer SSAPy
for core orbit models, propagators, accelerations, observers, TLE/SGP4 support,
coordinate transforms, sampling, and track association.

Use `ssapy-toolkit` for higher-level workflows, plotting dashboards, convenience
wrappers, brightness calculations, and reusable analysis utilities. New reusable
datasets should generally live in `ssapy-data` rather than this repository.

SSAPy builds a compiled extension, `ssapy._ssapy`, through `scikit-build` and
`CMakeLists.txt`. If an import fails immediately after cloning, install the
package in editable mode rather than importing from an unbuilt source tree.

## Common API Entry Points

Top-level imports are intentionally convenient and should not be casually
removed or lazy-loaded. Useful public objects include:

```python
from ssapy import (
    Orbit,
    EarthObserver,
    KeplerianPropagator,
    SGP4Propagator,
    RK4Propagator,
    RK78Propagator,
    AccelKepler,
    AccelSum,
    AccelHarmonic,
    AccelThirdBody,
    rv,
    radec,
    altaz,
    groundTrack,
)
```

Minimal propagation example:

```python
import numpy as np
from ssapy import Orbit, KeplerianPropagator, rv
from ssapy.constants import RGEO

orbit = Orbit.fromKeplerianElements(
    a=RGEO,
    e=0.0,
    i=np.deg2rad(5.0),
    pa=0.0,
    raan=0.0,
    trueAnomaly=0.0,
    t=0.0,
)

times = orbit.t + np.linspace(0.0, 3600.0, 10)
r, v = rv(orbit, times, propagator=KeplerianPropagator())
```

## Module Map

- `ssapy.orbit`: `Orbit`, `EarthObserver`, `OrbitalObserver`, TLE constructors, and orbital element conversions.
- `ssapy.propagator`: Keplerian, SGP4, SciPy ODE, Runge-Kutta, and leapfrog propagators.
- `ssapy.accel`: Keplerian, drag, solar radiation pressure, Earth radiation pressure, and constant NTW accelerations.
- `ssapy.gravity`: harmonic gravity coefficients plus harmonic and third-body acceleration models.
- `ssapy.compute`: position, velocity, RA/Dec, Alt/Az, pass, ground-track, and observer conversion utilities.
- `ssapy.body`: Earth, Moon, Sun, and planetary position/orientation helpers.
- `ssapy.io`: TLE parsing, B3 observation parsing, telescope position helpers, and TLE generation.
- `ssapy.rvsampler`: initializers, priors, probability models, MCMC samplers, and least-squares optimizers.
- `ssapy.particles`: importance-resampling support for orbit model particles.
- `ssapy.correlate_tracks`: track fitting, priors, hypotheses, and multiple-hypothesis tracking utilities.
- `ssapy.orbit_solver`: Gauss, Danchick, Shefer, and three-angle orbit solvers.
- `ssapy.utils`: coordinate math, rotations, zero finding, interpolation, sampling helpers, and wrappers.
- `ssapy.plotUtils`: low-level Earth/Moon plotting helpers.
- `ssapy.tle_drag`: TLE drag fitting helpers.

## Data and Binary Rules

SSAPy currently includes required package data under `ssapy/data`, and that data
is still part of the base package. Do not migrate or delete it unless the task
explicitly calls for a coordinated data migration.

Use `ssapy.datadir` or `ssapy.utils.find_file()` when reading existing bundled
resources. Some model loaders already resolve packaged gravity, SPICE, and Earth
orientation files internally; prefer those public loaders over hard-coded paths.

Avoid adding new large datasets, generated images, movies, notebooks with
embedded outputs, archives, local shared libraries, or binary payloads to this
repository. If new reusable data are needed, prefer adding them to SSAPy-Data
and consuming them through a released `llnl-ssapy-data` wheel.

Packaging should not include local compiled artifacts such as `_ssapy*.so` from
developer builds. Generated build outputs belong outside the repository or in
ignored directories.

## Development Workflow

Before changing behavior, search for an existing implementation:

```bash
rg "def function_name|class ClassName|keyword" ssapy tests docs
```

Keep changes focused and scientific-behavior-preserving unless the task asks for
a model change. When changing orbital mechanics behavior, add or update a
regression test that checks the numerical behavior directly.

When two modules duplicate small helpers, prefer a narrow shared utility over a
new broad dependency. Avoid import-time side effects in core modules because
SSAPy exposes many modules through top-level convenience imports.

Use the existing top-level import style in `ssapy/__init__.py` for user-facing
convenience. Do not remove common top-level imports only to reduce import time
unless that has been explicitly requested and reviewed.

For package metadata changes, keep versions synchronized across
`pyproject.toml`, `ssapy/__init__.py`, `CMakeLists.txt`, `CITATION.cff`, and the
documentation configuration.

## Validation Commands

Run focused tests first, then broader checks as needed:

```bash
python -m pytest -q tests/test_compute.py
python -m pytest -q tests/test_accel.py tests/test_tle_drag.py tests/test_utils.py
python -m pytest -q --durations=20
git diff --check
```

Check Python syntax for changed package and test files:

```bash
python -m py_compile $(git ls-files 'ssapy/*.py' 'tests/*.py') setup.py
```

Build-check docs and release metadata when docs, packaging, or README content
changes:

```bash
python -m sphinx -b html -W docs/source /tmp/ssapy-docs-build
python -m build --sdist
python -m twine check dist/*
```

## Common Pitfalls

- Do not assume all packaged resources are small; base SSAPy still carries existing required data.
- Do not add Git LFS as a default solution for new data in this repository.
- Do not commit generated `_ssapy*.so`, `build/`, `dist/`, egg-info, docs build output, caches, or local environments.
- Do not rely on unbuilt in-tree imports for code paths that require `ssapy._ssapy`.
- Do not mutate input `Orbit` objects in fitting utilities unless the API explicitly promises mutation.
- Do not reload `ssapy` modules in-process for import-variant tests; use subprocesses for order-independent import checks.
- Do not change scientific constants, reference frames, or SGP4/TEME handling without targeted numerical tests.
