==============================================
SSAPy - Space Situational Awareness for Python
==============================================

|ci_badge| |docs_badge| |codecov_badge| |joss_badge| |pypi_badge|

.. |ci_badge| image:: https://github.com/LLNL/SSAPy/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/LLNL/SSAPy/actions/workflows/ci.yml

.. |docs_badge| image:: https://github.com/LLNL/SSAPy/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://LLNL.github.io/SSAPy

.. |codecov_badge| image:: https://codecov.io/gh/LLNL/SSAPy/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/LLNL/SSAPy

.. |joss_badge| image:: https://joss.theoj.org/papers/a629353cbdd8d64a861bb807e12c5d06/status.svg
   :target: https://joss.theoj.org/papers/a629353cbdd8d64a861bb807e12c5d06

.. |pypi_badge| image:: https://badge.fury.io/py/llnl-ssapy.svg
   :target: https://badge.fury.io/py/llnl-ssapy

`SSAPy <https://github.com/LLNL/SSAPy>`_ is a fast, flexible, high-fidelity
orbital modeling and analysis tool for orbits spanning from low-Earth orbit
into the cislunar regime.

For higher-level utilities, convenience workflows, plotting tools, GCRF-to-ITRF
coordinate conversion helpers, Lambertian magnitude / brightness calculations,
and related extensions, see the companion project
`SSAPy-Toolkit <https://github.com/llnl/SSAPy-Toolkit>`_.

SSAPy includes:

- Ability to define satellite parameters (area, mass, radiation and drag coefficients, etc.)
- Support for multiple orbit representations and input types, including TLE-based
  initialization and Keplerian, equinoctial, and Kozai mean Keplerian elements
- Fully customizable analytic force propagation models, including:

  - Earth gravity models (WGS84, EGM84, EGM96, EGM2008)
  - Lunar gravity models (point source and harmonic)
  - Radiation pressure (Earth and solar)
  - Forces for planets out to Neptune
  - Atmospheric drag models
  - Maneuvering with user-defined burn profiles

- Multiple integrators, including SGP4, Runge-Kutta (4, 8, and 7/8), SciPy,
  Keplerian, and Taylor series methods
- User-definable timesteps and orbit information retrieval times, allowing
  queries for quantities of interest such as magnitude, state vectors, TLEs,
  Keplerian elements, periapsis, apoapsis, specific angular momentum, and more
- Ground- and space-based observer models
- Lighting and visibility condition analysis
- Multiple-hypothesis tracking (MHT) UCT linker
- Vectorized computations using array broadcasting for efficient execution and
  easy deployment on HPC systems
- Short-arc probabilistic orbit determination methods
- Conjunction probability estimation
- Built-in uncertainty quantification
- Support for Monte Carlo runs and data fusion
- Support for multiple coordinate frames and coordinate transformations,
  including GCRF, IERS, GCRS Cartesian, TEME Cartesian, RA/Dec, NTW,
  zenith/azimuth, apparent positions, and orthogonal tangent plane coordinates

Installation
------------

For installation details, see the
`Installing SSAPy <https://LLNL.github.io/SSAPy/installation.html>`_
section of the documentation.

If you are looking for higher-level utilities or plotting-oriented workflows,
you may also want to install or explore
`SSAPy-Toolkit <https://github.com/llnl/SSAPy-Toolkit>`_.

Strict dependencies
-------------------

- `Python <http://docs.python-guide.org/en/latest/starting/installation/>`_ (3.8+)

The following Python packages are installed automatically when you install SSAPy:

- `numpy <https://scipy.org/install.html>`_
- `scipy <https://scipy.org/scipylib/index.html>`_
- `astropy <https://www.astropy.org/>`_
- `pyerfa <https://pypi.org/project/pyerfa/>`_
- `emcee <https://pypi.org/project/emcee/>`_
- `lmfit <https://pypi.org/project/lmfit/>`_
- `sgp4 <https://pypi.org/project/sgp4/>`_
- `jplephem <https://pypi.org/project/jplephem/>`_
- `ipyvolume <https://pypi.org/project/ipyvolume/>`_
- `tqdm <https://pypi.org/project/tqdm/>`_

Documentation
-------------

The documentation is hosted at:

`https://LLNL.github.io/SSAPy/ <https://LLNL.github.io/SSAPy/>`_

The API documentation may also be explored interactively:

.. code-block:: bash

   python3

.. code-block:: python

   import ssapy
   help(ssapy)

Contributing
------------

Contributing to SSAPy is straightforward. Please open a
`pull request <https://help.github.com/articles/using-pull-requests/>`_
targeting the ``main`` branch of the
`SSAPy repository <https://github.com/LLNL/SSAPy>`_.

For work that primarily concerns plotting, dashboards, convenience utilities,
or higher-level workflows, please also consider whether the contribution belongs
in the companion repository
`SSAPy-Toolkit <https://github.com/llnl/SSAPy-Toolkit>`_.

Your PR must pass SSAPy's required CI checks. For local testing guidance,
documentation builds, and Git workflow tips, see the
`Contribution Guide <https://LLNL.github.io/SSAPy/contribution_guide.html>`_.

SSAPy's ``main`` branch contains the latest development work.

Releases
--------

For stable installations, we recommend using one of SSAPy's tagged
`releases <https://github.com/LLNL/SSAPy/releases>`_.

The latest release is always available from the ``releases/latest`` tag.

Code of Conduct
---------------

Please note that SSAPy has a
`Code of Conduct <https://github.com/LLNL/SSAPy/blob/main/CODE_OF_CONDUCT.md>`_.
By participating in the SSAPy community, you agree to abide by its rules.

Authors
-------

SSAPy was developed with support from Lawrence Livermore National Laboratory's
(LLNL) Laboratory Directed Research and Development (LDRD) Program under projects
`19-SI-004 <https://ldrd-annual.llnl.gov/archives/ldrd-annual-2021/project-highlights/high-performance-computing-simulation-and-data-science/madstare-modeling-and-analysis-data-starved-or-ambiguous-environments>`_
and
`22-ERD-054 <https://ldrd-annual.llnl.gov/ldrd-annual-2023/project-highlights/space-security/data-demand-capable-space-domain-awareness-architecture>`_,
by the following individuals (in alphabetical order):

- `Robert Armstrong <https://orcid.org/0000-0002-6911-1038>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Julia Ebert <https://orcid.org/0000-0002-1975-772X>`_ (formerly `LLNL <https://www.llnl.gov/>`_, now at Fleet Robotics)
- `Nathan Golovich <https://orcid.org/0000-0003-2632-572X>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Noah Lifset <https://orcid.org/0000-0003-3397-7021>`_ (formerly `LLNL <https://www.llnl.gov/>`_, now PhD student at `UT Austin <https://www.utexas.edu>`_)
- `Dan Merl <https://orcid.org/0000-0003-4196-5354>`_ (`LLNL <https://www.llnl.gov/>`_) - Developer
- `Joshua Meyers <https://orcid.org/0000-0002-2308-4230>`_ (formerly `LLNL <https://www.llnl.gov/>`_, now at `KIPAC <https://kipac.stanford.edu/>`_) - Former Lead Developer
- `Caleb Miller <https://orcid.org/0000-0001-6249-0031>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Alexx Perloff <https://orcid.org/0000-0001-5230-0396>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Kerianne Pruett <https://orcid.org/0000-0002-2911-8657>`_ (formerly `LLNL <https://www.llnl.gov/>`_)
- `Edward Schlafly <https://orcid.org/0000-0002-3569-7421>`_ (formerly `LLNL <https://www.llnl.gov/>`_, now at `STScI <https://www.stsci.edu/>`_) - Former Lead Developer
- `Michael Schneider <https://orcid.org/0000-0002-8505-7094>`_ (`LLNL <https://www.llnl.gov/>`_) - Creator, Former Lead Developer
- `Travis Yeager <https://orcid.org/0000-0002-2582-0190>`_ (`LLNL <https://www.llnl.gov/>`_) - Current Lead Developer

Many thanks go to SSAPy's other
`contributors <https://github.com/llnl/ssapy/graphs/contributors>`_.

Citing SSAPy
------------

On GitHub, you can copy a citation in APA or BibTeX format via the
"Cite this repository" button. If you prefer MLA or Chicago style citations,
see the comments in
`CITATION.cff <https://github.com/LLNL/SSAPy/blob/main/CITATION.cff>`_.

You may also cite the following publications (click
`here <https://github.com/LLNL/SSAPy/blob/main/docs/source/citations.bib>`_
for BibTeX entries):

- Yeager, T., Pruett, K., & Schneider, M. (2022). *Unaided Dynamical Orbit Stability in the Cislunar Regime.* Poster presentation, Cislunar Security Conference, USA.
- Yeager, T., Pruett, K., & Schneider, M. (2023). *Long-term N-body Stability in Cislunar Space.* Poster presentation, Advanced Maui Optical and Space Surveillance (AMOS) Technologies Conference, USA.
- Yeager, T., Pruett, K., & Schneider, M. (2023, September). Long-term N-body Stability in Cislunar Space. In S. Ryan (Ed.), *Proceedings of the Advanced Maui Optical and Space Surveillance (AMOS) Technologies Conference* (p. 208). Retrieved from `https://amostech.com/TechnicalPapers/2023/Poster/Yeager.pdf <https://amostech.com/TechnicalPapers/2023/Poster/Yeager.pdf>`_

License
-------

SSAPy is distributed under the terms of the MIT license. All new contributions
must be made under the MIT license.

See the
`LICENSE <https://github.com/LLNL/SSAPy/blob/main/LICENSE>`_
and
`NOTICE <https://github.com/LLNL/SSAPy/blob/main/NOTICE>`_
files for details.

SPDX-License-Identifier: MIT

LLNL-CODE-862420

Documentation Inspiration
-------------------------

The structure and organization of this repository's documentation were inspired
by the excellent design and layout of the
`Coffea <https://coffea-hep.readthedocs.io/en/latest/index.html>`_ project.