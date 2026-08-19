==============================================
SSAPy - Space Situational Awareness for Python
==============================================

|ci_badge| |docs_badge| |codecov_badge| |joss_badge| |pypi_badge|

.. |ci_badge| image:: https://github.com/LLNL/SSAPy/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/LLNL/SSAPy/actions/workflows/ci.yml

.. |docs_badge| image:: https://github.com/LLNL/SSAPy/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://software.llnl.gov/SSAPy/

.. |codecov_badge| image:: https://codecov.io/gh/LLNL/SSAPy/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/LLNL/SSAPy

.. |joss_badge| image:: https://joss.theoj.org/papers/10.21105/joss.08147/status.svg
   :target: https://doi.org/10.21105/joss.08147

.. |pypi_badge| image:: https://badge.fury.io/py/llnl-ssapy.svg
   :target: https://badge.fury.io/py/llnl-ssapy

`SSAPy <https://github.com/LLNL/SSAPy>`_ is a flexible, physics-based
orbital modeling and analysis tool for orbits spanning from low-Earth orbit
into the cislunar regime.

SSAPy retains the core coordinate, observer-geometry, and propagation routines.
For higher-level utilities, convenience workflows, plotting tools, workflow
wrappers around coordinate conversions, Lambertian magnitude / brightness
calculations, and related extensions, see the companion project
`SSAPy-Toolkit <https://github.com/llnl/SSAPy-Toolkit>`__.

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
  queries for quantities of interest such as state vectors, TLEs,
  Keplerian elements, periapsis, apoapsis, specific angular momentum, and more
- Ground- and space-based observer models
- Lighting and visibility condition analysis
- Multiple-hypothesis tracking (MHT) UCT linker
- Vectorized computations using array broadcasting for efficient execution and
  easy deployment on HPC systems
- Short-arc probabilistic orbit determination methods
- Conjunction probability estimation
- Built-in uncertainty quantification
- Monte Carlo sampling, particle-based uncertainty representations, and
  track linking/model selection
- Support for multiple coordinate frames and coordinate transformations,
  including GCRF, IERS, GCRS Cartesian, TEME Cartesian, RA/Dec, NTW,
  zenith/azimuth, apparent positions, and orthogonal tangent plane coordinates

SSAPy-Toolkit
-------------

SSAPy provides the core propagation and modeling engine. Many
higher-level, analysis-ready capabilities built on top of it live in the
companion project
`SSAPy-Toolkit <https://github.com/LLNL/SSAPy-Toolkit>`__ (sometimes abbreviated
*SSATK*), including:

- Higher-level utilities and convenience workflows that wrap common SSAPy tasks
- Plotting tools for orbit and analysis visualization
- Workflow-level wrappers around SSAPy's coordinate and observer-geometry
  routines
- Lambertian magnitude / brightness calculations
- Additional related extensions

If your work centers on plotting, dashboards, convenience utilities, or
higher-level workflows, SSAPy-Toolkit is often the best place to start — and the
natural home for contributions of that kind.

Installation
------------

For installation details, see the
`Installing SSAPy <https://software.llnl.gov/SSAPy/installation.html>`_
section of the documentation.

If you are looking for higher-level utilities or plotting-oriented workflows,
you may also want to install or explore
`SSAPy-Toolkit <https://github.com/llnl/SSAPy-Toolkit>`__.

Strict dependencies
-------------------

- `Python <http://docs.python-guide.org/en/latest/starting/installation/>`_ (3.10+)

The following Python packages are installed automatically when you install SSAPy:

- `numpy <https://scipy.org/install.html>`_
- `scipy <https://scipy.org/scipylib/index.html>`_
- `astropy <https://www.astropy.org/>`_
- `pyerfa <https://pypi.org/project/pyerfa/>`_
- `emcee <https://pypi.org/project/emcee/>`_
- `lmfit <https://pypi.org/project/lmfit/>`_
- `sgp4 <https://pypi.org/project/sgp4/>`_
- `ipython_genutils <https://pypi.org/project/ipython-genutils/>`_
- `jplephem <https://pypi.org/project/jplephem/>`_
- `ipyvolume <https://pypi.org/project/ipyvolume/>`_
- `tqdm <https://pypi.org/project/tqdm/>`_

Documentation
-------------

The documentation is hosted at:

`https://software.llnl.gov/SSAPy/ <https://software.llnl.gov/SSAPy/>`_

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
`SSAPy-Toolkit <https://github.com/llnl/SSAPy-Toolkit>`__.

Your PR must pass SSAPy's required CI checks. For local testing guidance,
documentation builds, and Git workflow tips, see the
`Contribution Guide <https://software.llnl.gov/SSAPy/contribution_guide.html>`_.

SSAPy's ``main`` branch contains the latest development work.

Releases
--------

For stable installations, we recommend installing the published
`llnl-ssapy <https://pypi.org/project/llnl-ssapy/>`_ package from PyPI or using
one of SSAPy's versioned source
`tags <https://github.com/LLNL/SSAPy/tags>`_.

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
by the following individuals. For software citation order and metadata, use
`CITATION.cff <https://github.com/LLNL/SSAPy/blob/main/CITATION.cff>`_.

- `Joshua E. Meyers <https://orcid.org/0000-0002-2308-4230>`_ (SLAC National Accelerator Laboratory)
- `Travis Yeager <https://orcid.org/0000-0002-2582-0190>`_ (`LLNL <https://www.llnl.gov/>`_) - Current Lead Developer
- `Michael Schneider <https://orcid.org/0000-0002-8505-7094>`_ (`LLNL <https://www.llnl.gov/>`_) - Creator, Former Lead Developer
- `Edward Schlafly <https://orcid.org/0000-0002-3569-7421>`_ (`STScI <https://www.stsci.edu/>`_) - Former Lead Developer
- `Julia Ebert <https://orcid.org/0000-0002-1975-772X>`_ (Fleet Robotics)
- `Denvir Higgins <https://orcid.org/0000-0002-7579-1092>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Jason Bernstein <https://orcid.org/0000-0002-3391-5931>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Daniel Merl <https://orcid.org/0000-0003-4196-5354>`_ (`LLNL <https://www.llnl.gov/>`_)
- Imène Goumiri
- `Robert Armstrong <https://orcid.org/0000-0002-6911-1038>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Noah Lifset <https://orcid.org/0000-0003-3397-7021>`_ (`UT Austin <https://www.utexas.edu/>`_)
- `Alexx Perloff <https://orcid.org/0000-0001-5230-0396>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Peter McGill <https://orcid.org/0000-0002-1052-6749>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Nathan Golovich <https://orcid.org/0000-0003-2632-572X>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Kerianne Pruett <https://orcid.org/0000-0002-2911-8657>`_ (`LLNL <https://www.llnl.gov/>`_)
- `Caleb Miller <https://orcid.org/0000-0001-6249-0031>`_ (`LLNL <https://www.llnl.gov/>`_)
- `William A. Dawson <https://orcid.org/0000-0003-0248-6123>`_ (`LLNL <https://www.llnl.gov/>`_)

Many thanks go to SSAPy's additional
`contributors <https://github.com/llnl/ssapy/graphs/contributors>`_.

Citing SSAPy
------------

If you use SSAPy in your research, please cite the software using the
repository metadata in
`CITATION.cff <https://github.com/LLNL/SSAPy/blob/main/CITATION.cff>`_.
On GitHub, use the "Cite this repository" button to copy the citation in APA
or BibTeX format.

The related JOSS paper may also be cited separately:

- Meyers, J. E., Schneider, M. D., Ebert, J. T., Schlafly, E. F., Yeager, T.,
  Perloff, A., Merl, D., Lifset, N., Bernstein, J., Dawson, W. A., Golovich, N.,
  Higgins, D., McGill, P., Miller, C., & Pruett, K. (2025). *SSAPy - Space
  Situational Awareness for Python.* Journal of Open Source Software, 10(111),
  8147. `doi:10.21105/joss.08147 <https://doi.org/10.21105/joss.08147>`_

BibTeX::

    @article{Meyers2025,
      doi       = {10.21105/joss.08147},
      url       = {https://doi.org/10.21105/joss.08147},
      year      = {2025},
      publisher = {The Open Journal},
      volume    = {10},
      number    = {111},
      pages     = {8147},
      author    = {Meyers, Joshua E. and Schneider, Michael D. and Ebert, Julia T. and Schlafly, Edward F. and Yeager, Travis and Perloff, Alexx and Merl, Daniel and Lifset, Noah and Bernstein, Jason and Dawson, William A. and Golovich, Nathan and Higgins, Denvir and McGill, Peter and Miller, Caleb and Pruett, Kerianne},
      title     = {SSAPy - Space Situational Awareness for Python},
      journal   = {Journal of Open Source Software}
    }

You may also cite the following publications (click
`here <https://github.com/LLNL/SSAPy/blob/main/docs/source/refs.bib>`_
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
