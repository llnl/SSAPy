Getting Started
===============

System Prerequisites
--------------------

``SSAPy`` has the following minimum system requirements, which are assumed to be present on the machine where ``SSAPy`` is run:

.. csv-table:: System prerequisites for SSAPy
   :file: tables/system_prerequisites.csv
   :header-rows: 1

These requirements can be easily installed on most modern macOS and Linux systems.

.. tabs::

    .. tab:: Debian/Ubuntu

       .. code-block:: console

          apt update
          apt install build-essential git python3 python3-distutils python3-venv graphviz

    .. tab:: RHEL

       .. code-block:: console

          dnf install epel-release
          dnf group install "Development Tools"
          dnf install git gcc-gfortran python3 python3-pip python3-setuptools graphviz

    .. tab:: macOS Brew

       .. code-block:: console

          brew update
          brew install gcc git python3 graphviz

Installation
------------

As the package has been published on `PyPI <https://pypi.org/project/llnl-ssapy/>`_, it can be installed using pip. 

.. code-block:: console

   pip install llnl-ssapy

SSAPy's data (ephemerides, gravity models, and textures) is installed with the
package; no Git LFS is needed.

Planetary ephemerides
^^^^^^^^^^^^^^^^^^^^^

SSAPy uses JPL DE440 by default, from the short kernel ``de440s.bsp``
(1849-2150). A 1900-2150 excerpt of DE430 is also installed for reproducing
results from SSAPy 1.1.10 and earlier; select it with
``ssapy.ephemeris.set_planetary_ephemeris("de430")`` or
``SSAPY_EPHEMERIS=de430``.

For epochs outside those spans SSAPy downloads a longer kernel of the same
ephemeris from NAIF (``de440.bsp`` or ``de430.bsp``, 1550-2650, 120 MB) and,
for DE440 beyond 1550-2650, DE441 (1.65 GB per half), checks its SHA-256,
caches it in ``~/.cache/ssapy`` (``SSAPY_DATA_CACHE``), and warns before
downloading. Set ``SSAPY_EPHEMERIS_DOWNLOAD=0`` to forbid downloads, for
example on compute nodes without network access, and prefetch on a node that
has it:

.. code-block:: console

   SSAPY_DATA_CACHE=/shared/ssapy-cache python -c "import ssapy.ephemeris as e; e.fetch()"

Orekit dependency
^^^^^^^^^^^^^^^^^

`Orekit <https://www.orekit.org/>`_ is an optional dependency, including the ``Orekit`` Python wrapper that is hard to find. Clone the python wrappper from here:

    `https://gitlab.orekit.org/orekit-labs/python-wrapper <https://gitlab.orekit.org/orekit-labs/python-wrapper>`_

Alternatively, the ``Orekit`` python wrapper can be installed from `Anaconda <https://www.anaconda.com/>`_.
