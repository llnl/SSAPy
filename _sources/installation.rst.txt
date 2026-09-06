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
          apt install build-essential git git-lfs python3 python3-distutils python3-venv graphviz

    .. tab:: RHEL

       .. code-block:: console

          dnf install epel-release
          dnf group install "Development Tools"
          dnf install git git-lfs gcc-gfortran python3 python3-pip python3-setuptools graphviz

    .. tab:: macOS Brew

       .. code-block:: console

          brew update
          brew install gcc git git-lfs python3 graphviz

Installation
------------

As the package has been published on `PyPI <https://pypi.org/project/llnl-ssapy/>`_, it can be installed using pip. 

.. code-block:: console

   pip install llnl-ssapy

Orekit dependency
^^^^^^^^^^^^^^^^^

`Orekit <https://www.orekit.org/>`_ is an optional dependency, including the ``Orekit`` Python wrapper that is hard to find. Clone the python wrappper from here:

    `https://gitlab.orekit.org/orekit-labs/python-wrapper <https://gitlab.orekit.org/orekit-labs/python-wrapper>`_

Alternatively, the ``Orekit`` python wrapper can be installed from `Anaconda <https://www.anaconda.com/>`_.
