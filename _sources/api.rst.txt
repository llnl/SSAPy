API Reference Guide
*******************
SSAPy: Space Situational Awareness for Python

When executing

    >>> import SSAPy

a subset of the full SSAPy package is imported into the python environment.
Some packages must be imported explicitly, so as to avoid importing unnecessary
and/or heavy dependencies.  Below lists the packages available in the ``ssapy`` namespace.

   .. autosummary::
      :toctree: modules
      :template: automodapi_templ.rst
      
      ssapy.accel
      ssapy.body
      ssapy.compute
      ssapy.constants
      ssapy.correlate_tracks
      ssapy.ellipsoid
      ssapy.gravity
      ssapy.io
      ssapy.linker
      ssapy.orbit_solver
      ssapy.orbit
      ssapy.particles
      ssapy.plotUtils
      ssapy.propagator
      ssapy.rvsampler
      ssapy.simple
      ssapy.utils
