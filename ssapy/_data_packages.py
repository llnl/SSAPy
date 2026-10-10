"""The ``ssa-data-*`` distributions that ship SSAPy's data.

Each distribution ``ssa-data-<name>`` installs the import package
``ssa_data_<name>`` with its files below ``ssa_data_<name>/data``. This module
is the single list SSAPy searches.
"""

#: Required dependencies of SSAPy (ephemerides, gravity models, textures).
REQUIRED = (
    "ssa_data_core",
    "ssa_data_gravity",
    "ssa_data_lunar",
    "ssa_data_lunar_gravity",
)

#: Added by the ``all-data`` extra.
OPTIONAL = (
    "ssa_data_propulsion",
    "ssa_data_benchmarks",
)

ALL = REQUIRED + OPTIONAL

#: Command that installs every data package.
INSTALL_HINT = "pip install 'llnl-ssapy[all-data]'"
