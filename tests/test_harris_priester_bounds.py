import numpy as np

from ssapy.accel import AccelDrag
from ssapy.constants import EARTH_RADIUS


def test_harris_priester_upper_table_boundary_uses_endpoint_density():
    atmosphere = AccelDrag().atm
    radius = EARTH_RADIUS + 1_000_000.0

    density = atmosphere.density(radius, 0.0, 0.0, 0.0, 0.0)

    assert np.isfinite(density)
    assert 1.150e-15 <= density <= 1.810e-14


def test_harris_priester_above_upper_table_boundary_is_zero():
    atmosphere = AccelDrag().atm
    radius = EARTH_RADIUS + 1_000_001.0

    density = atmosphere.density(radius, 0.0, 0.0, 0.0, 0.0)

    assert density == 0.0
