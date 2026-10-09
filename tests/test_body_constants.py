import pytest

from ssapy import constants

G = 6.67430e-11  # CODATA 2018 [m^3 kg^-1 s^-2]


@pytest.mark.parametrize("body", ["SUN", "MOON", "MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"])
def test_masses_are_consistent_with_gravitational_parameters(body):
    # Each tabulated mass must equal its GM / G (CODATA 2018 G) to 0.2%; the
    # masses and GMs come from different compilations, which differ by <0.05%
    # here. VENUS_MASS was 4.687e24 kg against GM / G = 4.8673e24 kg (3.7%).
    mu = getattr(constants, f"{body}_MU")
    mass = getattr(constants, f"{body}_MASS")
    assert mass == pytest.approx(mu / G, rel=2e-3)
