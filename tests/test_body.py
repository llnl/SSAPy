import numpy as np
import pytest

from ssapy import body
from ssapy.constants import (
    EARTH_MU,
    EARTH_RADIUS,
    JUPITER_MU,
    JUPITER_RADIUS,
    MARS_MU,
    MARS_RADIUS,
    MERCURY_MU,
    MERCURY_RADIUS,
    MOON_MU,
    NEPTUNE_MU,
    NEPTUNE_RADIUS,
    SATURN_MU,
    SATURN_RADIUS,
    SUN_MU,
    URANUS_MU,
    URANUS_RADIUS,
    VENUS_MU,
    VENUS_RADIUS,
)


class _FakeSegment:
    def __init__(self, value):
        self.value = np.array(value, dtype=float)

    def compute(self, jd0, jd1):
        assert jd0 == 2400000.5
        assert jd1 == 7.0
        return self.value.copy()


class _FakeKernel:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return _FakeSegment(self.values[key])


def test_body_defaults():
    obj = body.Body(mu=1.0, radius=2.0)

    assert obj.mu == 1.0
    assert obj.radius == 2.0
    np.testing.assert_allclose(obj.position(0.0), np.zeros(3))
    np.testing.assert_allclose(obj.orientation(0.0), np.eye(3))
    assert obj.harmonics is None


def test_position_helpers_use_kernel_differences(monkeypatch):
    monkeypatch.setattr(body, "_gpsToTT", lambda t: 7.0)

    moon_position = body.MoonPosition.__new__(body.MoonPosition)
    moon_position.kernel = _FakeKernel({
        (3, 301): [4.0, 5.0, 6.0],
        (3, 399): [1.0, 2.0, 3.0],
    })
    np.testing.assert_allclose(moon_position(123.0), [3000.0, 3000.0, 3000.0])

    sun_position = body.SunPosition.__new__(body.SunPosition)
    sun_position.kernel = _FakeKernel({
        (0, 10): [10.0, 20.0, 30.0],
        (0, 3): [1.0, 2.0, 3.0],
        (3, 399): [0.1, 0.2, 0.3],
    })
    np.testing.assert_allclose(sun_position(123.0), [8900.0, 17800.0, 26700.0])

    planet_position = body.PlanetPosition.__new__(body.PlanetPosition)
    planet_position.planet_index = 5
    planet_position.kernel = _FakeKernel({
        (0, 5): [7.0, 8.0, 9.0],
        (0, 3): [1.0, 2.0, 3.0],
        (3, 399): [0.5, 1.0, 1.5],
    })
    np.testing.assert_allclose(planet_position(123.0), [5500.0, 5000.0, 4500.0])


def test_planet_position_init_loads_kernel(monkeypatch):
    from jplephem.spk import SPK

    opened = []

    def fake_open(path):
        opened.append(path)
        return "planet-kernel"

    monkeypatch.setattr(SPK, "open", fake_open)

    planet_position = body.PlanetPosition(planet_index=5)

    assert planet_position.kernel == "planet-kernel"
    assert planet_position.planet_index == 5
    assert opened[0].endswith("de430.bsp")


def test_orientation_helpers(monkeypatch):
    monkeypatch.setattr(body, "_gpsToTT", lambda t: 7.0)

    earth_orientation = body.EarthOrientation()
    supplied = np.arange(9.0).reshape(3, 3)
    assert earth_orientation(0.0, _E=supplied) is supplied

    np.testing.assert_allclose(
        body.MoonOrientation._Rx(np.pi / 2),
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        atol=1e-15,
    )
    np.testing.assert_allclose(
        body.MoonOrientation._Rz(np.pi / 2),
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-15,
    )

    moon_orientation = body.MoonOrientation.__new__(body.MoonOrientation)
    segment = type(
        "Segment",
        (),
        {"compute": lambda self, jd0, jd1: (np.zeros(3), None)},
    )()
    moon_orientation.kernel = type("Kernel", (), {"segments": [segment]})()
    np.testing.assert_allclose(moon_orientation(123.0), np.eye(3))


def test_get_body_uses_expected_models_without_loading_data(monkeypatch):
    egm_calls = []
    tab_calls = []

    class FakeHarmonics:
        @staticmethod
        def fromEGM(model):
            egm_calls.append(model)
            return ("egm", model)

        @staticmethod
        def fromTAB(model):
            tab_calls.append(model)
            return ("tab", model)

    monkeypatch.setattr(body, "EarthOrientation", lambda: "earth-orientation")
    monkeypatch.setattr(body, "MoonOrientation", lambda: "moon-orientation")
    monkeypatch.setattr(body, "MoonPosition", lambda: "moon-position")
    monkeypatch.setattr(body, "SunPosition", lambda: "sun-position")
    monkeypatch.setattr(body, "_HarmonicCoefficients", FakeHarmonics)

    class FakePlanetPosition:
        def __init__(self, planet_index):
            self.planet_index = planet_index

    monkeypatch.setattr(body, "PlanetPosition", FakePlanetPosition)

    earth = body.get_body("Earth")
    assert earth.mu == EARTH_MU
    assert earth.radius == EARTH_RADIUS
    assert earth.orientation == "earth-orientation"
    assert earth.harmonics == ("egm", "EGM84")

    model_aliases = [
        ("84", "EGM84"),
        ("1984", "EGM84"),
        ("96", "EGM96"),
        ("1996", "EGM96"),
        ("08", "EGM2008"),
        ("2008", "EGM2008"),
    ]
    for alias, expected in model_aliases:
        assert body.get_body("earth", alias).harmonics == ("egm", expected)

    moon = body.get_body("moon")
    assert moon.mu == MOON_MU
    assert moon.position == "moon-position"
    assert moon.orientation == "moon-orientation"
    assert moon.harmonics == ("tab", "gggrx_1200a_sha.tab")

    sun = body.get_body("sun")
    assert sun.mu == SUN_MU
    assert sun.radius == 695700000.0
    assert sun.position == "sun-position"

    planet_cases = [
        ("mercury", MERCURY_MU, MERCURY_RADIUS, 1),
        ("venus", VENUS_MU, VENUS_RADIUS, 2),
        ("mars", MARS_MU, MARS_RADIUS, 4),
        ("jupiter", JUPITER_MU, JUPITER_RADIUS, 5),
        ("saturn", SATURN_MU, SATURN_RADIUS, 6),
        ("uranus", URANUS_MU, URANUS_RADIUS, 7),
        ("neptune", NEPTUNE_MU, NEPTUNE_RADIUS, 8),
    ]
    for name, mu, radius, index in planet_cases:
        obj = body.get_body(name)
        assert obj.mu == mu
        assert obj.radius == radius
        assert obj.position.planet_index == index

    assert egm_calls[:7] == [
        "EGM84",
        "EGM84",
        "EGM84",
        "EGM96",
        "EGM96",
        "EGM2008",
        "EGM2008",
    ]
    assert tab_calls == ["gggrx_1200a_sha.tab"]
    with pytest.raises(ValueError, match="Unknown body pluto"):
        body.get_body("pluto")
