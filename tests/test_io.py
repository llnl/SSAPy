import numpy as np
from astropy.time import Time
from ssapy import io


def test_read_tle_catalog(tmp_path):
    file = tmp_path / "tle.txt"
    file.write_text(
        "1 25544U 98067A   21073.51465278  .00000282\n"
        "2 25544  51.6430 249.4256 0001791 160.3235 199.7986 15.48988277272524\n"
    )
    result = io.read_tle_catalog(str(file))
    assert len(result) == 1
    assert result[0][0].startswith("1 ")
    assert result[0][1].startswith("2 ")


def test_read_tle(tmp_path):
    file = tmp_path / "tle.txt"
    file.write_text(
        "ISS (ZARYA)\n"
        "1 25544U 98067A   21073.51465278  .00000282\n"
        "2 25544  51.6430 249.4256 0001791 160.3235 199.7986 15.48988277272524\n"
    )
    line1, line2 = io.read_tle("ISS (ZARYA)", str(file))
    assert line1.startswith("1 ")
    assert line2.startswith("2 ")


def test_make_tle_and_parse_tle():
    a = 6780000.0
    e = 0.001
    i = np.radians(51.6)
    pa = np.radians(45.0)
    raan = np.radians(120.0)
    true_anomaly = np.radians(60.0)
    t = Time.now()

    line1, line2 = io.make_tle(a, e, i, pa, raan, true_anomaly, t)
    parsed = io.parse_tle((line1, line2))

    assert isinstance(parsed, tuple)
    assert len(parsed) == 7
    assert np.isclose(parsed[0], a, rtol=0.01)


def test_parse_overpunched():
    assert io.parse_overpunched("J1234") == "-11234"
    assert io.parse_overpunched("51234") == "51234"