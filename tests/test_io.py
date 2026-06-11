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

def test_parseB3Line_type9_sensor_position():
    # Regression test: the z coordinate of the sensor position for
    # space-based observation types (8 and 9) was previously read from
    # columns [65:73] instead of [64:73], dropping a digit and
    # misaligning the field.
    from ssapy.io import parseB3Line

    line = list(" " * 76)

    def put(start, text):
        line[start:start + len(text)] = list(text)

    put(0, "U")          # security classification
    put(1, "12345")      # satellite ID
    put(6, "511")        # sensor ID
    put(9, "21")         # year
    put(11, "123")       # day of year
    put(14, "12")        # hour
    put(16, "34")        # minute
    put(18, "56000")     # milliseconds of minute
    put(23, " 451234")   # polar angle (declination)
    put(30, "1230456")   # RA in hhmmsss (type 9)
    put(46, "  1234567")  # sensor x, columns 47-55
    put(55, " -2345678")  # sensor y, columns 56-64
    put(64, "  3456789")  # sensor z, columns 65-73
    put(74, "9")          # observation type
    put(75, "1")          # equinox type

    rec = parseB3Line("".join(line))
    assert rec['x'][0] == 1234567.0
    assert rec['y'][0] == -2345678.0
    assert rec['z'][0] == 3456789.0
    assert rec['type'][0] == 9
    assert rec['equinoxType'][0] == 1


def test_b3obs2pos_blank_equinox_column():
    # Regression test: b3obs2pos previously did int(line[75])
    # unconditionally, raising ValueError for the (common) observation
    # types whose equinox column is blank.
    from ssapy.io import b3obs2pos

    line = list(" " * 76)

    def put(start, text):
        line[start:start + len(text)] = list(text)

    put(0, "U")
    put(1, "12345")
    put(6, "511")
    put(9, "21")
    put(11, "123")
    put(14, "12")
    put(16, "34")
    put(18, "56000")
    put(23, " 451234")   # elevation
    put(30, "1230456")   # azimuth
    put(74, "1")         # obs type 1: azimuth & elevation, no equinox

    pos = b3obs2pos("".join(line))
    assert pos["satnum"] == 12345
    assert pos["sensnum"] == 511
