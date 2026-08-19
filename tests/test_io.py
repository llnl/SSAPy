import numpy as np
import pytest
from astropy.time import Time
import astropy.units as u
from ssapy import io


def _make_b3_line(obs_type="5", sensor="511", equinox="1"):
    line = list(" " * 76)

    def put(start, text):
        line[start:start + len(text)] = list(text)

    put(0, "U")
    put(1, "12345")
    put(6, sensor)
    put(9, "21")
    put(11, "123")
    put(14, "12")
    put(16, "34")
    put(18, "56000")
    put(23, " 45123")
    put(30, "1230456")
    put(38, " 123456")
    put(45, "2")
    put(47, " 123456")
    put(74, obs_type)
    put(75, equinox)
    return "".join(line)


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

    file3 = tmp_path / "tle3.txt"
    file3.write_text(
        "ISS (ZARYA)\n"
        "1 25544U 98067A   21073.51465278  .00000282\n"
        "2 25544  51.6430 249.4256 0001791 160.3235 199.7986 15.48988277272524\n"
    )
    result3 = io.read_tle_catalog(str(file3), n_lines=3)
    assert result3[0][0] == "ISS (ZARYA)"


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

    with pytest.raises(KeyError):
        io.read_tle("MISSING", str(file))

    bad_file = tmp_path / "bad_tle.txt"
    bad_file.write_text("ISS (ZARYA)\n1 25544U 98067A\n")
    with pytest.raises(IOError):
        io.read_tle("ISS (ZARYA)", str(bad_file))


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


def test_parse_tle_legacy_year_and_epoch_fallback(monkeypatch):
    line1 = list("1 25544U 98067A   99001.00000000  .00000282  00000-0  00000-0 0  9991")
    line2 = "2 25544  51.6430 249.4256 0001791 160.3235 199.7986 15.48988277272524"
    parsed = io.parse_tle(("".join(line1), line2))
    assert Time(parsed[-1], format="gps").utc.datetime.year == 1999

    import sgp4.ext

    monkeypatch.setattr(sgp4.ext, "days2mdhms", lambda year, day: (13, 1, 0, 0, 0.0))
    monkeypatch.setattr(sgp4.ext, "jday", lambda *args: 0.0)
    monkeypatch.setattr(sgp4.ext, "invjday", lambda jd: (2021, 12, 31, 0, 0, 0.0))

    line1[18:32] = list("21001.00000000")
    parsed = io.parse_tle(("".join(line1), line2))
    assert Time(parsed[-1], format="gps").utc.datetime.year == 2021


def test_get_tel_pos_itrf_to_gcrs_returns_cartesian_quantity():
    pos = io.get_tel_pos_itrf_to_gcrs(Time(0.0, format="gps"), tel_label="511")
    assert pos.shape == (3,)
    assert pos.unit.is_equivalent(u.m)


def test_parse_overpunched():
    assert io.parse_overpunched("J1234") == "-11234"
    assert io.parse_overpunched("51234") == "51234"


def test_b3_line_file_and_catalog_helpers(tmp_path):
    line = _make_b3_line(obs_type="5")
    rec = io.parseB3Line(line)
    assert rec['type'][0] == 5
    assert rec['equinoxType'][0] == 1
    assert np.isfinite(rec['range'][0])
    assert np.isfinite(rec['slantRangeRate'][0])

    invalid_type = list(line)
    invalid_type[74] = " "
    rec_invalid = io.parseB3Line("".join(invalid_type))
    assert rec_invalid['type'][0] == -999

    bad_range_exp = list(line)
    bad_range_exp[45] = " "
    rec_bad_exp = io.parseB3Line("".join(bad_range_exp))
    assert np.isfinite(rec_bad_exp['range'][0])

    bad_sensor_xyz = list(_make_b3_line(obs_type="9"))
    bad_sensor_xyz[46:55] = list(" notnum  ")
    rec_bad_xyz = io.parseB3Line("".join(bad_sensor_xyz))
    assert np.isnan(rec_bad_xyz["x"][0])
    assert np.isnan(rec_bad_xyz["y"][0])
    assert np.isnan(rec_bad_xyz["z"][0])

    with pytest.raises(ValueError, match="B3OBS format error"):
        io.b3obs2pos("too short")

    pos = io.b3obs2pos(line)
    assert pos['satnum'] == 12345
    assert pos['sensnum'] == 511
    assert np.isfinite(pos['ra'])
    assert np.isfinite(pos['dec'])
    assert pos['tel_pos'].unit == u.km

    path = tmp_path / "obs.b3"
    path.write_text(line + "\n")
    parsed = io.parseB3(str(path))
    assert len(parsed) == 1
    assert 'gps' in parsed.colnames

    catalog = io.load_b3obs_file(str(path))
    assert catalog['satnum'] == [12345]
    assert catalog['sensnum'] == [511]
    assert np.isfinite(catalog['ra'][0])

    type9_unknown_sensor = _make_b3_line(obs_type="9", sensor="999")
    pos9 = io.b3obs2pos(type9_unknown_sensor[:9] + "99" + type9_unknown_sensor[11:])
    assert pos9["sensnum"] == 999
    assert pos9["epoch"].year == 1999
    assert pos9["tel_pos"] is None

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
