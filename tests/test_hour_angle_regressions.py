import numpy as np
import pytest

from ssapy import utils


def test_rightascension_to_hourangle_matches_documented_examples():
    for right_ascension, local_time in [
        ("10:30:00", "12:45:00"),
        (157.5, 191.25),
    ]:
        hour_angle = utils.rightascension_to_hourangle(right_ascension, local_time)
        assert utils.hms_to_dd(hour_angle) == pytest.approx(33.75)


def test_rightascension_to_hourangle_wraps_through_midnight():
    hour_angle = utils.rightascension_to_hourangle("02:00:00", "01:00:00")
    assert utils.hms_to_dd(hour_angle) == pytest.approx(345.0)


def test_equatorial_to_horizontal_accepts_numeric_ra_and_local_time():
    from_ra = utils.equatorial_to_horizontal(
        observer_latitude=35.0,
        declination=20.0,
        right_ascension=157.5,
        local_time=191.25,
    )
    from_hour_angle = utils.equatorial_to_horizontal(
        observer_latitude=35.0,
        declination=20.0,
        hour_angle=33.75,
    )
    np.testing.assert_allclose(from_ra, from_hour_angle, atol=1e-12)
