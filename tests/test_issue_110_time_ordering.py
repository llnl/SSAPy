import numpy as np
from astropy.time import Time

from ssapy.correlate_tracks import time_ordered_satIDs


def _object_time_data():
    data = np.empty(5, dtype=[("satID", "i8"), ("time", object)])
    data["satID"] = [2, 1, 2, 3, 1]
    gps = np.array([30.0, 10.0, 20.0, 40.0, 15.0])
    data["time"] = [Time(t, format="gps") for t in gps]
    return data


def test_time_ordered_satids_accepts_object_array_of_astropy_times():
    data = _object_time_data()

    satids, times = time_ordered_satIDs(data, with_time=True)

    assert satids == [1, 2, 3]
    np.testing.assert_allclose(times, [10.0, 20.0, 40.0])


def test_time_ordered_satids_backward_uses_latest_observation_per_id():
    data = _object_time_data()

    satids, times = time_ordered_satIDs(data, with_time=True, order="backward")

    assert satids == [3, 2, 1]
    np.testing.assert_allclose(times, [40.0, 30.0, 15.0])


def test_time_ordered_satids_accepts_numeric_gps_seconds():
    data = np.empty(4, dtype=[("satID", "i8"), ("time", "f8")])
    data["satID"] = [7, 8, 7, 9]
    data["time"] = [3.0, 1.0, 2.0, 4.0]

    satids, times = time_ordered_satIDs(data, with_time=True)

    assert satids == [8, 7, 9]
    np.testing.assert_allclose(times, [1.0, 2.0, 4.0])
