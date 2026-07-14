import numpy as np
import ssapy


def timer(f):
    import functools

    @functools.wraps(f)
    def f2(*args, **kwargs):
        import time
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        fname = repr(f).split()[1]
        print('time for %s = %.2f' % (fname, t1-t0))
        return result
    return f2


def checkAngle(a, b, rtol=0, atol=1e-13):
    # atol is 1e-13 rad (~0.1 nrad): tight enough to be physically exact but
    # above the float64 round-off floor of the trig round-trips being tested,
    # which otherwise flakes at ~1e-14 on some platforms/Python versions.
    diff = (a-b)%(2*np.pi)
    absdiff = np.min([np.abs(diff), np.abs(2*np.pi-diff)], axis=0)
    np.testing.assert_allclose(absdiff, 0, rtol=rtol, atol=atol)


def checkSphere(lon1, lat1, lon2, lat2, atol=1e-14, verbose=False):
    from ssapy.utils import unitAngle3
    x1 = np.cos(lon1)*np.cos(lat1)
    y1 = np.sin(lon1)*np.cos(lat1)
    z1 = np.sin(lat1)
    x2 = np.cos(lon2)*np.cos(lat2)
    y2 = np.sin(lon2)*np.cos(lat2)
    z2 = np.sin(lat2)
    da = unitAngle3(np.array([x1, y1, z1]).T, np.array([x2, y2, z2]).T)
    if verbose:
        print(f"max angle difference {np.rad2deg(np.max(da))*3600} arcsec")
    np.testing.assert_allclose(da, 0, rtol=0, atol=atol)


def sample_orbit(t, r_low, r_high, v_low, v_high):
    """
    Sample a random orbit by drawing a position vector with magnitude in
    (r_low, r_high), a velocity vector with magnitude in (v_low, v_high), and
    a direction, at time t.
    """
    def sample_vector(lower, upper):
        """
        Helper function for sampling a position and velocity vector.
        """
        # Sample magnitude of vector
        mag = np.random.uniform(lower, upper)
        # Sample a random direction (not uniform on sphere)
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        vec_x = mag * np.cos(theta) * np.sin(phi)
        vec_y = mag * np.sin(theta) * np.sin(phi)
        vec_z = mag * np.cos(phi)
        vec = np.array([vec_x, vec_y, vec_z])
        return vec
    
    # Sample position and velocity
    r = sample_vector(r_low, r_high)
    v = sample_vector(v_low, v_high)

    return ssapy.Orbit(r, v, t)


def sample_GEO_orbit(t, r_low=4e7, r_high=5e7, v_low=2.7e3, v_high=3.3e3):
    """
    Returns a ssapy.Orbit object with a distance near RGEO and a velocity near VGEO.
    """
    return sample_orbit(t, r_low, r_high, v_low, v_high)


def sample_LEO_orbit(t, r_low=7e6, r_high=1e7, v_low=7.7e3, v_high=7.9e3):
    """
    Returns a ssapy.Orbit object with a position near LEO and a velocity near VLEO.
    """
    return sample_orbit(t, r_low, r_high, v_low, v_high)
