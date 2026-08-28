"""
Class to handle transformations between ECEF x,y,z coords and geodetic
longitude, latitude, and height.

Technically, only handles a one-axis ellipsoid, defined via a flattening
parameter f, but that's good enough for simple Earth models.
"""

import numpy as np
from .utils import continueClass
from . import _ssapy
from ._ssapy import Ellipsoid


@continueClass
class Ellipsoid:
    """
    A class representing an ellipsoid, providing methods to convert between spherical and Cartesian coordinates.

    Methods
    -------
    sphereToCart(lon, lat, height)
        Converts spherical coordinates (longitude, latitude, height) to Cartesian coordinates (x, y, z).

    cartToSphere(x, y, z)
        Converts Cartesian coordinates (x, y, z) to spherical coordinates (longitude, latitude, height).
    """

    def sphereToCart(self, lon, lat, height):
        """
        Converts spherical coordinates to Cartesian coordinates.

        Parameters
        ----------
        lon : array-like
            Longitude values in radians.
        lat : array-like
            Latitude values in radians.
        height : array-like
            Height above the ellipsoid surface.

        Returns
        -------
        tuple of array-like
            A tuple containing:
            - x : array-like
                Cartesian x-coordinate.
            - y : array-like
                Cartesian y-coordinate.
            - z : array-like
                Cartesian z-coordinate.

        Notes
        -----
        This method uses broadcasting to handle inputs of varying shapes and ensures
        contiguous arrays for efficient computation.
        """

        lon, lat, height = np.broadcast_arrays(lon, lat, height)
        lon = np.ascontiguousarray(lon, dtype=np.float64)
        lat = np.ascontiguousarray(lat, dtype=np.float64)
        height = np.ascontiguousarray(height, dtype=np.float64)

        x = np.empty_like(lon)
        y = np.empty_like(lon)
        z = np.empty_like(lon)

        self._sphereToCart(
            lon.ctypes.data, lat.ctypes.data, height.ctypes.data, lon.size,
            x.ctypes.data, y.ctypes.data, z.ctypes.data
        )

        return x, y, z

    def cartToSphere(self, x, y, z):
        """
        Converts Cartesian coordinates to spherical coordinates.

        Parameters
        ----------
        x : array-like
            Cartesian x-coordinate.
        y : array-like
            Cartesian y-coordinate.
        z : array-like
            Cartesian z-coordinate.

        Returns
        -------
        tuple of array-like
            A tuple containing:
            - lon : array-like
                Longitude values in radians.
            - lat : array-like
                Latitude values in radians.
            - height : array-like
                Height above the ellipsoid surface.

        Notes
        -----
        This method uses broadcasting to handle inputs of varying shapes and ensures
        contiguous arrays for efficient computation.
        """

        x, y, z = np.broadcast_arrays(x, y, z)
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        z = np.ascontiguousarray(z, dtype=np.float64)

        lon = np.empty_like(x)
        lat = np.empty_like(x)
        height = np.empty_like(x)

        self._cartToSphere(
            x.ctypes.data, y.ctypes.data, z.ctypes.data, x.size,
            lon.ctypes.data, lat.ctypes.data, height.ctypes.data
        )

        return lon, lat, height
