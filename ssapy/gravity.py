"""
Classes for gravity-related accelerations.
"""

# from functools import lru_cache

import numpy as np

from .accel import Accel as _Accel, _invnorm
from .utils import find_file, norm
from . import _ssapy


class HarmonicCoefficients:
    """Class to hold coefficients for a spherical harmonic expansion of a
    gravitational potential.
    """
    @classmethod
    def fromEGM(cls, filename):
        """Construct a HarmonicCoefficients object from a .egm/.cof file pair as
        available from https://geographiclib.sourceforge.io/html/gravity.html

        SSAPy comes with four models for Earth gravity of varying degrees and
        orders:

        name     |  degree  |  order
        ----------------------------
        WGS84    |      20  |      0
        EGM84    |     180  |    180
        EGM96    |     360  |    360
        EGM2008  |    2190  |   2159

        Note that many coefficients with large degree and/or order underflow
        double precision floats when denormalized for use in computing
        accelerations and are therefore effectively ignored.

        Parameters
        ----------
        filename : str
            Either the name of the .egm file or one of the names listed above.
        """
        original_filename = filename
        try:
            filename = find_file(filename, ext=".egm")
        except FileNotFoundError:
            # For backwards compatibility, also try with .lower()
            try:
                filename = find_file(filename.lower(), ext=".egm")
            except FileNotFoundError:
                raise FileNotFoundError(original_filename)

        egm_fn = filename
        cof_fn = egm_fn + ".cof"

        # First get metadata file
        with open(egm_fn, "r") as f:
            radius = None
            MG = None
            for line in f.readlines():
                if line.startswith("ModelRadius"):
                    radius = float(line[11:])
                if line.startswith("ModelMass"):
                    MG = float(line[9:])
            if radius is None:
                raise ValueError(
                    f"Could not find model radius in file {cof_fn}"
                )
            if MG is None:
                raise ValueError(
                    f"Could not find model mass in gravity file {cof_fn}"
                )

        with open(cof_fn, "rb") as f:
            # First 8 bytes are name
            name = f.read(8).decode("ascii")

            # Next 8 bytes are N, M
            n_max = n_max = int.from_bytes(f.read(4), "little")
            m_max = m_max = int.from_bytes(f.read(4), "little")

            # Rest of file are normalized coefficients
            nC = (m_max + 1) * (2 * n_max - m_max + 2) // 2
            nS = m_max * (2 * n_max - m_max + 1) // 2
            C = np.array(np.frombuffer(f.read(8 * nC)))
            S = np.array(np.frombuffer(f.read(8 * nS)))

        # Assemble index arrays
        Cn = []
        Cm = []
        m = 0
        while len(Cn) < nC:
            Cn.extend(range(m, n_max + 1))
            Cm.extend([m] * (n_max + 1 - m))
            m += 1

        Sn = []
        Sm = []
        m = 1
        while len(Sn) < nS:
            Sn.extend(range(m, n_max + 1))
            Sm.extend([m] * (n_max + 1 - m))
            m += 1

        # De-normalize coefficients
        for i in range(len(C)):
            C[i] *= _invnorm(Cn[i], Cm[i])
        for i in range(len(S)):
            S[i] *= _invnorm(Sn[i], Sm[i])

        # Now form the CS matrix to compactly hold both C and S coefficients.
        # C(n, m) = CS(n, m) and S(n, m) = CS(m-1, n)
        CS = np.empty((n_max + 1, n_max + 1))
        for i in range(len(C)):
            CS[Cn[i], Cm[i]] = C[i]
        for i in range(len(S)):
            CS[Sm[i] - 1, Sn[i]] = S[i]

        # We set the Keplerian term to 0.0 so it's easier to examine the
        # non-central forces directly.
        CS[0, 0] = 0.0

        ret = HarmonicCoefficients.__new__(HarmonicCoefficients)
        ret.name = name
        ret.radius = radius
        ret.MG = MG
        ret.CS = CS
        ret.n_max = n_max
        ret.m_max = m_max
        # print(ret, n_max, m_max)
        return ret

    @classmethod
    def fromTAB(cls, filename, n_max=40, m_max=40):
        """Construct a HarmonicCoefficients object from a .tab file as available
        from https://pgda.gsfc.nasa.gov/products/50
        """
        original_filename = filename
        try:
            filename = find_file(filename, ext=".tab")
        except FileNotFoundError:
            # Reraise with original filename
            raise FileNotFoundError(original_filename)

        # read header
        with open(filename, "r") as f:
            (
                r_ref_km,
                GM_km3_s2,
                dGM_km3_s2,
                n_max1,
                m_max1,
                norm,
                lon_ref_deg,
                lat_ref_deg
            ) = np.array(f.readline().replace(',', ' ').split()).astype(float)
        n_max = min(int(n_max1), n_max)
        m_max = min(int(m_max1), m_max, n_max)

        # read array (TODO: only read relevant rows)
        max_rows = (n_max + 2) * (n_max + 1) // 2 + 2
        degree, order, C, S, dC, dS = np.genfromtxt(
            filename,
            skip_header=1,
            delimiter=',', unpack=True, max_rows=max_rows
        )
        degree = degree.astype(int)
        order = order.astype(int)

        # denormalize
        for i in range(len(C)):
            # Only bother for degree <= n_max, order <= m_max
            dg, od = degree[i], order[i]
            if dg > n_max:
                break  # degree is sorted, so can break here.
            if od > m_max:
                continue
            C[i] *= _invnorm(dg, od)
            S[i] *= _invnorm(dg, od)  # cached

        # Now form the CS matrix to compactly hold both C and S coefficients.
        # C(n, m) = CS(n, m) and S(n, m) = CS(m-1, n)
        CS = np.full((n_max + 1, n_max + 1), np.nan)
        for i in range(len(C)):
            dg, od = degree[i], order[i]
            if dg > n_max:
                break  # degree is sorted, so can break here.
            if od > m_max:
                continue
            CS[dg, od] = C[i]
            if od != 0:
                CS[od - 1, dg] = S[i]

        # Set Keplerian term to 0.0 so it's easier to examine the non-central
        # forces directly.
        CS[0, 0] = 0.0

        ret = HarmonicCoefficients.__new__(HarmonicCoefficients)
        ret.name = original_filename
        ret.radius = r_ref_km * 1e3
        ret.MG = GM_km3_s2 * 1e9
        ret.CS = CS
        ret.n_max = n_max
        ret.m_max = m_max
        return ret

    # TODO
    # def fromGFC(cls, filename):
    #   """Construct a HarmonicCoefficients object from a .gfc file as available
    #   from http://icgem.gfz-potsdam.de
    #   """

    def __hash__(self):
        return hash((
            "HarmonicCoefficients",
            self.name, self.radius, self.MG, self.n_max, self.m_max,
            tuple(self.CS.ravel())
        ))

    def __eq__(self, rhs):
        if not isinstance(rhs, HarmonicCoefficients):
            return False
        return (
            self.name == rhs.name and self.radius == rhs.radius and self.MG == rhs.MG and self.n_max == rhs.n_max and self.m_max == rhs.m_max and np.array_equal(self.CS, rhs.CS)
        )


class AccelThirdBody(_Accel):
    """Acceleration due to a third body.

    Parameters
    ----------
    body : ssapy.Body
        The third body.
    """
    def __init__(self, body):
        super().__init__()
        self.body = body

    def __call__(self, r, v, t, **kwargs):
        """Evaluate acceleration at particular place/moment.

        Parameters
        ----------
        r : array_like, shape(3, )
            Position in meters.
        v : array_like, shape(3, )
            Velocity in meters per second.  Unused.
        t : float
            Time as GPS seconds.  Unused

        Returns
        -------
        accel : array_like, shape(3,)
            Acceleration in meters per second^2
        """
        s = self.body.position(t)
        d = s - r
        return -self.body.mu * (s / norm(s)**3 - d / norm(d)**3)


class AccelHarmonic(_Accel):
    """Acceleration due to a harmonic potential.

    Parameters
    ----------
    body : ssapy.Body
        The body.
    n_max : int
        The maximum degree of the potential.
    m_max : int
        The maximum order of the potential.
    """

    def __init__(self, body, n_max=None, m_max=None):
        super().__init__()
        if n_max is None:
            n_max = body.harmonics.n_max
        if m_max is None:
            m_max = body.harmonics.m_max
        self.body = body
        if n_max > body.harmonics.n_max:
            print(
                f"WARNING::The provided degree ({n_max}) is higher than the maximum value allowed by the "
                f"{body.harmonics.name} model ({body.harmonics.n_max}).\nSetting the degree to "
                f"{body.harmonics.n_max}."
            )
            self.n_max = body.harmonics.n_max
        else:
            self.n_max = n_max
        if m_max > body.harmonics.m_max:
            print(
                f"WARNING::The provided order ({m_max}) is higher than the maximum value allowed by the "
                f"{body.harmonics.name} model ({body.harmonics.m_max}).\nSetting the order to "
                f"{body.harmonics.m_max}."
            )
            self.m_max = body.harmonics.m_max
        else:
            self.m_max = m_max
        if self.m_max > self.n_max:
            print(
                f"WARNING::The provided order ({self.m_max}) is higher than the selected degree "
                f"({self.n_max}).\nSetting the order to {self.n_max}."
            )
            self.m_max = self.n_max
        self._harmonic = _ssapy.AccelHarmonic(
            self.body.mu,
            self.body.harmonics.radius,
            self.body.harmonics.CS.shape[0],
            self.body.harmonics.CS.ctypes.data
        )

    def __call__(self, r, v, t, _E=None, **kwargs):
        """Evaluate acceleration at particular place/moment.

        Parameters
        ----------
        r : array_like, shape(3, )
            Position in meters.
        v : array_like, shape(3, )
            Velocity in meters per second.  Unused.
        t : float
            Time as GPS seconds.  Unused

        Returns
        -------
        accel : array_like, shape(3,)
            Acceleration in meters per second^2
        """
        dr = self.body.position(t)
        if _E is None:
            _E = self.body.orientation(t)
        a = np.empty(3)
        # Transform to body frame
        rin = _E @ (r - dr)
        # Compute acceleration
        self._harmonic.accel(
            self.n_max, self.m_max, rin.ctypes.data, a.ctypes.data
        )
        # Transform back to integration frame
        return _E.T @ a
