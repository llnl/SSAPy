"""
Classes to solve for Keplerian orbital parameters from different initial inputs

References:
- Shefer, V. A. (2010). New method of orbit determination from two position vectors based on solving Gauss’s equations. Solar System Research, 44, 252-266.
- Montenbruck, O., Gill, E., & Lutze, F. H. (2002). Satellite orbits: models, methods, and applications. Appl. Mech. Rev., 55(2), B27-B28.
"""


import abc
import numpy as np
from astropy.time import Time

from .orbit import Orbit
from .compute import rv
from .constants import EARTH_MU
from .utils import norm, lazy_property, newton_raphson, find_all_zeros


__all__ = [
    "DanchickTwoPosOrbitSolver",
    "GaussTwoPosOrbitSolver",
    "SheferTwoPosOrbitSolver",
    "ThreeAngleOrbitSolver",
    "TwoPosOrbitSolver",
]


class TwoPosOrbitSolver(metaclass=abc.ABCMeta):
    """
    Parameters
    ----------
    r1, r2 : (3,) array_like
        Positions at t1 and t2 in meters.
    t1, t2 : float or astropy.time.Time
        Times of observations.
        If float, then should correspond to GPS seconds; i.e., seconds since
        1980-01-06 00:00:00 UTC
    mu : float, optional
        Gravitational constant of central body in m^3/s^2.  (Default: Earth's
        gravitational constant in WGS84).
    kappaSign : int, optional
        +1 for prograde, -1 for retrograde
    lam : int, optional
        Number of complete orbits between observations.  (Default: 0)
    eps : float, optional
        Iteration tolerance.
    maxiter : int, optional
        Maximum number of iterations.
    """
    def __init__(self, r1, r2, t1, t2, mu=EARTH_MU,
                 kappaSign=1, lam=0, eps=3e-16, maxiter=100):
        if isinstance(t1, Time):
            t1 = t1.gps
        if isinstance(t2, Time):
            t2 = t2.gps
        self.r1 = r1
        self.r2 = r2
        self.t1 = t1
        self.t2 = t2
        self.mu = mu
        self.kappaSign = kappaSign
        self.lam = lam
        self.eps = eps
        self.maxiter = maxiter

        # Preliminary calculations
        self.tau = self.t2 - self.t1

        self.r1mag = norm(self.r1)
        self.r2mag = norm(self.r2)
        self.U1 = self.r1 / self.r1mag
        self.U2 = self.r2 / self.r2mag

        self.r0 = self.r2 - np.dot(self.r2, self.U1) * self.U1  # M&G (2.110)
        self.r0mag = norm(self.r0)
        self.U0 = self.r0 / self.r0mag

        self.cos2f = np.dot(self.U1, self.U2)
        self.rr = np.sqrt(self.r1mag * self.r2mag)
        # Shefer (8)
        self.kappa = self.rr * norm(self.U1 + self.U2) * self.kappaSign
        self.sigma = self.rr * norm(self.U2 - self.U1)  # Shefer (8)
        self.rbar = 0.5 * (self.r1mag + self.r2mag)
        self.m = self.mu * self.tau**2 / self.kappa**3  # Shefer (5.5)
        self.ell = self.rbar / self.kappa - 0.5  # Shefer (5.5)

        # Can solve for the orbital plane immediately
        self.W = np.cross(self.U1, self.U0) * self.kappaSign
        # M&G (2.58)
        self.raan = np.arctan2(self.W[0], -self.W[1]) % (2 * np.pi)
        # M&G (2.58)
        self.i = np.arctan2(self.W[0] / np.sin(self.raan), self.W[2])

        # M&G (2.66)
        self.u1 = np.arctan2(
            self.U1[2],
            -self.U1[0] * self.W[1] + self.U1[1] * self.W[0]
        )

    def _finishOrbit(self, p):
        # M&G (2.114), good for both elliptical and hyperbolic orbits
        ecosnu1 = p / self.r1mag - 1.0
        ecosnu2 = p / self.r2mag - 1.0  # M&G (2.114)
        # M&G (2.116)
        esinnu1 = ((ecosnu1 * self.cos2f - ecosnu2)
                   / (self.r0mag / self.r2mag)
                   * self.kappaSign)
        e = np.hypot(ecosnu1, esinnu1)
        nu1 = np.arctan2(esinnu1, ecosnu1) % (2 * np.pi)  # trueAnomaly at t1
        pa = (self.u1 - nu1) % (2 * np.pi)  # M&G (2.117)
        a = (p / (1 - e**2))  # M&G (2.118)

        return Orbit.fromKeplerianElements(
            a, e, self.i, pa, self.raan, nu1, self.t1, self.mu
        )

    # The main part that varies between methods (i.e., subclasses)
    @abc.abstractmethod
    def _getP(self):
        pass

    def solve(self):
        p = self._getP()
        return self._finishOrbit(p)


class GaussTwoPosOrbitSolver(TwoPosOrbitSolver):
    """
    A class for solving two-position orbit determination problems using 
    the Gauss method. This class extends the `TwoPosOrbitSolver` base class 
    and implements a method to compute the orbital parameter `p` based on 
    Shefer's equations.

    Attributes:
        eps (float): Convergence tolerance for iterative calculations.
        maxiter (int): Maximum number of iterations allowed for convergence.
        m (float): A parameter related to the orbit determination problem.
        ell (float): A parameter related to the orbit determination problem.
        kappa (float): A constant used in orbital calculations.
        sigma (float): A constant used in orbital calculations.
        tau (float): A constant used in orbital calculations.
        mu (float): Standard gravitational parameter.

    Methods:
        _getP():
            Computes the orbital parameter `p` using an iterative approach 
            based on Shefer's equations. This method solves for `eta` 
            iteratively until convergence and then calculates `p` using the 
            converged value of `eta`.
    """
    def __init__(self, *args, **kwargs):
        TwoPosOrbitSolver.__init__(self, *args, **kwargs)

    def _getP(self):
        """Compute p from Shefer (2)."""
        # Solve for eta
        eta = 1
        d_eta = 1
        niter = 0
        while np.abs(d_eta) > self.eps and niter < self.maxiter:
            x = (self.m / eta**2 - self.ell)  # Shefer (5)
            if 1e-15 < x < 1:  # Shefer (6.5ish)
                g = 2 * np.arcsin(np.sqrt(x))
                X = (2 * g - np.sin(2 * g)) / np.sin(g)**3
            elif x < -1e-15:
                h = 2 * np.arcsinh(np.sqrt(-x))
                X = (np.sinh(2 * h) - 2 * h) / np.sinh(h)**3
            else:  # -1e-15<x<1e-15
                X = 4. / 3 + 8. / 5 * x + 64. / 35 * x
            d_eta = 1. + (self.ell + x) * X - eta
            eta += d_eta
            niter += 1
        # Plug eta into (2) to obtain p
        p = (0.5 * eta * self.kappa * self.sigma / self.tau)**2 / self.mu
        return p


class DanchickTwoPosOrbitSolver(TwoPosOrbitSolver):
    """
    A class for solving two-position orbit determination problems using 
    the Danchick method. This class extends the `TwoPosOrbitSolver` base 
    class and implements methods to compute the orbital parameter `p` 
    based on Shefer's equations.

    Attributes:
        eps (float): Convergence tolerance for iterative calculations.
        maxiter (int): Maximum number of iterations allowed for convergence.
        m (float): A parameter related to the orbit determination problem.
        ell (float): A parameter related to the orbit determination problem.
        kappa (float): A constant used in orbital calculations.
        sigma (float): A constant used in orbital calculations.
        tau (float): A constant used in orbital calculations.
        mu (float): Standard gravitational parameter.
        cos2f (float): Cosine of twice the true anomaly, used to determine 
            the branch of the solution.

    Methods:
        X(g):
            Compute the function X(g) as described in Shefer's equation (11).
            This is used in the iterative solution process.

        dXdg(g):
            Compute the derivative of X(g) with respect to g, based on 
            Shefer's equation (12). This is used to refine the solution 
            during iteration.

        _getP():
            Computes the orbital parameter `p` using iterative methods 
            based on Shefer's equations. Depending on the value of `cos2f`, 
            the method selects the appropriate branch of the solution and 
            iteratively solves for `eta` or `x` until convergence. The final 
            value of `p` is calculated using Shefer's equation (2).
    """
    def __init__(self, *args, **kwargs):
        TwoPosOrbitSolver.__init__(self, *args, **kwargs)

    @staticmethod
    def X(g):
        """Compute X(g) from Shefer (11)."""
        return (2 * g - np.sin(2 * g)) / (np.sin(g)**3)

    @staticmethod
    def dXdg(g):
        """Compute dX(g)/dg from Shefer (12)."""
        return (2 * (1 - np.cos(2 * g)) - 3 * (2 * g - np.sin(2 * g)) / np.tan(g)) / (np.sin(g)**3)

    def _getP(self):
        if self.cos2f >= 0:
            # TODO: rewrite to use newton_raphson function.
            eta = 0.5 * (np.sqrt(self.m / (self.ell + 1)) + np.sqrt(self.m / self.ell))
            Geta = 1
            niter = 0
            while np.abs(Geta) > self.eps and niter < self.maxiter:
                x = self.m / eta**2 - self.ell  # Shefer (14)
                if (1 - 2 * x) > 1 or (1 - 2 * x) < -1:
                    raise RuntimeError("Invalid x")
                g = np.arccos(1 - 2 * x)  # Shefer (9)
                dgdx = 2 / np.sin(g)  # Shefer (10)
                dxdeta = -2 * self.m / eta**3  # next few lines are Shefer (14ish)
                dXdeta = self.dXdg(g) * dgdx * dxdeta
                Geta = eta - 1 - (self.ell + x) * self.X(g)
                dGdeta = 1 - self.X(g) * dxdeta - (self.ell + x) * dXdeta
                eta -= Geta / dGdeta
                niter += 1
            x = self.m / eta**2 - self.ell  # Shefer (9)
        elif self.cos2f < 0:
            x = 0.5
            Fx = 1
            niter = 0
            while np.abs(Fx) > self.eps and niter < self.maxiter:
                if 1 - 2 * x > 1 or 1 - 2 * x < -1:
                    raise RuntimeError("Invalid x")
                g = np.arccos(1 - 2 * x)  # Shefer (9)
                dgdx = 2 / np.sin(g)  # Shefer (10)
                eta = 1 + (self.ell + x) * self.X(g)  # Shefer (15)
                detadx = (self.ell + x) * self.dXdg(g) * dgdx + self.X(g)
                Fx = x - self.m / eta**2 + self.ell  # Shefer (15ish)
                dFdx = 1 + (2 * self.m / eta**3) * detadx
                x -= Fx / dFdx
                niter += 1
            eta = 1 + (self.ell + x) * self.X(np.arccos(1 - 2 * x))
        else:
            raise "Invalid value of cos2f"

        # Shefer (2)
        p = (0.5 * eta * self.kappa * self.sigma / self.tau)**2 / self.mu
        return p


class SheferTwoPosOrbitSolver(TwoPosOrbitSolver):
    """
    A class for solving two-position orbit determination problems using 
    Shefer's method. This class extends the `TwoPosOrbitSolver` base class 
    and implements a robust algorithm for determining the orbital parameter 
    `p` (semi-latus rectum) and auxiliary values based on Shefer's equations.

    Attributes:
        robust (bool): If True, enables a robust solution method that examines 
            all possible solutions when the initial guess fails.
        nExam (int): Number of points to examine when searching for zeros of 
            the function F(x) during the robust solution process.
        eps (float): Convergence tolerance for iterative calculations.
        maxiter (int): Maximum number of iterations allowed for convergence.
        m (float): A parameter related to the orbit determination problem.
        ell (float): A parameter related to the orbit determination problem.
        kappa (float): A constant used in orbital calculations.
        sigma (float): A constant used in orbital calculations.
        tau (float): A constant used in orbital calculations.
        mu (float): Standard gravitational parameter.
        lam (float): A parameter related to Shefer's equations.
        rbar (float): A normalized radial distance used in Shefer's calculations.

    Methods:
        alpha(x):
            Compute the alpha(x) function and its derivative based on Shefer's 
            equation (18).

        beta(xi):
            Compute the beta(xi) function and its derivative based on Shefer's 
            equations (A.4) and (A.9).

        Y(x):
            Compute the Y(x) function and its derivative based on Shefer's 
            equation (17).

        Yxi(xi):
            Compute the Y(xi) function using Shefer's equation (A.15).

        X(x):
            Compute the X(x) function and its derivative for elliptical orbits 
            (Shefer 19) and hyperbolic orbits (Shefer 20). Handles special 
            cases for small values of x.

        Z(xi):
            Compute the Z(xi) function and its derivative based on Shefer's 
            equation (A.5).

        D(x):
            Compute the D(x) function and its derivative based on Shefer's 
            equation (43).

        semiMajorAxis(x):
            Compute the semi-major axis a(x) and its derivative based on 
            Shefer's equation (42).

        Flam0(x):
            Compute Flam0(x) and its derivative based on Shefer's equations 
            (21) and (22).

        F(x):
            Compute F(x) and its derivative based on Shefer's equations (40) 
            and (41).

        G(xi):
            Compute G(xi) and its derivative based on Shefer's equations (A.7) 
            and (A.8).

        yPoly(y):
            Compute the polynomial for y and its derivative based on Shefer's 
            equations (44) and (38).

        _getInitialXGuess():
            Compute the initial guess for x using Shefer's step B. Handles 
            both cases where lam is zero and non-zero.

        _getInitialXiGuess():
            Compute the initial guess for xi using Shefer's equation (A.16).

        _getP():
            Compute the semi-latus rectum (p) using Shefer's algorithm. This 
            is the main result of the orbit determination process.

        _getAllP():
            Compute all possible values of p by finding zeros of the function 
            F(x) within a given range.

        _getEta(p):
            Compute the auxiliary value eta defined in Shefer's equation (2).

        solve():
            Solve the orbit determination problem. First attempts to find a 
            solution using Shefer's initial guess. If the solution fails, 
            employs a robust method to examine all possible zeros of F(x) 
            and determine a valid orbit.
    """
    def __init__(self, *args, **kwargs):
        self.robust = kwargs.pop('robust', False)
        self.nExam = kwargs.pop('nExam', 100)
        TwoPosOrbitSolver.__init__(self, *args, **kwargs)

    def alpha(self, x):
        """Evaluate alpha(x) from Shefer (18)."""
        return (self.rbar + self.kappa * (x - 0.5)), self.kappa

    def beta(self, xi):
        """Evaluate beta(xi) and its derivative from Shefer (A.4) and (A.9)."""
        val = self.rbar - 0.5 * self.kappa + xi * (self.rbar + 0.5 * self.kappa)
        grad = self.rbar + 0.5 * self.kappa
        return val, grad

    def Y(self, x):
        """Evaluate Y(x) and dY(x)/dx from Shefer (17)."""
        XVal, XGrad = self.X(x)
        alphaVal, alphaGrad = self.alpha(x)
        val = self.kappa + alphaVal * XVal
        # Evaluate dY(x)/dx using the chain rule
        grad = alphaVal * XGrad + alphaGrad * XVal
        return val, grad

    def Yxi(self, xi):
        betaVal, _ = self.beta(xi)
        return self.kappa + betaVal * self.Z(xi)[0]

    @staticmethod
    def X(x):
        """Evaluate X(x) function from Shefer (19) for elliptical orbits
        and (20) for hyperbolic orbits. The derivative of X(x) is given in
        (23).
        """
        import astropy.units as u
        if isinstance(x, u.Quantity):
            x = x.value
        if x < -1e-15:
            om2x = 1 - 2 * x
            xxm1 = x * (x - 1)
            val = om2x * np.sqrt(xxm1) - np.arcsinh(np.sqrt(-x))
            val /= 2 * (xxm1)**1.5
            grad = 4 - 3 * (om2x) * val
            grad /= -2 * xxm1
            return val, grad
        elif -1e-15 < x < 1e-15:
            grad = 8 / 5 + 64 / 35 * x
            val = grad * x + 4 / 3
            return val, grad
        elif x < 1.:
            om2x = 1 - 2 * x
            x1mx = x * (1 - x)
            val = np.arcsin(np.sqrt(x)) - om2x * np.sqrt(x1mx)
            val /= 2 * x1mx**1.5
            grad = 4 - 3 * om2x * val
            grad /= 2 * x1mx
            return val, grad
        else:  # x > 1.
            return np.inf  # X(x) actually asymptotes to Infinity as x -> 1

    @staticmethod
    def Z(xi):
        """Compute Z(xi) function from Shefer (A.5)."""
        num = (1 + xi)**2 * np.arctanh(np.sqrt(-xi)) - (1 - xi) * np.sqrt(-xi)
        denom = 2 * xi * np.sqrt(-xi)
        val = num / denom
        grad = (4 - (3 - xi) * val) / (2 * xi * (1 + xi))
        return val, grad

    def D(self, x):
        """Compute D(x) and its derivative, dD(x)/dx, from Shefer (43)."""
        aval, agrad = self.semiMajorAxis(x)
        val = self.tau * np.sqrt(self.mu) - 2 * np.pi * self.lam * aval**1.5
        grad = -3 * np.pi * self.lam * np.sqrt(aval) * agrad
        # print("----- D: ", x, aval, agrad, self.tau, self.mu, self.lam)
        return val, grad

    def semiMajorAxis(self, x):
        """Compute semi-major axis a(x) and its derivative, da(x)/dx, from
        Shefer (42).
        """
        assert x >= 0. and x <= 1., \
            "ERROR: invalid x in Shefer semi_major_axis: {}".format(x)
        alphaVal, alphaGrad = self.alpha(x)
        x1mx = x * (1 - x)
        val = alphaVal / (4 * x1mx)
        grad = (self.kappa * x1mx - alphaVal * (1 - 2 * x)) / (4 * x1mx**2)
        return val, grad

    def Flam0(self, x):  # Shefer (21), (22)
        Y, _ = self.Y(x)
        Ysq = Y**2
        alphaVal, _ = self.alpha(x)
        XVal, XGrad = self.X(x)
        val = alphaVal * Ysq - self.mu * self.tau**2
        grad = self.kappa * Ysq + 2 * alphaVal * Y * (self.kappa * XVal + alphaVal * XGrad)
        return val, grad

    def F(self, x):
        """Compute F(x) and dF(x)/dx from Shefer (40) and (41)."""
        if x >= 1.:
            # Should not get here
            val = 1.e16
            grad = 1.e16
        else:
            alphaVal, alphaGrad = self.alpha(x)
            YVal, YGrad = self.Y(x)
            DVal, DGrad = self.D(x)

            val = alphaVal * YVal**2 - DVal**2
            grad = alphaGrad * YVal**2 + alphaVal * 2 * YVal * YGrad - 2 * DVal * DGrad
        return val, grad

    def G(self, xi):
        """Compute G(xi) and its derivative from Shefer (A.7) and (A.8)."""
        betaVal, betaGrad = self.beta(xi)
        Y = self.Yxi(xi)
        Ysq = Y**2
        val = betaVal * Ysq - (1 + xi) * self.mu * self.tau**2
        ZVal, ZGrad = self.Z(xi)
        grad = (betaGrad * Ysq
                + 2 * betaVal * Y * (betaGrad * ZVal + betaVal * ZGrad)
                - self.mu * self.tau**2)
        return val, grad

    def yPoly(self, y):  # Shefer (44) and derivative
        coef = 0.6 * (self.rbar + 1.5 * self.kappa)
        ysqr = y * y
        den = ysqr + self.kappa
        x = (ysqr - self.rbar + 0.5 * self.kappa) / den  # Shefer (36)
        if x > 0 and x < 1 and self.lam > 0:
            dxdy = (self.kappa + 2 * self.rbar) * y / den**2
            Dval, Dgrad = self.D(x)
            val = (ysqr + coef) * y - 1.2 * Dval
            grad = 3 * ysqr + coef - 1.2 * Dgrad * dxdy
        else:
            val = (ysqr + coef) * y - 1.2 * self.tau * np.sqrt(self.mu)  # Shefer (38)
            grad = 3 * ysqr + coef
        return val, grad

    def _getInitialXGuess(self):
        # Shefer step B)
        if self.lam == 0:
            # Shefer (38)
            poly = [1.0, 0.0, 0.6 * (self.rbar + 1.5 * self.kappa),
                    -1.2 * (self.tau * np.sqrt(self.mu))]
            roots = np.roots(poly)
            # There should be exactly one positive real root.
            w = np.where((np.abs(roots.imag) < 3e-16) & (roots.real > 0))
            if len(w) > 1:
                raise RuntimeError(
                    "Found more than one positive, real root!  {}".format(roots)
                )
            if len(w) == 0:
                raise RuntimeError(
                    "Found no positive real roots!  {}".format(roots))
            y = roots[w][0].real
        else:
            y = np.sqrt(2 * self.rbar - self.kappa)
            y = newton_raphson(
                y, self.yPoly, eps=self.eps, maxiter=self.maxiter
            )
        ysqr = y * y
        # Shefer (36)
        x = (ysqr - self.rbar + 0.5 * self.kappa) / (ysqr + self.kappa)
        return x

    def _getInitialXiGuess(self):
        # Shefer (A.16)
        poly = [1.0, 0.0, (3. / 13.) * (self.rbar + 3.5 * self.kappa),
                -(12. / 13.) * (self.tau * np.sqrt(self.mu))]
        roots = np.roots(poly)
        # There should be exactly one positive real root.
        w = np.where((np.abs(roots.imag) < 3e-16) & (roots.real > 0))
        if len(w) > 1:
            raise RuntimeError(
                "Found more than one positive, real root!  {}".format(roots))
        if len(w) == 0:
            raise RuntimeError(
                "Found no positive real roots!  {}".format(roots))
        y = roots[w][0].real
        ysqr = y * y
        # Shefer (A.13)
        xi = (ysqr - self.rbar + 0.5 * self.kappa) / (self.rbar + 0.5 * self.kappa)
        return xi

    def _getP(self):
        """Get the semi-latus rectum. This is the final result of the Shefer algorithm."""
        x = self._getInitialXGuess()
        if x < -1. or x > 1.:
            xi = self._getInitialXiGuess()
            # Shefer step C)
            xi = newton_raphson(xi, self.G, eps=self.eps, maxiter=self.maxiter)
            alpha = self.beta(xi)[0] / (1 + xi)
        else:
            if x < 0.:
                self.lam = 0
            # Shefer step C)
            if self.lam == 0:
                x = newton_raphson(
                    x, self.Flam0, eps=self.eps, maxiter=self.maxiter
                )
            else:
                x = newton_raphson(
                    x, self.F, eps=self.eps, maxiter=self.maxiter
                )
            alpha = self.alpha(x)[0]
        p = self.sigma**2 / (4 * alpha)
        return p

    def _getAllP(self):
        xmin = 1e-15
        xmax = 1.0
        xs = find_all_zeros(lambda x: self.F(x)[0], xmin, xmax, n=self.nExam)
        return [self.sigma**2 / (4 * self.alpha(x)[0]) for x in xs]

    def _getEta(self, p):
        """Compute auxiliary value defined in Shefer (2)."""
        return 2 * np.sqrt(p * self.mu) * self.tau / (self.kappa * self.sigma)

    def solve(self):
        # Try normal solution first.
        orbit = TwoPosOrbitSolver.solve(self)
        if not self.robust:
            return orbit
        r1_test, _ = rv(orbit, self.t1)
        r2_test, _ = rv(orbit, self.t2)
        try:
            # 1 m tolerance...?
            # I'd really like better than this, but probably good enough for a
            # preliminary orbit determination.
            np.testing.assert_allclose(self.r1, r1_test, atol=1, rtol=0)
            np.testing.assert_allclose(self.r2, r2_test, atol=1, rtol=0)
        except AssertionError:
            pass
        else:
            # No AssertionError, so return early
            return orbit
        # Get here when there was an AssertionError above
        # Didn't find a solution with Shefer initial guess, so brute-force find
        # all the zeros of F and see if any of them yield a solution.
        for p in self._getAllP():
            orbit = self._finishOrbit(p)
            r1_test, _ = rv(orbit, self.t1)
            r2_test, _ = rv(orbit, self.t2)
            try:
                np.testing.assert_allclose(self.r1, r1_test, atol=1, rtol=0)
                np.testing.assert_allclose(self.r2, r2_test, atol=1, rtol=0)
            except AssertionError:
                pass
            else:
                return orbit
        # Get here when None of the p's work
        raise RuntimeError("Cannot find orbit")


class ThreeAngleOrbitSolver:
    """Determine orbit of satellite for set of three angle-only observations.

    Might only work well for smallish changes in sector position of
    observations.

    Parameters
    ----------
    e1, e2, e3 : (3,) array_like
        Unit vectors indicating observed directions (dimensionless).
    R1, R2, R3 : (3,) array_like
        Vectors indicating the positions of the observers (in meters).
    t1, t2, t3 : float or astropy.time.Time
        Times of observations.
        If float, then should correspond to GPS seconds; i.e., seconds since
        1980-01-06 00:00:00 UTC
    mu : float, optional
        Gravitational constant of central body in m^3/s^2.  (Default: Earth's
        gravitational constant in WGS84).
    tol : float, optional
        Tolerance to use to stop iteratively improving solution.
        (default 1e-8).
    maxiter : int, optional
        Maximum number of iterations to use.  (default: 100)
    """
    def __init__(self, e1, e2, e3, R1, R2, R3, t1, t2, t3, mu=EARTH_MU,
                 tol=1e-8, maxiter=100):
        # Follows Montenbruck and Gill section 2.4.2
        if isinstance(t1, Time):
            t1 = t1.gps
        if isinstance(t2, Time):
            t2 = t2.gps
        if isinstance(t3, Time):
            t3 = t3.gps
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.mu = mu
        self.tol = tol
        self.maxiter = maxiter

    @lazy_property
    def d1(self):
        return np.cross(self.e2, self.e3)

    @lazy_property
    def d2(self):
        return np.cross(self.e3, self.e1)

    @lazy_property
    def d3(self):
        return np.cross(self.e1, self.e2)

    @lazy_property
    def D11(self):
        return np.dot(self.d1, self.R1)

    @lazy_property
    def D12(self):
        return np.dot(self.d1, self.R2)

    @lazy_property
    def D13(self):
        return np.dot(self.d1, self.R3)

    @lazy_property
    def D21(self):
        return np.dot(self.d2, self.R1)

    @lazy_property
    def D22(self):
        return np.dot(self.d2, self.R2)

    @lazy_property
    def D23(self):
        return np.dot(self.d2, self.R3)

    @lazy_property
    def D31(self):
        return np.dot(self.d3, self.R1)

    @lazy_property
    def D32(self):
        return np.dot(self.d3, self.R2)

    @lazy_property
    def D33(self):
        return np.dot(self.d3, self.R3)

    @lazy_property
    def D(self):
        return np.dot(self.e1, self.d1)

    @lazy_property
    def t21(self):
        return self.t2 - self.t1

    @lazy_property
    def t31(self):
        return self.t3 - self.t1

    @lazy_property
    def t32(self):
        return self.t3 - self.t2

    @lazy_property
    def t3231(self):
        return self.t32 / self.t31

    @lazy_property
    def t2131(self):
        return self.t21 / self.t31

    def _getRho(self, n1, n3):
        """Compute three unknown station-satellite distances from Montenbruck and Gill (2.129).
        """
        rho1 = -1 / (n1 * self.D) * (n1 * self.D11 - self.D12 + n3 * self.D13)
        rho2 = 1 / self.D * (n1 * self.D21 - self.D22 + n3 * self.D23)
        rho3 = -1 / (n3 * self.D) * (n1 * self.D31 - self.D32 + n3 * self.D33)
        return rho1, rho2, rho3

    def _getR(self, rho1, rho2, rho3):
        """Compute three Earth-satellite position vectors from Montenbruck and Gill (2.122).
        """
        r1 = self.R1 + rho1 * self.e1
        r2 = self.R2 + rho2 * self.e2
        r3 = self.R3 + rho3 * self.e3
        return r1, r2, r3

    def _getN(self, eta21, eta23):
        """Compute n1 and n3 terms from Montenbruck and Gill (2.132)."""
        n1 = eta21 * self.t3231
        n3 = eta23 * self.t2131
        return n1, n3

    def _getEta(self, r1, r2, t1, t2):
        """Use Shefer algorithm to improve eta estimates."""
        solver = SheferTwoPosOrbitSolver(r1, r2, t1, t2)
        p = solver._getP()
        eta = solver._getEta(p)
        return eta

    def solve(self):
        # initial estimate
        eta21 = eta23 = 1.0
        n1, n3 = self._getN(eta21, eta23)
        # iterate to improve eta21, eta23
        dn1 = np.inf
        dn3 = np.inf
        niter = 0
        while ((np.abs(dn1) > self.tol or np.abs(dn3) > self.tol)
               and niter < self.maxiter):
            rho1, rho2, rho3 = self._getRho(n1, n3)
            r1, r2, r3 = self._getR(rho1, rho2, rho3)
            eta1 = self._getEta(r2, r3, self.t2, self.t3)
            eta2 = self._getEta(r1, r3, self.t1, self.t3)
            eta3 = self._getEta(r1, r2, self.t1, self.t2)
            eta21 = eta2 / eta1
            eta23 = eta2 / eta3
            newn1, newn3 = self._getN(eta21, eta23)
            dn1, n1 = newn1 - n1, newn1
            dn3, n3 = newn3 - n3, newn3
            niter += 1
        solver = SheferTwoPosOrbitSolver(r1, r3, self.t1, self.t3)
        return solver.solve()
