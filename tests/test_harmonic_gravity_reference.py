"""AccelHarmonic against an independent spherical-harmonic potential.

The coefficient files are parsed here without SSAPy's loaders, the potential
U = GM/r sum_n (R/r)^n sum_m Pbar_nm(sin phi) (C cos m lam + S sin m lam) is
built from scipy's associated Legendre functions with the 4-pi normalisation,
and its gradient is taken by central differences. The 20x20 accelerations
agree with AccelHarmonic to 1e-8 relative (finite-difference limited,
measured 1.3e-9 to 2.7e-9). Before AccelHarmonic used each model's own GM,
GRGM1200A lunar accelerations were 2.2e-7 low.
"""
from math import factorial

import numpy as np
import pytest
from scipy.special import lpmv

from ssapy.body import get_body
from ssapy.gravity import AccelHarmonic
from ssapy.utils import find_file

DEGREE = 20
T = 1.4e9


def _read_egm(name):
    egm = find_file(name, ext=".egm")
    header = open(egm).read().splitlines()
    radius = float(next(line for line in header if line.startswith("ModelRadius"))[11:])
    gm = float(next(line for line in header if line.startswith("ModelMass"))[9:])
    with open(egm + ".cof", "rb") as f:
        f.read(8)
        n_max = int.from_bytes(f.read(4), "little")
        m_max = int.from_bytes(f.read(4), "little")
        n_c = (m_max + 1) * (2 * n_max - m_max + 2) // 2
        n_s = m_max * (2 * n_max - m_max + 1) // 2
        c_values = np.frombuffer(f.read(8 * n_c))
        s_values = np.frombuffer(f.read(8 * n_s))
    c, s, k = {}, {}, 0
    for m in range(m_max + 1):
        for n in range(m, n_max + 1):
            c[(n, m)] = c_values[k]
            k += 1
    k = 0
    for m in range(1, m_max + 1):
        for n in range(m, n_max + 1):
            s[(n, m)] = s_values[k]
            k += 1
    return c, s, radius, gm


def _read_tab(name):
    with open(find_file(name)) as f:
        header = [float(x) for x in f.readline().replace(",", " ").split()]
        c, s = {}, {}
        for line in f:
            fields = line.replace(",", " ").split()
            n, m = int(fields[0]), int(fields[1])
            if n > DEGREE:
                break
            c[(n, m)], s[(n, m)] = float(fields[2]), float(fields[3])
    return c, s, header[0] * 1e3, header[1] * 1e9


def _potential(r, c, s, radius, gm):
    rr = np.linalg.norm(r)
    x = r[2] / rr
    lam = np.arctan2(r[1], r[0])
    total = 0.0
    for n in range(1, DEGREE + 1):
        scale = (radius / rr) ** n
        for m in range(n + 1):
            norm = np.sqrt((2 - (m == 0)) * (2 * n + 1) * factorial(n - m) / factorial(n + m))
            pbar = norm * (-1) ** m * lpmv(m, n, x)  # remove scipy's Condon-Shortley phase
            total += scale * pbar * (c.get((n, m), 0.0) * np.cos(m * lam) + s.get((n, m), 0.0) * np.sin(m * lam))
    return gm / rr * total


def _gradient(r, *model, h=0.5):
    return np.array([(_potential(r + h * e, *model) - _potential(r - h * e, *model)) / (2 * h) for e in np.eye(3)])


@pytest.mark.parametrize("model", ["egm96", "egm2008"])
def test_earth_harmonics_match_an_independent_potential(model):
    earth = get_body("earth", model=model.upper())
    coefficients = _read_egm(model)
    rotation = np.asarray(earth.orientation(T))
    for r_body in (np.array([7000e3, 1000e3, 500e3]), np.array([-2000e3, 3000e3, 6000e3])):
        expected = _gradient(r_body, *coefficients)
        got = rotation @ AccelHarmonic(earth, DEGREE, DEGREE)(rotation.T @ r_body, np.zeros(3), T)
        assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 1e-8


def test_moon_harmonics_match_an_independent_potential():
    moon = get_body("moon")
    coefficients = _read_tab("gggrx_1200a_sha.tab")
    rotation = np.asarray(moon.orientation(T))
    moon_position = np.asarray(moon.position(T)).reshape(3)
    for r_body in (np.array([1900e3, 300e3, -200e3]), np.array([-500e3, 800e3, 1700e3])):
        expected = _gradient(r_body, *coefficients)
        got = rotation @ AccelHarmonic(moon, DEGREE, DEGREE)(moon_position + rotation.T @ r_body, np.zeros(3), T)
        assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 1e-8
