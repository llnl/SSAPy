from types import SimpleNamespace

import numpy as np

import ssapy.gravity as gravity


def test_accel_harmonic_uses_coefficient_reference_radius(monkeypatch):
    captured = {}

    class FakeAccelHarmonic:
        def __init__(self, mu, radius, ncol, cs_ptr):
            captured['mu'] = mu
            captured['radius'] = radius
            captured['ncol'] = ncol
            captured['cs_ptr'] = cs_ptr

    monkeypatch.setattr(gravity._ssapy, 'AccelHarmonic', FakeAccelHarmonic)

    coefficients = SimpleNamespace(
        name='test-gravity-model',
        radius=1_738_000.0,
        n_max=2,
        m_max=2,
        CS=np.zeros((3, 3), dtype=float),
    )
    body = SimpleNamespace(
        mu=4.9048695e12,
        radius=1_737_400.0,
        harmonics=coefficients,
    )

    gravity.AccelHarmonic(body)

    assert captured['mu'] == body.mu
    assert captured['radius'] == coefficients.radius
    assert captured['radius'] != body.radius
    assert captured['ncol'] == coefficients.CS.shape[0]
