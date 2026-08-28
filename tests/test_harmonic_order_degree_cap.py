from types import SimpleNamespace

import numpy as np

import ssapy.gravity as gravity


def test_harmonic_order_is_capped_at_selected_degree(monkeypatch):
    captured = {}

    class NativeHarmonic:
        def __init__(self, mu, radius, size, ptr):
            captured["init"] = (mu, radius, size, ptr)

    monkeypatch.setattr(gravity._ssapy, "AccelHarmonic", NativeHarmonic)

    harmonics = SimpleNamespace(
        n_max=8,
        m_max=8,
        name="synthetic",
        radius=2.0,
        CS=np.zeros((9, 9), dtype=float),
    )
    body = SimpleNamespace(mu=1.0, radius=2.0, harmonics=harmonics)

    accel = gravity.AccelHarmonic(body, n_max=3, m_max=7)

    assert accel.n_max == 3
    assert accel.m_max == 3
    assert captured["init"][2] == 9
