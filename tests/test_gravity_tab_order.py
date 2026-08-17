import numpy as np

import ssapy.gravity as gravity


def test_from_tab_caps_order_at_truncated_degree(tmp_path, monkeypatch):
    model = tmp_path / 'test.tab'
    model.write_text(
        '1738.0,4902.8,0.0,2,2,1,0,0\n'
        '0,0,1,0,0,0\n'
        '1,0,0,0,0,0\n'
        '1,1,0,0,0,0\n'
        '2,0,0,0,0,0\n'
        '2,1,0,0,0,0\n'
        '2,2,0,0,0,0\n'
    )
    monkeypatch.setattr(
        gravity,
        'find_file',
        lambda filename, ext=None: str(model),
    )

    coefficients = gravity.HarmonicCoefficients.fromTAB(
        'test', n_max=1, m_max=2
    )

    assert coefficients.n_max == 1
    assert coefficients.m_max == 1
    assert coefficients.m_max <= coefficients.n_max
    assert coefficients.CS.shape == (2, 2)
    assert np.isfinite(coefficients.CS[1, 1])
