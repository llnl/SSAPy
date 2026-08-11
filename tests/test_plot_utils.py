import numpy as np
from astropy.time import Time

from ssapy import plotUtils


class _FakeImage:
    def __init__(self, path):
        self.path = path
        self.resize_size = None

    def resize(self, size):
        self.resize_size = size
        return self


def test_load_texture_files_resize_known_images(monkeypatch):
    opened = []

    monkeypatch.setattr(plotUtils, "find_file", lambda name, ext=None: f"/data/{name}{ext}")

    def fake_open(path):
        opened.append(path)
        return _FakeImage(path)

    monkeypatch.setattr(plotUtils.PILImage, "open", fake_open)

    earth = plotUtils.load_earth_file()
    moon = plotUtils.load_moon_file()

    assert opened == ["/data/earth.png", "/data/moon.png"]
    assert earth.resize_size == (1080, 540)
    assert moon.resize_size == (1080, 540)


def test_draw_earth_and_moon_pass_expected_mesh_inputs(monkeypatch):
    meshes = []

    def fake_plot_mesh(x, y, z, **kwargs):
        meshes.append((x, y, z, kwargs))
        return {"mesh_index": len(meshes)}

    monkeypatch.setattr(plotUtils.ipv, "plot_mesh", fake_plot_mesh)
    monkeypatch.setattr(plotUtils, "load_earth_file", lambda: "earth-texture")
    monkeypatch.setattr(plotUtils, "load_moon_file", lambda: "moon-texture")

    earth_mesh = plotUtils.draw_earth(0.0, ngrid=4, R=2.0, rfactor=3.0)
    moon_mesh = plotUtils.draw_moon(Time(0.0, format="gps"), ngrid=3, R=5.0, rfactor=2.0)
    earth_time_mesh = plotUtils.draw_earth(Time(0.0, format="gps"), ngrid=2)

    assert earth_mesh == {"mesh_index": 1}
    assert moon_mesh == {"mesh_index": 2}
    assert earth_time_mesh == {"mesh_index": 3}

    earth_x, earth_y, earth_z, earth_kwargs = meshes[0]
    assert earth_x.shape == (4, 4)
    assert earth_y.shape == (4, 4)
    assert earth_z.shape == (4, 4)
    assert earth_kwargs["u"].shape == (1, 4, 4)
    assert earth_kwargs["v"].shape == (1, 4, 4)
    assert earth_kwargs["wireframe"] is False
    assert earth_kwargs["texture"] == "earth-texture"
    assert np.max(np.abs(earth_x)) <= 6.0

    moon_x, moon_y, moon_z, moon_kwargs = meshes[1]
    assert moon_x.shape == (3, 3)
    assert moon_y.shape == (3, 3)
    assert moon_z.shape == (3, 3)
    assert moon_kwargs["u"].shape == (1, 3, 3)
    assert moon_kwargs["v"].shape == (1, 3, 3)
    assert moon_kwargs["wireframe"] is False
    assert moon_kwargs["texture"] == "moon-texture"
    assert np.max(np.abs(moon_x)) <= 10.0
