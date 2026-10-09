import os

import pytest

from ssapy import utils

POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n"
    "size 119741440\n"
)


def test_lfs_pointer_is_not_returned_as_data(tmp_path, monkeypatch):
    # A clone without `git lfs pull` holds 130-byte pointer files in place of
    # the kernels; find_file used to return them, and jplephem then failed on
    # an unreadable "kernel". It must now raise a FileNotFoundError that says
    # what the file is (git LFS spec v1 pointer format).
    (tmp_path / "de440s.bsp").write_text(POINTER)
    monkeypatch.setattr(utils, "datadir", str(tmp_path))
    monkeypatch.setattr(utils, "_ssapy_data_files", lambda: [])
    with pytest.raises(FileNotFoundError, match="git LFS pointer.*llnl-ssapy-data"):
        utils.find_file("de440s.bsp")


def test_ssapy_data_is_searched_when_the_package_data_is_missing(tmp_path, monkeypatch):
    # With the package copy only a pointer, a real file of the same name
    # shipped by llnl-ssapy-data is returned instead.
    local = tmp_path / "local"
    shipped = tmp_path / "shipped" / "ephemerides"
    local.mkdir()
    shipped.mkdir(parents=True)
    (local / "de440s.bsp").write_text(POINTER)
    real = shipped / "de440s.bsp"
    real.write_bytes(b"DAF/SPK " + b"\0" * 2048)
    monkeypatch.setattr(utils, "datadir", str(local))
    monkeypatch.setattr(utils, "_ssapy_data_files", lambda: [str(real)])
    assert os.path.samefile(utils.find_file("de440s.bsp"), real)


def test_datadir_is_the_ssapy_tree_of_llnl_ssapy_data():
    # SSAPy no longer ships data: datadir is ssapy/ inside llnl-ssapy-data, and
    # every file the body and gravity loaders need resolves there.
    import ssapy
    import ssapy_data
    from ssapy.body import _planetary_ephemeris_path

    assert os.path.samefile(ssapy.datadir, os.path.join(os.path.dirname(ssapy_data.__file__), "data", "ssapy"))
    for name in ("moon_pa_de440_200625.bpc", "egm84.egm", "egm2008.egm.cof", "gggrx_1200a_sha.tab", "earth.png", "moon.png"):
        assert os.path.dirname(utils.find_file(name)) == os.path.normpath(ssapy.datadir)
    assert os.path.basename(_planetary_ephemeris_path()) == "de440s.bsp"


def test_missing_file_names_the_data_package(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "datadir", str(tmp_path))
    monkeypatch.setattr(utils, "_ssapy_data_files", lambda: [])
    with pytest.raises(FileNotFoundError, match="llnl-ssapy-data"):
        utils.find_file("no_such_file.bsp")
