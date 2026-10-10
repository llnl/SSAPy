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
    with pytest.raises(FileNotFoundError, match="git LFS pointer.*ssa-data"):
        utils.find_file("de440s.bsp")


def test_ssapy_data_is_searched_when_the_package_data_is_missing(tmp_path, monkeypatch):
    # With the package copy only a pointer, a real file of the same name
    # shipped by an ssa-data-* package is returned instead.
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


def test_split_data_packages_are_searchable():
    # The four required ssa-data-* packages install with SSAPy.
    import ssapy
    import ssa_data_core, ssa_data_gravity, ssa_data_lunar, ssa_data_lunar_gravity
    assert ssapy.datadir
    assert all(pkg.__file__ for pkg in (ssa_data_core, ssa_data_gravity, ssa_data_lunar, ssa_data_lunar_gravity))


def test_missing_file_names_the_data_package(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "datadir", str(tmp_path))
    monkeypatch.setattr(utils, "_ssapy_data_files", lambda: [])
    with pytest.raises(FileNotFoundError, match="ssa-data"):
        utils.find_file("no_such_file.bsp")


def test_datadir_falls_back_to_a_path_without_data_packages(monkeypatch):
    # With no ssa-data-* package installed, datadir must stay a path so
    # find_file raises FileNotFoundError with install instructions rather
    # than TypeError from os.path.join(None, ...).
    import importlib.resources
    import ssapy

    def _missing(package):
        raise ModuleNotFoundError(package)

    monkeypatch.setattr(importlib.resources, "files", _missing)
    fallback = ssapy._datadir()
    assert isinstance(fallback, str)

    monkeypatch.setattr(utils, "datadir", fallback)
    monkeypatch.setattr(utils, "_ssapy_data_files", lambda: [])
    with pytest.raises(FileNotFoundError, match=r"llnl-ssapy\[all-data\]"):
        utils.find_file("no_such_file.bsp")
