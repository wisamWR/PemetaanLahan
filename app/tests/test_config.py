from pathlib import Path

import pytest

import config

ENV_VARS = ("LC_BACKEND", "LC_DATA_DIR", "LC_RUNS_DIR", "LC_CORE_BIN")


@pytest.fixture(autouse=True)
def bersih_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_nilai_bawaan_relatif_ke_root_repo():
    cfg = config.load_config()
    assert cfg.backend == "mock"
    assert cfg.data_dir == config.ROOT / "data"
    assert cfg.runs_dir == config.ROOT / "runs"
    assert cfg.core_bin == config.ROOT / "build" / "landcover"


def test_root_adalah_root_repo():
    assert (config.ROOT / "app" / "config.py").is_file()


def test_env_menimpa_semua_nilai(monkeypatch, tmp_path):
    monkeypatch.setenv("LC_BACKEND", "CLI")  # huruf besar tetap diterima
    monkeypatch.setenv("LC_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("LC_RUNS_DIR", str(tmp_path / "r"))
    monkeypatch.setenv("LC_CORE_BIN", str(tmp_path / "bin" / "landcover"))
    cfg = config.load_config()
    assert cfg.backend == "cli"
    assert cfg.data_dir == tmp_path / "d"
    assert cfg.runs_dir == tmp_path / "r"
    assert cfg.core_bin == tmp_path / "bin" / "landcover"


def test_tilde_diperluas(monkeypatch):
    monkeypatch.setenv("LC_CORE_BIN", "~/PemetaanLahan/build/landcover")
    assert config.load_config().core_bin == Path.home() / "PemetaanLahan" / "build" / "landcover"


def test_env_kosong_dianggap_tidak_diset(monkeypatch):
    monkeypatch.setenv("LC_BACKEND", "")
    monkeypatch.setenv("LC_DATA_DIR", "")
    cfg = config.load_config()
    assert cfg.backend == "mock"
    assert cfg.data_dir == config.ROOT / "data"


def test_backend_tidak_dikenal_ditolak(monkeypatch):
    monkeypatch.setenv("LC_BACKEND", "gpu")
    with pytest.raises(ValueError, match="LC_BACKEND"):
        config.load_config()
