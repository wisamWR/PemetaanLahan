"""Tes alat bantu `app/tools/` (logika murni): terjemahan path pembungkus WSL dan pemilihan environment/build `run_web`.

Modul dimuat lewat path berkas (bukan `import tools.*`) supaya tidak bergantung pada nama paket `tools`.
Tidak ada yang memanggil `wsl.exe` atau `npm`."""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"lc_tool_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lc_wsl = _load("lc_wsl")
run_web = _load("run_web")


# --- lc_wsl ------------------------------------------------------------------------------------ #

@pytest.mark.parametrize("arg, expected", [
    ("C:\\Users\\Ridlo\\data\\a.png", "/mnt/c/Users/Ridlo/data/a.png"),
    ("c:/x/y", "/mnt/c/x/y"),
    ("D:\\! Kuliah\\Semester 5 - UGM\\projek\\runs\\core", "/mnt/d/! Kuliah/Semester 5 - UGM/projek/runs/core"),
    ("--mode", "--mode"),
    ("8", "8"),
    ("/home/ridlo/data/a.png", "/home/ridlo/data/a.png"),
    ("data/a.png", "data/a.png"),
    ("C:", "C:"),
    ("C:relatif", "C:relatif"),
    ("--out=C:\\x", "--out=C:\\x"),                       # hanya argumen utuh yang diterjemahkan
    ("", ""),
])
def test_to_wsl_path(arg, expected):
    assert lc_wsl.to_wsl_path(arg) == expected


def test_perintah_bawaan_memperluas_home_lewat_bash_dan_meneruskan_argumen_apa_adanya():
    cmd = lc_wsl.build_wsl_command(["--mode", "cpu", "--input", "C:\\a b\\x.png", "--k", "8"], {})
    assert cmd[:4] == ["wsl.exe", "-e", "bash", "-c"]
    assert cmd[4] == 'exec "$HOME/PemetaanLahan/build/landcover" "$@"'
    assert cmd[5] == "landcover"                                      # $0 untuk bash -c
    assert cmd[6:] == ["--mode", "cpu", "--input", "/mnt/c/a b/x.png", "--k", "8"]


def test_lc_wsl_bin_dan_distro_dipakai():
    env = {"LC_WSL_BIN": "/opt/lc/landcover", "LC_WSL_DISTRO": "Ubuntu-24.04"}
    assert lc_wsl.build_wsl_command(["--info"], env) == ["wsl.exe", "-d", "Ubuntu-24.04", "-e", "/opt/lc/landcover", "--info"]


def test_nilai_env_kosong_atau_spasi_diabaikan():
    cmd = lc_wsl.build_wsl_command(["--info"], {"LC_WSL_BIN": "  ", "LC_WSL_DISTRO": ""})
    assert "-d" not in cmd and cmd[2] == "bash"


def test_dryrun_mencetak_perintah_tanpa_menjalankan(monkeypatch, capsys):
    monkeypatch.setenv("LC_WSL_DRYRUN", "1")
    monkeypatch.setattr(lc_wsl.subprocess, "call", lambda *a, **k: pytest.fail("tidak boleh menjalankan proses"))
    assert lc_wsl.main(["--info"]) == 0
    assert "wsl.exe" in capsys.readouterr().out


def test_kode_keluar_diteruskan(monkeypatch):
    monkeypatch.delenv("LC_WSL_DRYRUN", raising=False)
    seen = {}
    monkeypatch.setattr(lc_wsl.subprocess, "call", lambda cmd: seen.setdefault("cmd", cmd) and 3)
    assert lc_wsl.main(["--mode", "opt"]) == 3
    assert seen["cmd"][0] == "wsl.exe"


def test_wsl_tidak_ada_memberi_kode_127_dan_pesan(monkeypatch, capsys):
    monkeypatch.delenv("LC_WSL_DRYRUN", raising=False)

    def boom(cmd):
        raise FileNotFoundError

    monkeypatch.setattr(lc_wsl.subprocess, "call", boom)
    assert lc_wsl.main(["--info"]) == 127
    assert "wsl.exe tidak ditemukan" in capsys.readouterr().err


# --- run_web ----------------------------------------------------------------------------------- #

def test_resolve_env_bawaan_mock():
    assert run_web.resolve_env(None, None, None, None, {}) == {"LC_BACKEND": "mock"}


def test_resolve_env_opsi_mengalahkan_environment():
    got = run_web.resolve_env("cli", "D:\\d", "D:\\r", "D:\\bin.cmd", {"LC_BACKEND": "mock", "LC_CORE_BIN": "x"}, "win32")
    assert got == {"LC_BACKEND": "cli", "LC_DATA_DIR": "D:\\d", "LC_RUNS_DIR": "D:\\r", "LC_CORE_BIN": "D:\\bin.cmd"}


def test_resolve_env_backend_dari_environment():
    assert run_web.resolve_env(None, None, None, None, {"LC_BACKEND": " CLI "}, "linux")["LC_BACKEND"] == "cli"


def test_cli_di_windows_memakai_pembungkus_wsl_kecuali_sudah_ditentukan():
    got = run_web.resolve_env("cli", None, None, None, {}, "win32")
    assert got["LC_CORE_BIN"] == str(run_web.WSL_WRAPPER) and run_web.WSL_WRAPPER.is_file()
    assert "LC_CORE_BIN" not in run_web.resolve_env("cli", None, None, None, {"LC_CORE_BIN": "C:\\mine.cmd"}, "win32")
    assert "LC_CORE_BIN" not in run_web.resolve_env("cli", None, None, None, {}, "linux")
    assert "LC_CORE_BIN" not in run_web.resolve_env("mock", None, None, None, {}, "win32")


def _make_web(tmp_path: Path) -> Path:
    web = tmp_path / "web"
    (web / "src").mkdir(parents=True)
    (web / "src" / "App.tsx").write_text("x")
    (web / "index.html").write_text("x")
    (web / "package.json").write_text("{}")
    return web


def _touch(path: Path, when: float) -> None:
    os.utime(path, (when, when))


def test_needs_build_tanpa_dist(tmp_path):
    assert run_web.needs_build(_make_web(tmp_path)) is True


def test_needs_build_dist_lebih_baru_dari_sumber(tmp_path):
    web = _make_web(tmp_path)
    (web / "dist").mkdir()
    index = web / "dist" / "index.html"
    index.write_text("x")
    now = time.time()
    for f in (web / "src" / "App.tsx", web / "index.html", web / "package.json"):
        _touch(f, now - 100)
    _touch(index, now)
    assert run_web.needs_build(web) is False


def test_needs_build_sumber_lebih_baru(tmp_path):
    web = _make_web(tmp_path)
    (web / "dist").mkdir()
    index = web / "dist" / "index.html"
    index.write_text("x")
    now = time.time()
    _touch(index, now - 100)
    _touch(web / "src" / "App.tsx", now)
    assert run_web.needs_build(web) is True


def test_needs_build_berkas_sumber_baru_di_subfolder(tmp_path):
    web = _make_web(tmp_path)
    (web / "src" / "features").mkdir()
    (web / "dist").mkdir()
    index = web / "dist" / "index.html"
    index.write_text("x")
    now = time.time()
    for f in (web / "src" / "App.tsx", web / "index.html", web / "package.json"):
        _touch(f, now - 100)
    _touch(index, now - 50)
    new = web / "src" / "features" / "Baru.tsx"
    new.write_text("x")
    _touch(new, now)
    assert run_web.needs_build(web) is True


def test_build_web_berhenti_jelas_tanpa_npm_atau_node_modules(tmp_path, monkeypatch):
    web = _make_web(tmp_path)
    monkeypatch.setattr(run_web.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="npm tidak ditemukan"):
        run_web.build_web(web)
    monkeypatch.setattr(run_web.shutil, "which", lambda name: "npm")
    with pytest.raises(SystemExit, match="npm --prefix web install"):
        run_web.build_web(web)                                         # node_modules belum ada


def test_opsi_bertentangan_ditolak():
    with pytest.raises(SystemExit, match="tidak bisa dipakai bersamaan"):
        run_web.main(["--no-build", "--rebuild"])


def test_port_free_mendeteksi_port_terpakai():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        assert run_web.port_free("127.0.0.1", port) is False
    assert run_web.port_free("127.0.0.1", port) is True
