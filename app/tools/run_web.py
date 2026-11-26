r"""Satu perintah untuk menjalankan UI web: bangun frontend bila perlu, lalu sajikan API + `web/dist` di satu port.

Dari root repo (Windows):

    .venv\Scripts\python.exe app\tools\run_web.py                    # backend tiruan (DATA TIRUAN)
    .venv\Scripts\python.exe app\tools\run_web.py --backend cli      # program landcover asli lewat WSL
    app\tools\run_web.cmd --backend cli --data-dir D:\data --open    # pintasan, opsi sama

Yang dilakukan, berurutan:
  1. Frontend (`web/dist`) dibangun dengan `npm run build` bila belum ada atau lebih tua daripada sumbernya
     (`--no-build` melewati, `--rebuild` memaksa). Paket npm TIDAK diunduh di sini: bila `web/node_modules`
     belum ada, skrip berhenti dan menyebut perintahnya (`npm --prefix web install`).
  2. Environment `LC_*` disetel dari opsi (opsi mengalahkan environment yang sudah ada). Untuk `--backend cli`
     di Windows, `LC_CORE_BIN` bawaannya adalah pembungkus `app/tools/lc_wsl.cmd` (biner di WSL; lihat
     `LC_WSL_BIN` di `lc_wsl.py`).
  3. uvicorn menyajikan `/api/*` dan frontend di http://127.0.0.1:8000 (ubah dengan `--port`/`--host`).

Mode pengembangan frontend (hot reload) tidak lewat skrip ini: lihat `web/README.md`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
APP = ROOT / "app"
WSL_WRAPPER = APP / "tools" / "lc_wsl.cmd"

# Berkas yang memengaruhi isi `web/dist`: kalau salah satunya lebih baru daripada `dist/index.html`, bangun ulang.
SOURCE_NAMES = (
    "src", "public", "index.html", "package.json", "package-lock.json", "vite.config.ts",
    "tsconfig.json", "tsconfig.app.json", "tsconfig.node.json", "components.json",
)


def _newest_mtime(paths: Iterable[Path]) -> float:
    newest = 0.0
    for path in paths:
        if path.is_dir():
            for dirpath, _dirs, files in os.walk(path):
                for name in files:
                    newest = max(newest, (Path(dirpath) / name).stat().st_mtime)
        elif path.is_file():
            newest = max(newest, path.stat().st_mtime)
    return newest


def needs_build(web: Path = WEB) -> bool:
    """True bila `dist/index.html` belum ada atau lebih tua daripada sumber frontend."""
    index = web / "dist" / "index.html"
    if not index.is_file():
        return True
    return _newest_mtime(web / name for name in SOURCE_NAMES) > index.stat().st_mtime


def resolve_env(
    backend: str | None,
    data_dir: str | None,
    runs_dir: str | None,
    core_bin: str | None,
    environ: Mapping[str, str],
    platform: str = sys.platform,
) -> dict[str, str]:
    """Nilai `LC_*` yang harus disetel (hanya yang berubah/ditentukan). Opsi mengalahkan environment."""
    chosen = (backend or environ.get("LC_BACKEND") or "mock").strip().lower()
    updates = {"LC_BACKEND": chosen}
    if data_dir:
        updates["LC_DATA_DIR"] = data_dir
    if runs_dir:
        updates["LC_RUNS_DIR"] = runs_dir
    if core_bin:
        updates["LC_CORE_BIN"] = core_bin
    elif chosen == "cli" and not environ.get("LC_CORE_BIN") and platform == "win32":
        updates["LC_CORE_BIN"] = str(WSL_WRAPPER)
    return updates


def build_web(web: Path = WEB) -> None:
    """`npm run build` di `web/`. Berhenti dengan pesan jelas bila npm atau node_modules tidak ada."""
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("run_web: npm tidak ditemukan di PATH (butuh Node.js). Pasang Node di Windows, atau pakai --no-build bila web/dist sudah dibangun.")
    if not (web / "node_modules").is_dir():
        raise SystemExit("run_web: web/node_modules belum ada. Jalankan dulu: npm --prefix web install")
    print("run_web: membangun frontend (npm run build) ...", flush=True)
    result = subprocess.run([npm, "--prefix", str(web), "run", "build"])
    if result.returncode != 0:
        raise SystemExit(f"run_web: npm run build gagal (kode {result.returncode})")


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jalankan UI web (API + frontend) dalam satu proses.")
    p.add_argument("--backend", choices=("mock", "cli"), help="mock (DATA TIRUAN) atau cli (landcover asli); bawaan LC_BACKEND atau mock")
    p.add_argument("--data-dir", help="folder dataset (LC_DATA_DIR; bawaan <repo>/data)")
    p.add_argument("--runs-dir", help="folder hasil run (LC_RUNS_DIR; bawaan <repo>/runs)")
    p.add_argument("--core-bin", help="program landcover atau pembungkus (LC_CORE_BIN)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-build", action="store_true", help="jangan bangun frontend, pakai web/dist apa adanya")
    p.add_argument("--rebuild", action="store_true", help="paksa bangun ulang frontend")
    p.add_argument("--open", action="store_true", help="buka browser setelah server siap")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_build and args.rebuild:
        raise SystemExit("run_web: --no-build dan --rebuild tidak bisa dipakai bersamaan")
    if not args.no_build and (args.rebuild or needs_build()):
        build_web()
    elif not (WEB / "dist" / "index.html").is_file():
        print("run_web: peringatan: web/dist belum ada, hanya API yang disajikan (bangun dulu tanpa --no-build).", file=sys.stderr)
    if not port_free(args.host, args.port):
        raise SystemExit(f"run_web: port {args.port} sudah dipakai; pilih lain dengan --port atau hentikan proses itu")

    updates = resolve_env(args.backend, args.data_dir, args.runs_dir, args.core_bin, os.environ)
    os.environ.update(updates)
    backend = updates["LC_BACKEND"]
    print(f"run_web: backend={backend}" + (f", core={os.environ.get('LC_CORE_BIN', '(bawaan)')}" if backend == "cli" else " (hasil bertanda DATA TIRUAN)"), flush=True)

    import uvicorn  # impor di sini: `api.main` membaca LC_* saat diimpor, jadi env harus sudah disetel

    url = f"http://{args.host}:{args.port}"
    print(f"run_web: buka {url}", flush=True)
    if args.open:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run("api.main:app", app_dir=str(APP), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
