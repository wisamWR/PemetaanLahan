r"""Pembungkus WSL untuk program `landcover`: dipanggil dari Windows, program jalan di dalam WSL.

`CliLauncher` memanggil `LC_CORE_BIN` dengan argumen berbentuk path Windows (`C:\...`). Pembungkus ini
mengubah tiap argumen yang diawali huruf drive menjadi path WSL (`/mnt/c/...`), lalu meneruskan semuanya ke
biner `landcover` di WSL lewat `wsl.exe`. Keluaran (stdout, stderr) mengalir langsung dan kode keluar
diteruskan, jadi progres tampil per baris dan galat program (kode 1, 2, 3) sampai ke launcher apa adanya.

Pengaturan (environment variable, semuanya opsional):

    LC_WSL_BIN      path biner DI DALAM WSL (bawaan: $HOME/PemetaanLahan/build/landcover)
    LC_WSL_DISTRO   nama distro bila bukan distro bawaan (mis. Ubuntu-24.04)
    LC_WSL_DRYRUN   bila 1: cetak perintah wsl.exe yang akan dijalankan, tidak menjalankan apa pun

Batas yang diketahui: mematikan pembungkus ini (mis. karena batas waktu `CliLauncher`) TIDAK mematikan
proses `landcover` di dalam WSL; proses itu selesai sendiri. Data dan folder run di `/mnt/*` (disk Windows)
jauh lebih lambat daripada filesystem Linux WSL; untuk pengukuran waktu, taruh di `~/...` (lihat README).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Mapping, Sequence

DEFAULT_BIN = "$HOME/PemetaanLahan/build/landcover"

_DRIVE = re.compile(r"^([A-Za-z]):[\\/](.*)$", re.DOTALL)


def to_wsl_path(arg: str) -> str:
    """`C:\\a\\b` atau `C:/a/b` menjadi `/mnt/c/a/b`; argumen lain (flag, angka, path Linux) tetap."""
    match = _DRIVE.match(arg)
    if match is None:
        return arg
    drive, rest = match.groups()
    return f"/mnt/{drive.lower()}/{rest.replace(chr(92), '/')}"


def build_wsl_command(args: Sequence[str], env: Mapping[str, str] | None = None) -> list[str]:
    """Perintah `wsl.exe` lengkap untuk `args` (argumen `landcover`, path Windows boleh)."""
    env = os.environ if env is None else env
    converted = [to_wsl_path(a) for a in args]
    cmd = ["wsl.exe"]
    distro = (env.get("LC_WSL_DISTRO") or "").strip()
    if distro:
        cmd += ["-d", distro]
    binary = (env.get("LC_WSL_BIN") or "").strip()
    if binary:
        cmd += ["-e", binary, *converted]
    else:
        # bash hanya untuk memperluas $HOME pada path bawaan; argumen lain lewat "$@" apa adanya
        cmd += ["-e", "bash", "-c", f'exec "{DEFAULT_BIN}" "$@"', "landcover", *converted]
    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = build_wsl_command(args)
    if os.environ.get("LC_WSL_DRYRUN") == "1":
        print(" ".join(repr(c) for c in cmd))
        return 0
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("lc_wsl: wsl.exe tidak ditemukan (WSL belum terpasang atau bukan Windows)", file=sys.stderr)
        return 127


if __name__ == "__main__":
    sys.exit(main())
