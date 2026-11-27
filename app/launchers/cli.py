"""CliLauncher: menjalankan program `landcover` (C++, Langkah 11a) lewat subprocess (plan Langkah 11).

Launcher hanya memanggil biner dan membaca folder run yang ditulisnya; `counts`, pratinjau, dan
`labels.u8` sudah dihitung oleh program itu. Semua format flag ada di `build_command()` (satu-satunya
tempat); semua pembacaan format folder run ada di `landcover.contract`.

Urutan `CliLauncher.run`:
  1. dataset di-resolve (`--input` = PATH, bukan nama), biner dicari, folder keluaran diperiksa;
  2. berkas run lama di `out_dir` dibuang (`meta.json` DULUAN): run yang gagal tidak boleh tampak
     valid, dan `labels.u8`/`verification.json`/`map.png` basi tidak boleh dibaca sebagai milik
     run baru. Biner melakukan hal yang sama setelah memvalidasi masukannya (pertahanan berlapis);
  3. biner dijalankan; stdout diteruskan per baris ke `on_progress`, stderr ditampung;
  4. kode keluar 1/2/3 (dan yang lain) menjadi `CoreFailedError` dengan stderr, batas waktu menjadi
     `CoreTimeoutError`;
  5. kode 0: folder dibaca (`read_run`), diperiksa (`validate`) dan dicocokkan dengan parameter;
     masalah apa pun menjadi `CoreOutputError` (tidak pernah diam-diam).

`landcover --info` (`probe_core`) memberi tahu mesin mana yang terhubung pada build ini; UI memakainya
untuk menyembunyikan opsi OpenMP/CUDA yang tidak ada.

Variabel lingkungan: `LC_CORE_TIMEOUT` (detik, bawaan 1800) = batas waktu satu run.
Batasan: batas waktu mematikan proses yang dijalankan (dan turunannya di POSIX / lewat `taskkill`
di Windows). Bila `LC_CORE_BIN` adalah pembungkus ke WSL (`wsl.exe`), proses `landcover` di dalam
WSL bisa tetap berjalan sampai selesai sendiri.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from landcover.contract import ContractError, RunResult, read_run, validate
from landcover.datasets import resolve_dataset

from .base import KMEANS_CHOICES, ProgressCallback, RunParams

TIMEOUT_ENV = "LC_CORE_TIMEOUT"
DEFAULT_TIMEOUT_S = 1800.0           # satu tile penuh di mesin uji ~1 menit; sisanya cadangan
INFO_TIMEOUT_S = 15.0                # `landcover --info` selesai dalam milidetik
JOIN_TIMEOUT_S = 10.0                # tunggu pembaca pipa setelah proses mati
STDERR_KEEP_CHARS = 32_768           # stderr yang ditampung (ekor); sisanya dibuang
DETAIL_CHARS = 4_000                 # ekor stderr yang ditampilkan di pesan galat
LINE_CHARS = 400                     # satu baris progres dipotong sebesar ini
READ_CHUNK = 8_192

EXIT_RUNTIME, EXIT_USAGE, EXIT_ENGINE = 1, 2, 3

# Berkas yang ditulis `landcover` ke folder run (sama dengan `kRunFiles` di landcover.cpp).
# `meta.json` DULUAN: tanda "run lengkap", jadi folder yang gagal dibersihkan tidak tampak valid.
RUN_FILES = (
    "meta.json", "timing.json", "centroids.json", "counts.json", "labels_preview.png",
    "map_preview.png", "map.png", "labels.u8", "verification.json",
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


# --- galat ---------------------------------------------------------------- #

class CliLauncherError(RuntimeError):
    """Run lewat `landcover` gagal. `summary` = satu kalimat untuk judul; `detail` = rincian
    (stderr atau daftar masalah) yang boleh ditampilkan di kotak kode; `exit_code` bila ada."""

    def __init__(self, summary: str, *, detail: str = "", exit_code: int | None = None):
        self.summary = summary
        self.detail = detail
        self.exit_code = exit_code
        super().__init__(f"{summary}\n{detail}" if detail else summary)


class CoreNotFoundError(CliLauncherError):
    """Biner `landcover` tidak ada, tidak bisa dijalankan, atau perintahnya tak bisa dimulai."""


class CoreFailedError(CliLauncherError):
    """Biner keluar dengan kode bukan 0 (1 = galat jalan, 2 = argumen, 3 = mesin tidak terhubung)."""


class CoreTimeoutError(CliLauncherError):
    """Biner melewati batas waktu dan dihentikan."""


class CoreOutputError(CliLauncherError):
    """Biner keluar 0 tetapi folder run tidak lengkap, tidak valid, atau tidak cocok dengan parameter."""


# --- kemampuan build (`--info`) -------------------------------------------- #

@dataclass(frozen=True)
class CoreInfo:
    """Isi `landcover --info` yang dipakai UI."""

    cpu: str | None                       # "kmeans_cpu" | "ref_kmeans" | None (tidak ada mesin CPU)
    omp: bool                             # kmeans_omp terhubung
    cuda: bool                            # kmeans_cuda terhubung
    openmp: bool = False                  # dikompilasi dengan -fopenmp (I/O dan pewarnaan paralel)
    max_threads: int = 1
    auto_cuda_threshold_pixels: int | None = None   # padanan dalam piksel pada K=8 (hanya informasi)
    auto_cuda_threshold_work: int | None = None     # aturan auto: cuda bila N*K >= nilai ini (#12)

    @property
    def kmeans_choices(self) -> tuple[str, ...]:
        """Pilihan `--kmeans` yang ada pada build ini (`auto` selalu ada)."""
        return tuple(c for c in KMEANS_CHOICES if c == "auto" or getattr(self, c))

    @property
    def notice(self) -> str | None:
        """Catatan untuk pengguna bila build ini kekurangan mesin; None bila lengkap."""
        if self.cpu is None:
            return ("Build landcover ini tidak punya mesin CPU (kmeans_cpu atau ref_kmeans): "
                    "semua run akan gagal.")
        if not self.omp and not self.cuda:
            return ("OpenMP dan CUDA belum terhubung pada build landcover ini, jadi mode "
                    "Optimalisasi menjalankan K-Means serial (hanya baca dan pewarnaan yang "
                    "paralel); speedup K-Means belum akan tampak.")
        if not self.cuda:
            return "CUDA tidak terhubung pada build landcover ini (opsi CUDA disembunyikan)."
        if not self.omp:
            return "OpenMP K-Means (kmeans_omp) belum terhubung pada build ini (opsi OpenMP disembunyikan)."
        return None


def parse_core_info(text: str) -> CoreInfo:
    """Urai keluaran `landcover --info`. ValueError bila bukan JSON atau bentuknya tidak dikenali."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"keluaran --info bukan JSON: {exc}") from None
    engines = data.get("engines") if isinstance(data, dict) else None
    if not isinstance(engines, dict):
        raise ValueError("keluaran --info tidak punya objek 'engines'")
    cpu = engines.get("cpu")
    if cpu is not None and not isinstance(cpu, str):
        raise ValueError("--info: engines.cpu harus teks atau null")
    for key in ("omp", "cuda"):
        if not isinstance(engines.get(key), bool):
            raise ValueError(f"--info: engines.{key} harus true atau false")
    threads = data.get("max_threads", 1)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("--info: max_threads harus bilangan bulat >= 1")
    threshold = data.get("auto_cuda_threshold_pixels")
    if threshold is not None and (not isinstance(threshold, int) or isinstance(threshold, bool)):
        raise ValueError("--info: auto_cuda_threshold_pixels harus bilangan bulat")
    work = data.get("auto_cuda_threshold_work")             # tidak ada pada biner sebelum #12
    if work is not None and (not isinstance(work, int) or isinstance(work, bool)):
        raise ValueError("--info: auto_cuda_threshold_work harus bilangan bulat")
    return CoreInfo(
        cpu=cpu, omp=engines["omp"], cuda=engines["cuda"], openmp=bool(data.get("openmp", False)),
        max_threads=threads, auto_cuda_threshold_pixels=threshold, auto_cuda_threshold_work=work,
    )


@dataclass(frozen=True)
class CoreProbe:
    info: CoreInfo | None
    problem: str | None          # alasan `info` kosong (untuk ditampilkan), None bila berhasil


_PROBE_CACHE: dict[tuple, CoreInfo] = {}
_PROBE_LOCK = threading.Lock()


def _stamp(cmd: Sequence[str]) -> tuple:
    """Identitas perintah untuk cache: tiap elemen yang berupa berkas + waktu ubah + ukurannya."""
    parts = []
    for c in cmd:
        try:
            st = os.stat(c)
        except OSError:
            parts.append((c, None, None))
        else:
            parts.append((c, st.st_mtime_ns, st.st_size))
    return tuple(parts)


def probe_core(core_bin: "str | os.PathLike[str] | Sequence[str]", *, timeout: float = INFO_TIMEOUT_S) -> CoreProbe:
    """Jalankan `landcover --info`. TIDAK PERNAH melempar: kegagalan apa pun (biner hilang, kode
    keluar ≠ 0, bukan JSON, batas waktu) menjadi `CoreProbe(None, alasan)`, supaya halaman tetap
    tampil. Hasil berhasil di-cache per (perintah, waktu ubah, ukuran); kegagalan tidak."""
    try:
        cmd = as_command(core_bin)
        locate_executable(cmd[0])
    except (CliLauncherError, ValueError) as exc:
        return CoreProbe(None, getattr(exc, "summary", str(exc)))
    key = _stamp(cmd)
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get(key)
    if cached is not None:
        return CoreProbe(cached, None)
    try:
        done = subprocess.run(
            [*cmd, "--info"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CoreProbe(None, f"`landcover --info` tidak selesai dalam {timeout:g} s")
    except OSError as exc:
        return CoreProbe(None, f"`landcover --info` tidak bisa dijalankan: {exc}")
    if done.returncode != 0:
        why = _tail(done.stderr, DETAIL_CHARS).strip()
        return CoreProbe(None, f"`landcover --info` gagal (kode {done.returncode})" + (f": {why}" if why else ""))
    try:
        info = parse_core_info(done.stdout)
    except ValueError as exc:
        return CoreProbe(None, str(exc))
    with _PROBE_LOCK:
        _PROBE_CACHE[key] = info
    return CoreProbe(info, None)


# --- perintah -------------------------------------------------------------- #

def as_command(core_bin: "str | os.PathLike[str] | Sequence[str]") -> list[str]:
    """Awalan perintah dari `LC_CORE_BIN`: satu path, atau urutan argumen (mis. [python, skrip.py]
    untuk skrip palsu atau pembungkus). ValueError bila kosong."""
    if isinstance(core_bin, (str, os.PathLike)):
        parts = [os.fspath(core_bin)]
    else:
        parts = [os.fspath(p) for p in core_bin]
    if not parts or not parts[0]:
        raise ValueError("perintah program landcover kosong (LC_CORE_BIN)")
    return parts


def locate_executable(first: str) -> None:
    """CoreNotFoundError bila `first` bukan berkas yang bisa dijalankan (atau tidak ada di PATH)."""
    hint = ("Bangun dengan `make landcover` (di WSL) lalu arahkan LC_CORE_BIN ke biner itu.")
    if os.path.dirname(first):
        p = Path(first)
        if not p.is_file():
            raise CoreNotFoundError(f"Program landcover tidak ditemukan: {first}. {hint}")
        if os.name != "nt" and not os.access(p, os.X_OK):
            raise CoreNotFoundError(f"Program landcover tidak punya izin eksekusi: {first}. {hint}")
    elif shutil.which(first) is None:
        raise CoreNotFoundError(f"Program landcover tidak ditemukan di PATH: {first}. {hint}")


def build_command(
    params: RunParams,
    out_dir: Path | str,
    input_path: Path | str,
    *,
    core: Sequence[str] = ("landcover",),
    write_labels: bool = True,
) -> list[str]:
    """Perintah lengkap untuk `params`: SATU-SATUNYA tempat format flag CLI (plan 4.1).

    - `--input` = PATH dataset (PNG atau folder dengan manifest.json); `--name` = nama dataset di UI,
      supaya `meta.dataset` cocok dengan `params.dataset`.
    - mode `cpu`: tanpa `--kmeans` dan `--threads` (program mengabaikannya; hasil cpu tidak
      bergantung padanya, `run_id` pun menormalkannya).
    - `--write-labels` bawaan SELALU: verifikasi antarmode jadi penuh dan ekspor PNG resolusi
      penuh tersedia; harganya 1 byte per piksel di disk dan waktu di luar `total_ms`.
    - Tiap nilai satu elemen argv sendiri (bukan `--flag=nilai`), supaya pembungkus yang menerjemahkan
      path per argumen (mis. ke WSL) bekerja.
    """
    cmd = [*core, "--mode", params.mode]
    if params.mode == "opt":
        cmd += ["--kmeans", params.kmeans]
    cmd += [
        "--input", os.fspath(input_path), "--name", params.dataset,
        "--k", str(params.k), "--iters", str(params.iters), "--seed", str(params.seed),
    ]
    if params.mode == "opt":
        cmd += ["--threads", str(params.threads)]
    cmd += ["--out", os.fspath(out_dir)]
    if write_labels:
        cmd.append("--write-labels")
    return cmd


def resolve_timeout(value: float | str | None = None) -> float:
    """Batas waktu (detik): `value`, bila None `LC_CORE_TIMEOUT`, bila kosong `DEFAULT_TIMEOUT_S`.
    ValueError bila bukan angka > 0 yang terhingga."""
    raw = value if value is not None else os.getenv(TIMEOUT_ENV)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_TIMEOUT_S
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{TIMEOUT_ENV} harus angka detik, ditemukan {raw!r}") from None
    if not (0 < seconds < float("inf")):
        raise ValueError(f"{TIMEOUT_ENV} harus > 0 dan terhingga, ditemukan {raw!r}")
    return seconds


# --- pembersihan ----------------------------------------------------------- #

def clear_run_files(out_dir: Path | str) -> None:
    """Buang berkas run lama (`RUN_FILES`, `meta.json` duluan) dan sisa `*.tmp` di `out_dir`.

    Berkas lain dibiarkan; folder tidak ada = tidak ada yang dibersihkan. Hanya berkas biasa yang
    dihapus. CliLauncherError bila ada yang tidak bisa dihapus (mis. sedang dibuka program lain)."""
    out = Path(out_dir)
    if not out.is_dir():
        return
    victims = [out / n for n in RUN_FILES if (out / n).is_file()]
    victims += sorted(p for p in out.glob("*.tmp") if p.is_file())
    for p in victims:
        try:
            p.unlink()
        except OSError as exc:
            raise CliLauncherError(f"Tidak bisa menghapus berkas run lama {p}: {exc}") from exc


def _check_out_dir(input_path: Path, out_dir: Path) -> None:
    """Folder keluaran tidak boleh folder dataset (atau di dalamnya), atau folder tempat PNG masukan
    berada: pembersihan di atas bisa menghapus masukan."""
    out, src = out_dir.resolve(), input_path.resolve()
    if out == src or src in out.parents or (src.is_file() and out == src.parent):
        raise ValueError(
            f"folder keluaran {out_dir} bertumpuk dengan masukan {input_path}; pakai folder run terpisah"
        )


# --- proses ---------------------------------------------------------------- #

def _clean_line(raw: str) -> str:
    line = _CONTROL_CHARS.sub("", raw.rstrip("\r\n")).rstrip()
    return line if len(line) <= LINE_CHARS else line[: LINE_CHARS - 1] + "…"


def _tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else "…" + text[-(limit - 1):]


def _kill(proc: subprocess.Popen) -> None:
    """Matikan proses beserta turunannya sebisanya, lalu pastikan prosesnya sendiri mati."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)     # start_new_session=True: pgid == pid
        elif os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


@dataclass
class _Finished:
    returncode: int
    timed_out: bool
    stderr: str
    stdout_tail: list[str]


def _execute(cmd: list[str], on_line: Callable[[str], None], timeout: float) -> _Finished:
    """Jalankan `cmd`: stdout diteruskan per baris ke `on_line`, stderr ditampung (ekor saja).

    Dua thread pembaca menguras kedua pipa SELAMA proses berjalan, jadi keluaran sebesar apa pun
    tidak membuat proses menunggu pipa penuh. Galat dari `on_line` tidak menghentikan pengurasan;
    galat pertama dilempar ulang sesudah proses selesai."""
    popen_kw: dict = {}
    if os.name == "posix":
        popen_kw["start_new_session"] = True
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8", errors="replace", **popen_kw,
    )
    stdout_tail: deque[str] = deque(maxlen=20)
    err_box: dict[str, object] = {"text": "", "callback": None}

    def pump_stdout() -> None:
        while True:
            raw = proc.stdout.readline(READ_CHUNK)
            if not raw:
                return
            line = _clean_line(raw)
            if not line:
                continue
            stdout_tail.append(line)
            try:
                on_line(line)
            except Exception as exc:        # tetap menguras; dilempar ulang di bawah
                if err_box["callback"] is None:
                    err_box["callback"] = exc

    def pump_stderr() -> None:
        text = ""
        while True:
            chunk = proc.stderr.read(READ_CHUNK)
            if not chunk:
                break
            text = (text + chunk)[-STDERR_KEEP_CHARS:]
        err_box["text"] = text

    readers = [threading.Thread(target=pump_stdout, daemon=True),
               threading.Thread(target=pump_stderr, daemon=True)]
    for t in readers:
        t.start()
    timed_out = False
    try:
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill(proc)
            proc.wait()
    finally:
        if proc.poll() is None:             # mis. KeyboardInterrupt di thread utama
            _kill(proc)
            proc.wait()
        for t in readers:
            t.join(timeout=JOIN_TIMEOUT_S)
        for pipe in (proc.stdout, proc.stderr):
            try:
                pipe.close()
            except OSError:
                pass
    if err_box["callback"] is not None:
        raise err_box["callback"]           # type: ignore[misc]
    return _Finished(proc.returncode, timed_out, str(err_box["text"]), list(stdout_tail))


def _failure(done: _Finished, timeout: float) -> CliLauncherError:
    detail = _tail(done.stderr.strip(), DETAIL_CHARS)
    if not detail and done.stdout_tail:
        detail = "(stderr kosong; baris terakhir stdout)\n" + "\n".join(done.stdout_tail[-5:])
    code = done.returncode
    if done.timed_out:
        return CoreTimeoutError(
            f"landcover melewati batas waktu {timeout:g} s dan dihentikan "
            f"(ubah dengan {TIMEOUT_ENV}).", detail=detail, exit_code=code,
        )
    if code == EXIT_RUNTIME:
        summary = "landcover gagal saat berjalan (kode 1)."
    elif code == EXIT_USAGE:
        summary = ("landcover menolak argumen (kode 2): argumen dari aplikasi tidak cocok dengan "
                   "build ini (biner lama, atau galat di build_command).")
    elif code == EXIT_ENGINE:
        summary = ("Mesin K-Means yang diminta tidak terhubung pada build landcover ini (kode 3); "
                   "pilih mesin lain atau bangun ulang dengan modulnya (lihat `landcover --info`).")
    elif code < 0:
        summary = (f"landcover dihentikan sinyal {-code} "
                   "(9 = dimatikan, sering karena memori habis).")
    else:
        summary = f"landcover keluar dengan kode {code} (tidak dikenal)."
    return CoreFailedError(summary, detail=detail, exit_code=code)


# --- pemeriksaan keluaran ---------------------------------------------------- #

def meta_mismatches(run: RunResult, params: RunParams) -> list[str]:
    """Bagian `meta.json` yang tidak cocok dengan `params`: menangkap flag yang diabaikan diam-diam
    (mis. `--name`), biner yang menulis data tiruan, atau mesin yang bukan yang diminta."""
    m = run.meta
    out: list[str] = []
    if m.get("mock") is not False:
        out.append(f"meta.mock = {m.get('mock')!r}; hasil core harus false (data tiruan hanya di runs/mock/)")
    for key, want in (("dataset", params.dataset), ("mode", params.mode), ("k", params.k),
                      ("iters", params.iters), ("seed", params.seed)):
        if m.get(key) != want:
            out.append(f"meta.{key} = {m.get(key)!r}, diminta {want!r}")
    if params.mode == "cpu" and m.get("kmeans_device") != "serial":
        out.append(f"meta.kmeans_device = {m.get('kmeans_device')!r}, mode cpu harus 'serial'")
    if params.mode == "opt" and params.kmeans in ("omp", "cuda") and m.get("kmeans_device") != params.kmeans:
        out.append(f"meta.kmeans_device = {m.get('kmeans_device')!r}, diminta {params.kmeans!r}")
    return out


def check_output(out_dir: Path, params: RunParams) -> RunResult:
    """Baca dan periksa folder run sesudah kode keluar 0; CoreOutputError bila ada masalah."""
    try:
        run = read_run(out_dir)
    except ContractError as exc:
        raise CoreOutputError(
            "landcover selesai (kode 0) tetapi folder run tidak terbaca.",
            detail="\n".join(exc.problems), exit_code=0,
        ) from exc
    except OSError as exc:
        raise CoreOutputError(f"Folder run tidak bisa dibaca: {exc}", exit_code=0) from exc
    problems = validate(run) + meta_mismatches(run, params)
    if problems:
        raise CoreOutputError(
            "landcover selesai (kode 0) tetapi folder run tidak valid.",
            detail="\n".join(f"- {p}" for p in problems), exit_code=0,
        )
    return run


# --- launcher -------------------------------------------------------------- #

class CliLauncher:
    """Launcher yang memanggil program `landcover` (`LC_BACKEND=cli`).

    `core_bin`: path biner, atau urutan argumen awalan perintah (mis. `[sys.executable, "skrip.py"]`).
    `timeout`: detik; None = `LC_CORE_TIMEOUT` atau 1800. `write_labels`: kirim `--write-labels`
    (bawaan ya; matikan hanya untuk uji)."""

    def __init__(
        self,
        data_dir: Path | str,
        core_bin: "str | os.PathLike[str] | Sequence[str]",
        *,
        timeout: float | str | None = None,
        write_labels: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.core = as_command(core_bin)
        self.timeout = resolve_timeout(timeout)
        self.write_labels = write_labels

    def command(self, params: RunParams, out_dir: Path | str) -> list[str]:
        """Perintah untuk `params` (dataset di-resolve dari `data_dir`). ValueError bila dataset
        tidak ditemukan (nama sintetis hanya ada di backend mock) atau folder keluaran bertumpuk."""
        ds = resolve_dataset(self.data_dir, params.dataset)
        if ds is None or ds.path is None:
            raise ValueError(
                f"dataset {params.dataset!r} tidak ditemukan di {self.data_dir} "
                "(backend cli butuh folder tile dengan manifest.json atau berkas PNG)"
            )
        _check_out_dir(ds.path, Path(out_dir))
        return build_command(params, out_dir, ds.path, core=self.core, write_labels=self.write_labels)

    def run(
        self,
        params: RunParams,
        out_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        say = on_progress or (lambda _line: None)
        out_dir = Path(out_dir)
        cmd = self.command(params, out_dir)
        locate_executable(self.core[0])
        clear_run_files(out_dir)
        try:
            done = _execute(cmd, say, self.timeout)
        except OSError as exc:          # Popen gagal: format biner salah, izin, dst.
            raise CoreNotFoundError(f"Program landcover tidak bisa dijalankan: {exc}") from exc
        if done.timed_out or done.returncode != 0:
            raise _failure(done, self.timeout)
        check_output(out_dir, params)
        return out_dir


__all__ = [
    "CliLauncher", "CliLauncherError", "CoreFailedError", "CoreInfo", "CoreNotFoundError",
    "CoreOutputError", "CoreProbe", "CoreTimeoutError", "DEFAULT_TIMEOUT_S", "RUN_FILES",
    "TIMEOUT_ENV", "as_command", "build_command", "check_output", "clear_run_files",
    "locate_executable", "meta_mismatches", "parse_core_info", "probe_core", "resolve_timeout",
]
