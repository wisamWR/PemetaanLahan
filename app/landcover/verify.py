"""Verifikasi antarmode: apakah hasil Optimalisasi identik dengan CPU saja (plan 4.5 dan Langkah 9).

Fungsi murni tanpa Streamlit. `verify_pair(ref, run)` membandingkan run Optimalisasi `run` dengan
run CPU saja `ref` (L0, acuan) dan mengembalikan `Verification`.

Sumber, dicoba berurutan; yang gagal dilewati dan alasannya dicatat di `notes` (tidak pernah
melempar karena berkasnya hilang, rusak, atau tidak cocok):
  1. `verification.json` di folder run yang diuji (modul verifikasi Azka). Dibaca ketat.
  2. `labels.u8` di KEDUA folder: dibandingkan per blok (tidak pernah dimuat utuh).
  3. Pratinjau: `labels_preview.png` (bila ukurannya sama), `counts.json`, `centroids.json`.
     Ini verifikasi PARSIAL: label resolusi penuh tidak diperiksa.

Aturan yang tidak boleh dilonggarkan:
  * Centroid dan counts SELALU dibandingkan dari `centroids.json`/`counts.json` kedua run, apa pun
    sumber utamanya. Bukti negatif dari sumber mana pun menang atas bukti positif sumber lain, jadi
    `verification.json` basi yang berkata "100 % identik" tidak bisa menutupi counts yang berbeda.
  * "Identik" diputuskan dari bilangan bulat (`n_diff == 0`), bukan dari `persen == 100.0`. Satu
    piksel berbeda dari 120 juta tidak boleh menjadi 100 %: `identical_pct` kurang dari 100 bila
    `n_diff > 0`, dan tampilan memotong (bukan membulatkan).
  * Hanya run sebanding yang dibandingkan (`comparability_problems`); selain itu status
    `not_comparable`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import numpy as np

from landcover.contract import RunResult

STATUS_IDENTICAL = "identical"
STATUS_DIFFERENT = "different"
STATUS_NOT_COMPARABLE = "not_comparable"

SOURCE_FILE = "verification.json"
SOURCE_LABELS = "labels.u8"
SOURCE_PREVIEW = "preview"
FULL_SOURCES = (SOURCE_FILE, SOURCE_LABELS)      # sumber yang menjangkau label resolusi penuh

DIFF_LABELS = "labels"
DIFF_CENTROIDS = "centroids"
DIFF_COUNTS = "counts"

VERIFICATION_FILE = "verification.json"
LABELS_FILE = "labels.u8"
BLOCK_BYTES = 1 << 24                            # 16 MiB per blok saat membandingkan labels.u8

# Berkas yang memengaruhi hasil verifikasi; dipakai untuk kunci cache (`verification_stamp`).
_STAMPED = ("meta.json", "counts.json", "centroids.json", "labels_preview.png", LABELS_FILE, VERIFICATION_FILE)


@dataclass(frozen=True)
class Verification:
    status: str                          # STATUS_*
    source: str | None                   # SOURCE_*; None bila tidak sebanding
    n_compared: int | None               # piksel yang labelnya dibandingkan (penuh atau pratinjau); None = tidak
    n_diff: int | None                   # piksel berbeda; None = tidak diketahui (verification.json hanya memberi persen)
    identical_pct: float | None          # 0..100; < 100 bila ada piksel berbeda
    centroids_equal: bool | None
    counts_equal: bool | None
    reasons: tuple[str, ...] = ()        # tidak sebanding: alasan (teks)
    differences: tuple[str, ...] = ()    # berbeda: DIFF_* yang gagal
    notes: tuple[str, ...] = ()          # sumber yang dilewati beserta sebabnya, dan ketidakkonsistenan data
    mock: bool = False

    @property
    def full(self) -> bool:
        """True bila label resolusi penuh dijangkau (bukan hanya pratinjau)."""
        return self.source in FULL_SOURCES


class _Unusable(Exception):
    """Satu sumber tidak bisa dipakai; pesan = alasan (masuk `notes`)."""


# --- sebanding ---------------------------------------------------------------- #

def comparability_problems(ref: RunResult, run: RunResult) -> list[str]:
    """Alasan `ref` (CPU saja) dan `run` (Optimalisasi) tidak boleh dibandingkan; kosong = sebanding."""
    problems: list[str] = []
    if ref.mode != "cpu":
        problems.append(f"acuan harus run CPU saja (mode cpu), ditemukan mode {ref.mode!r}")
    if run.mode != "opt":
        problems.append(f"yang diuji harus run Optimalisasi (mode opt), ditemukan mode {run.mode!r}")
    for label, a, b in (
        ("dataset", ref.meta.get("dataset"), run.meta.get("dataset")),
        ("K", ref.k, run.k),
        ("iterasi", ref.meta.get("iters"), run.meta.get("iters")),
        ("seed", ref.meta.get("seed"), run.meta.get("seed")),
    ):
        if a != b:
            problems.append(f"{label} berbeda ({a} vs {b})")
    if (ref.width, ref.height) != (run.width, run.height):
        problems.append(
            f"ukuran citra berbeda ({ref.width}×{ref.height} vs {run.width}×{run.height})"
        )
    if ref.width < 1 or ref.height < 1:
        problems.append(f"ukuran citra tidak sah ({ref.width}×{ref.height})")
    if ref.mock != run.mock:
        problems.append("salah satu run adalah data tiruan (mock) dan yang lain bukan")
    return problems


# --- sumber 1: verification.json ---------------------------------------------- #

def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _from_file(ref: RunResult, run: RunResult) -> tuple[float, bool] | None:
    """(identical_pct, centroids_equal) dari `verification.json` di folder `run`.

    None bila berkasnya tidak ada (bukan masalah). `_Unusable` bila ada tetapi rusak, salah bentuk,
    atau bukan tentang pasangan ini."""
    path = run.path / VERIFICATION_FILE
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _Unusable(f"{VERIFICATION_FILE} tidak terbaca ({exc})") from None
    if not isinstance(data, dict):
        raise _Unusable(f"{VERIFICATION_FILE}: harus berupa objek JSON")

    where = data.get("ref")
    if not isinstance(where, str) or not where.strip():
        raise _Unusable(f"{VERIFICATION_FILE}: 'ref' harus teks (folder run CPU saja)")
    # Hanya nama foldernya yang dibandingkan: path Windows dan WSL berbeda untuk folder yang sama.
    named = PureWindowsPath(where.strip()).name
    if named != ref.path.name:
        raise _Unusable(
            f"{VERIFICATION_FILE} merujuk ke folder '{named}', bukan acuan CPU '{ref.path.name}'"
        )
    n_pixels = data.get("n_pixels")
    if isinstance(n_pixels, bool) or not isinstance(n_pixels, int):
        raise _Unusable(f"{VERIFICATION_FILE}: 'n_pixels' harus bilangan bulat")
    if n_pixels != run.n_pixels:
        raise _Unusable(
            f"{VERIFICATION_FILE}: n_pixels {n_pixels} tidak sama dengan lebar×tinggi run ({run.n_pixels})"
        )
    pct = data.get("identical_pct")
    if not _is_number(pct) or not math.isfinite(pct) or not 0 <= pct <= 100:
        raise _Unusable(f"{VERIFICATION_FILE}: 'identical_pct' harus angka 0..100, ditemukan {pct!r}")
    centroids_equal = data.get("centroids_equal")
    if not isinstance(centroids_equal, bool):
        raise _Unusable(f"{VERIFICATION_FILE}: 'centroids_equal' harus true atau false")
    return float(pct), centroids_equal


# --- sumber 2: labels.u8 ------------------------------------------------------ #

def count_label_differences(
    path_a: Path | str, path_b: Path | str, n_pixels: int, *, block_bytes: int = BLOCK_BYTES
) -> int:
    """Jumlah byte yang berbeda antara dua berkas label `uint8` sepanjang `n_pixels` byte.

    Dibaca per blok `block_bytes` (memori terbatas ~2 blok). Sengaja tidak memakai `np.memmap`: di
    Windows pemetaan yang masih terbuka mengunci berkas, sehingga `os.replace` run berikutnya
    gagal. `OSError` bila berkas tidak bisa dibaca atau lebih pendek dari `n_pixels`."""
    if block_bytes < 1:
        raise ValueError(f"block_bytes harus >= 1, ditemukan {block_bytes}")
    diff = 0
    remaining = int(n_pixels)
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        while remaining > 0:
            want = min(block_bytes, remaining)
            a = np.fromfile(fa, dtype=np.uint8, count=want)
            b = np.fromfile(fb, dtype=np.uint8, count=want)
            if a.size != want or b.size != want:
                raise OSError("berkas label lebih pendek dari lebar×tinggi")
            diff += int(np.count_nonzero(a != b))
            remaining -= want
    return diff


def _from_labels(ref: RunResult, run: RunResult, block_bytes: int) -> int | None:
    """Jumlah piksel berbeda dari `labels.u8` kedua folder. None bila tidak ada satu pun (bukan
    masalah). `_Unusable` bila hanya satu yang ada, ukurannya salah, atau tidak terbaca."""
    a, b = ref.path / LABELS_FILE, run.path / LABELS_FILE
    present = [a.is_file(), b.is_file()]
    if not any(present):
        return None
    if not all(present):
        owner = ref.path.name if present[0] else run.path.name
        raise _Unusable(f"{LABELS_FILE} hanya ada di folder '{owner}': label resolusi penuh tidak dibandingkan")
    n = ref.n_pixels
    try:
        for p in (a, b):
            size = p.stat().st_size
            if size != n:
                raise _Unusable(
                    f"{LABELS_FILE} di '{p.parent.name}' berukuran {size} byte, seharusnya "
                    f"lebar×tinggi = {n} byte"
                )
        return count_label_differences(a, b, n, block_bytes=block_bytes)
    except OSError as exc:
        raise _Unusable(f"{LABELS_FILE} tidak terbaca ({exc})") from None


# --- gabungan ----------------------------------------------------------------- #

def _pct(n_diff: int, n_total: int) -> float:
    pct = 100.0 * (n_total - n_diff) / n_total
    if n_diff > 0 and pct >= 100.0:          # selisih terlalu kecil untuk float: jangan tampil 100 %
        pct = math.nextafter(100.0, 0.0)
    return pct


def verify_pair(
    ref: RunResult, run: RunResult, *, block_bytes: int = BLOCK_BYTES
) -> Verification:
    """Bandingkan `run` (Optimalisasi) dengan `ref` (CPU saja, acuan L0). Lihat docstring modul."""
    reasons = comparability_problems(ref, run)
    mock = ref.mock or run.mock
    if reasons:
        return Verification(
            status=STATUS_NOT_COMPARABLE, source=None, n_compared=None, n_diff=None,
            identical_pct=None, centroids_equal=None, counts_equal=None,
            reasons=tuple(reasons), mock=mock,
        )

    notes: list[str] = []
    counts_equal = bool(np.array_equal(ref.counts, run.counts))
    centroids_equal = bool(np.array_equal(ref.centroids, run.centroids))

    source = None
    n_compared: int | None = None
    n_diff: int | None = None
    pct: float | None = None

    try:
        report = _from_file(ref, run)
    except _Unusable as exc:
        notes.append(f"{exc}; sumber ini tidak dipakai")
        report = None
    if report is not None:
        source, n_compared = SOURCE_FILE, run.n_pixels
        pct, file_centroids_equal = report
        n_diff = 0 if pct == 100.0 else None
        if file_centroids_equal != centroids_equal:
            notes.append(
                f"{VERIFICATION_FILE} dan centroids.json tidak sepakat soal centroid "
                f"(berkas: {'sama' if file_centroids_equal else 'beda'}, centroids.json: "
                f"{'sama' if centroids_equal else 'beda'}); hasil dianggap berbeda"
            )
        centroids_equal = centroids_equal and file_centroids_equal

    if source is None:
        try:
            diff = _from_labels(ref, run, block_bytes)
        except _Unusable as exc:
            notes.append(f"{exc}; sumber ini tidak dipakai")
            diff = None
        if diff is not None:
            source, n_compared, n_diff = SOURCE_LABELS, run.n_pixels, diff
            pct = _pct(diff, run.n_pixels)

    if source is None:
        source = SOURCE_PREVIEW
        a, b = ref.labels_preview, run.labels_preview
        if a.shape == b.shape and a.size > 0:
            n_compared, n_diff = int(a.size), int(np.count_nonzero(a != b))
            pct = _pct(n_diff, n_compared)
        else:
            notes.append(
                f"labels_preview berbeda ukuran ({a.shape[1]}×{a.shape[0]} vs {b.shape[1]}×{b.shape[0]}): "
                "label pratinjau tidak dibandingkan"
                if a.ndim == 2 and b.ndim == 2 else
                "labels_preview tidak bisa dibandingkan: label pratinjau tidak dibandingkan"
            )

    # Identik = nol piksel berbeda (bilangan bulat). verification.json tanpa n_diff: hanya persen
    # tepat 100 yang diterima (sudah dijadikan n_diff = 0 di atas), selain itu berbeda.
    if n_diff is not None:
        labels_equal: bool | None = n_diff == 0
    elif pct is not None:
        labels_equal = False
    else:
        labels_equal = None
    differences: list[str] = []
    if labels_equal is False:
        differences.append(DIFF_LABELS)
    if not centroids_equal:
        differences.append(DIFF_CENTROIDS)
    if not counts_equal:
        differences.append(DIFF_COUNTS)
    if labels_equal and not counts_equal:
        notes.append("label identik tetapi counts.json berbeda: berkas run tidak konsisten")

    return Verification(
        status=STATUS_DIFFERENT if differences else STATUS_IDENTICAL,
        source=source, n_compared=n_compared, n_diff=n_diff, identical_pct=pct,
        centroids_equal=centroids_equal, counts_equal=counts_equal,
        differences=tuple(differences), notes=tuple(notes), mock=mock,
    )


def verification_stamp(run: RunResult) -> tuple:
    """Kunci cache hasil verifikasi untuk `run`: path dan (waktu ubah, ukuran) berkas yang dibaca
    verifikasi. Berubah bila salah satu berkas ditulis ulang, dibuat, atau dihapus."""
    entries: list[tuple] = [("path", str(run.path))]
    for name in _STAMPED:
        try:
            info = (run.path / name).stat()
        except OSError:
            entries.append((name, None, None))
        else:
            entries.append((name, info.st_mtime_ns, info.st_size))
    return tuple(entries)


__all__ = [
    "BLOCK_BYTES", "DIFF_CENTROIDS", "DIFF_COUNTS", "DIFF_LABELS", "FULL_SOURCES", "LABELS_FILE",
    "SOURCE_FILE", "SOURCE_LABELS", "SOURCE_PREVIEW", "STATUS_DIFFERENT", "STATUS_IDENTICAL",
    "STATUS_NOT_COMPARABLE", "VERIFICATION_FILE", "Verification", "comparability_problems",
    "count_label_differences", "verification_stamp", "verify_pair",
]
