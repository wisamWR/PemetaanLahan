"""Pembaca kontrak folder run (plan Bagian 4.3, 4.4 dan Langkah 2).

UI hanya membaca folder run lewat modul ini. MockLauncher dan core C++ menulis
format yang sama, jadi yang berbeda hanya launcher-nya.

Pembagian tugas:
  read_run()    membaca dan mengubah ke tipe Python; bila strukturnya rusak
                (file hilang, JSON rusak, kunci wajib tidak ada, tipe salah)
                melempar ContractError berisi SEMUA masalah yang ditemukan.
  validate()    memeriksa kewajaran isi RunResult (aturan Langkah 2) dan
                mengembalikan daftar masalah; kosong berarti valid.

Konversi milidetik ke detik dan pemetaan nama tahap (read_ms -> "baca", dst.)
hanya ada di sini. Bila tim memilih detik atau nama lain, ubah CORE_FIELDS.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

STAGES = ("baca", "inisialisasi", "kmeans", "pewarnaan", "tulis")
# Nama field di timing.json (milidetik; = StageTimes di src/common/kmeans.h).
CORE_FIELDS = {
    "baca": "read_ms",
    "inisialisasi": "init_ms",
    "kmeans": "kmeans_ms",
    "pewarnaan": "color_ms",
    "tulis": "write_ms",
}
# Kunci di timing.json["devices"] (bahasa Inggris, dari core) -> nama tahap.
DEVICE_KEYS = {
    "read": "baca",
    "init": "inisialisasi",
    "kmeans": "kmeans",
    "color": "pewarnaan",
    "write": "tulis",
}
# Rincian transfer/kernel CUDA, hanya untuk tahap kmeans (milidetik di berkas).
DETAIL_FIELDS = {"h2d": "h2d_ms", "kernel": "kernel_ms", "d2h": "d2h_ms"}
DEVICES = frozenset({"serial", "omp", "cuda"})
MODES = frozenset({"cpu", "opt"})
KNOWN_CONTRACT_VERSIONS = frozenset({1})

# File wajib. map.png (peta penuh) dan labels.u8 opsional dan tidak dibaca UI.
RUN_FILES = (
    "meta.json",
    "timing.json",
    "centroids.json",
    "counts.json",
    "labels_preview.png",
    "map_preview.png",
)

_MS_PER_S = 1000.0


class ContractError(ValueError):
    """Folder run (atau batch) tidak bisa dibaca; `problems` memuat semua sebabnya."""

    def __init__(self, path: Path, problems: list[str]):
        self.path = Path(path)
        self.problems = list(problems)
        detail = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"Folder run tidak valid: {self.path}\n{detail}")


@dataclass(frozen=True)
class StageTiming:
    time: float                              # detik
    device: str                              # "serial" | "omp" | "cuda"
    detail: dict[str, float] | None = None   # detik; h2d/kernel/d2h untuk kmeans CUDA


@dataclass(frozen=True, eq=False)
class RunResult:
    path: Path
    meta: dict
    stages: dict[str, StageTiming]
    total: float                             # detik, dari total_ms (tidak dihitung ulang)
    centroids: np.ndarray                    # (K, 3) int64
    counts: np.ndarray                       # (K,)   int64
    labels_preview: np.ndarray               # (h, w) uint8
    map_preview: np.ndarray                  # (h, w, 3) uint8

    @property
    def mock(self) -> bool:
        return bool(self.meta.get("mock", False))

    @property
    def k(self) -> int:
        return int(self.meta["k"])

    @property
    def width(self) -> int:
        return int(self.meta["width"])

    @property
    def height(self) -> int:
        return int(self.meta["height"])

    @property
    def n_pixels(self) -> int:
        return self.width * self.height

    @property
    def mode(self) -> str:
        return str(self.meta["mode"])

    @property
    def kmeans_device(self) -> str:
        return str(self.meta["kmeans_device"])


@dataclass(frozen=True, eq=False)
class BatchResult:
    path: Path
    names: tuple[str, ...]
    runs: tuple[RunResult, ...]
    total: float | None                      # detik; None bila batch.json tidak memuatnya
    pipelined: bool
    meta: dict                               # isi batch.json apa adanya

    @property
    def mock(self) -> bool:
        return any(r.mock for r in self.runs)


# --------------------------------------------------------------------------- #
# Pembacaan
# --------------------------------------------------------------------------- #

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _read_json(path: Path, name: str, problems: list[str]) -> Any:
    """Isi JSON, atau None bila file tidak ada (sudah dicatat) / rusak (dicatat di sini)."""
    f = path / name
    if not f.is_file():
        return None
    try:
        with f.open("r", encoding="utf-8-sig") as fh:  # -sig: toleran terhadap BOM
            data = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        problems.append(f"{name}: bukan JSON yang bisa dibaca ({e})")
        return None
    if data is None:
        problems.append(f"{name}: berisi null, harus objek atau daftar JSON")
    return data


def _read_png(path: Path, name: str, kind: str, problems: list[str]) -> np.ndarray | None:
    """kind 'labels' -> (h, w) uint8; kind 'map' -> (h, w, 3) uint8."""
    f = path / name
    if not f.is_file():
        return None
    try:
        with Image.open(f) as im:
            if kind == "labels":
                # L (grayscale 8-bit) atau P (indeks palet = id cluster). Mode lain
                # (RGB, 16-bit, ...) berarti isinya bukan id cluster.
                if im.mode not in ("L", "P"):
                    problems.append(
                        f"{name}: harus grayscale 8-bit (mode L), ditemukan mode {im.mode}"
                    )
                    return None
                return np.array(im, dtype=np.uint8)
            return np.array(im.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as e:
        problems.append(f"{name}: bukan gambar PNG yang bisa dibaca ({e})")
        return None


def _int_array(raw: Any, name: str, ndim: int, problems: list[str]) -> np.ndarray | None:
    """Daftar integer JSON -> array int64 (ndim 1 atau 2, kolom RGB bila 2)."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        problems.append(f"{name}: harus berupa daftar (list) JSON")
        return None

    def ints(seq: Any) -> bool:
        return isinstance(seq, list) and all(_is_int(v) for v in seq)

    if ndim == 1:
        if not ints(raw):
            problems.append(f"{name}: harus berupa daftar bilangan bulat")
            return None
        return np.array(raw, dtype=np.int64)
    if not all(ints(row) and len(row) == 3 for row in raw):
        problems.append(f"{name}: setiap baris harus [r, g, b] bilangan bulat")
        return None
    return np.array(raw, dtype=np.int64).reshape(len(raw), 3)


def _check_meta(meta: Any, problems: list[str]) -> bool:
    if not isinstance(meta, dict):
        problems.append("meta.json: harus berupa objek JSON")
        return False
    ok = True
    for key in ("contract_version", "k", "width", "height"):
        if key not in meta:
            problems.append(f"meta.json: kunci wajib '{key}' tidak ada")
            ok = False
        elif not _is_int(meta[key]):
            problems.append(f"meta.json: '{key}' harus bilangan bulat, ditemukan {meta[key]!r}")
            ok = False
    for key in ("mode", "kmeans_device"):
        if key not in meta:
            problems.append(f"meta.json: kunci wajib '{key}' tidak ada")
            ok = False
        elif not isinstance(meta[key], str):
            problems.append(f"meta.json: '{key}' harus teks, ditemukan {meta[key]!r}")
            ok = False
    return ok


def _fallback_devices(meta: dict) -> dict[str, str]:
    """Perangkat per tahap bila timing.json tidak punya 'devices' (plan 4.3)."""
    io_device = "serial" if meta["mode"] == "cpu" else "omp"
    return {
        "baca": io_device,
        "inisialisasi": "serial",
        "kmeans": meta["kmeans_device"],
        "pewarnaan": io_device,
        "tulis": io_device,
    }


def _parse_timing(
    timing: Any, meta: dict | None, problems: list[str]
) -> tuple[dict[str, StageTiming], float] | None:
    """meta None berarti meta.json bermasalah: timing tetap diperiksa agar semua
    masalah terlapor sekaligus, tetapi hasilnya tidak dibentuk."""
    if not isinstance(timing, dict):
        problems.append("timing.json: harus berupa objek JSON")
        return None
    start = len(problems)

    unit = timing.get("unit", "ms")
    if unit != "ms":
        problems.append(f"timing.json: unit harus 'ms', ditemukan {unit!r}")

    def number(key: str, required: bool) -> float | None:
        if key not in timing:
            if required:
                problems.append(f"timing.json: kunci wajib '{key}' tidak ada")
            return None
        if not _is_number(timing[key]):
            problems.append(f"timing.json: '{key}' harus angka, ditemukan {timing[key]!r}")
            return None
        return float(timing[key])

    ms = {stage: number(field, True) for stage, field in CORE_FIELDS.items()}
    total_ms = number("total_ms", True)
    detail_ms = {name: number(field, False) for name, field in DETAIL_FIELDS.items()}

    devices = _fallback_devices(meta) if meta is not None else {}
    raw_devices = timing.get("devices")
    if raw_devices is not None:
        if not isinstance(raw_devices, dict):
            problems.append("timing.json: 'devices' harus berupa objek JSON")
        else:
            for key, value in raw_devices.items():
                if key not in DEVICE_KEYS:
                    problems.append(
                        f"timing.json: devices punya kunci tidak dikenal {key!r} "
                        f"(yang dikenal: {', '.join(DEVICE_KEYS)})"
                    )
                elif not isinstance(value, str):
                    problems.append(f"timing.json: devices.{key} harus teks, ditemukan {value!r}")
                else:
                    devices[DEVICE_KEYS[key]] = value

    if len(problems) > start or meta is None:
        return None

    stages: dict[str, StageTiming] = {}
    for stage in STAGES:
        detail = None
        if stage == "kmeans" and devices[stage] == "cuda":
            detail = {n: (v or 0.0) / _MS_PER_S for n, v in detail_ms.items()}
        stages[stage] = StageTiming(
            time=ms[stage] / _MS_PER_S, device=devices[stage], detail=detail
        )
    return stages, total_ms / _MS_PER_S


def read_run(path: Path | str) -> RunResult:
    """Baca satu folder run. Melempar ContractError bila strukturnya rusak."""
    path = Path(path)
    if not path.is_dir():
        raise ContractError(path, ["folder run tidak ditemukan"])

    problems = [f"file wajib tidak ada: {n}" for n in RUN_FILES if not (path / n).is_file()]

    meta = _read_json(path, "meta.json", problems)
    timing = _read_json(path, "timing.json", problems)
    centroids = _int_array(_read_json(path, "centroids.json", problems), "centroids.json", 2, problems)
    counts = _int_array(_read_json(path, "counts.json", problems), "counts.json", 1, problems)
    labels = _read_png(path, "labels_preview.png", "labels", problems)
    colors = _read_png(path, "map_preview.png", "map", problems)

    parsed = None
    # meta/timing None berarti file hilang atau rusak: sudah dicatat di atas.
    meta_ok = meta is not None and _check_meta(meta, problems)
    if timing is not None:
        parsed = _parse_timing(timing, meta if meta_ok else None, problems)

    if problems:
        raise ContractError(path, problems)
    assert parsed is not None  # tanpa masalah, meta dan timing pasti terbaca
    stages, total = parsed
    return RunResult(
        path=path,
        meta=meta,
        stages=stages,
        total=total,
        centroids=centroids,
        counts=counts,
        labels_preview=labels,
        map_preview=colors,
    )


def read_batch(path: Path | str) -> BatchResult:
    """Baca folder batch (plan 4.4): batch.json + satu folder run per wilayah."""
    path = Path(path)
    problems: list[str] = []
    raw = _read_json(path, "batch.json", problems)
    if not (path / "batch.json").is_file():
        problems.append("file wajib tidak ada: batch.json")
    if raw is not None and not isinstance(raw, dict):
        problems.append("batch.json: harus berupa objek JSON")
        raw = None
    names: list[str] = []
    if isinstance(raw, dict):
        regions = raw.get("regions")
        if not isinstance(regions, list) or not regions or not all(isinstance(r, str) for r in regions):
            problems.append("batch.json: 'regions' harus daftar nama wilayah (teks) yang tidak kosong")
        else:
            names = regions
            for r in names:
                if Path(r).name != r or r in ("", ".", ".."):
                    problems.append(f"batch.json: nama wilayah tidak aman: {r!r}")
    total: float | None = None
    if isinstance(raw, dict):
        # Plan 4.4 menulis "total": 9.8 (tanpa _ms, tampak detik). Yang dibaca:
        # total_ms (ms, sesuai kesepakatan 21 Nov), cadangan 'total' dianggap detik.
        if "total_ms" in raw and _is_number(raw["total_ms"]):
            total = float(raw["total_ms"]) / _MS_PER_S
        elif "total" in raw and _is_number(raw["total"]):
            total = float(raw["total"])
        elif "total_ms" in raw or "total" in raw:
            problems.append("batch.json: 'total_ms'/'total' harus angka")
    if problems:
        raise ContractError(path, problems)

    runs: list[RunResult] = []
    for name in names:
        try:
            runs.append(read_run(path / name))
        except ContractError as e:
            problems.extend(f"[{name}] {p}" for p in e.problems)
    if problems:
        raise ContractError(path, problems)
    pipelined = bool(raw.get("pipelined", False))
    return BatchResult(
        path=path, names=tuple(names), runs=tuple(runs), total=total, pipelined=pipelined, meta=raw
    )


def read_any(path: Path | str) -> RunResult | BatchResult:
    """Folder dengan batch.json dibaca sebagai batch, selain itu sebagai satu run."""
    path = Path(path)
    if (path / "batch.json").is_file():
        return read_batch(path)
    return read_run(path)


# --------------------------------------------------------------------------- #
# Validasi
# --------------------------------------------------------------------------- #

def validate(run: RunResult) -> list[str]:
    """Daftar masalah pada isi `run`; kosong berarti valid.

    Sengaja TIDAK memeriksa total >= jumlah tahap: pada L3 tahap tumpang tindih,
    dan di WSL durasi cudaEvent (kmeans_ms) tercatat 1-1,5 % lebih besar daripada
    waktu dinding. Nilai counts = 0 juga sah (kontrak U2: K > jumlah warna unik).
    """
    problems: list[str] = []

    for name in RUN_FILES:
        if not (run.path / name).is_file():
            problems.append(f"file wajib tidak ada: {name}")

    version = run.meta.get("contract_version")
    if version not in KNOWN_CONTRACT_VERSIONS:
        problems.append(
            f"contract_version {version!r} tidak dikenal "
            f"(yang dikenal: {sorted(KNOWN_CONTRACT_VERSIONS)})"
        )
    if run.mode not in MODES:
        problems.append(f"mode {run.mode!r} tidak dikenal (harus salah satu dari {sorted(MODES)})")
    if run.kmeans_device not in DEVICES:
        problems.append(
            f"kmeans_device {run.kmeans_device!r} di meta.json tidak dikenal "
            f"(harus salah satu dari {sorted(DEVICES)})"
        )

    k = run.k
    if k < 1:
        problems.append(f"k harus >= 1, ditemukan {k}")
    if run.width < 1 or run.height < 1:
        problems.append(f"ukuran citra harus positif, ditemukan {run.width}x{run.height}")

    if run.centroids.ndim != 2 or run.centroids.shape[1] != 3:
        problems.append(f"centroids harus berbentuk (K, 3), ditemukan {run.centroids.shape}")
    if len(run.centroids) != k:
        problems.append(f"jumlah centroid ({len(run.centroids)}) tidak sama dengan k ({k})")
    if len(run.counts) != k:
        problems.append(f"jumlah counts ({len(run.counts)}) tidak sama dengan k ({k})")
    if run.centroids.size and (run.centroids.min() < 0 or run.centroids.max() > 255):
        problems.append(
            f"nilai centroid harus 0-255, ditemukan {run.centroids.min()}..{run.centroids.max()}"
        )

    if run.counts.size and run.counts.min() < 0:
        problems.append(f"counts tidak boleh negatif, ditemukan min {run.counts.min()}")
    if int(run.counts.sum()) != run.n_pixels:
        problems.append(
            f"jumlah counts ({int(run.counts.sum())}) tidak sama dengan "
            f"width*height ({run.n_pixels})"
        )

    if run.labels_preview.ndim != 2 or run.labels_preview.size == 0:
        problems.append(f"labels_preview harus gambar 2D tidak kosong, ditemukan {run.labels_preview.shape}")
    elif int(run.labels_preview.max()) >= k:
        problems.append(
            f"labels_preview berisi id cluster {int(run.labels_preview.max())} (harus < k = {k})"
        )
    if run.map_preview.shape[:2] != run.labels_preview.shape[:2]:
        problems.append(
            f"ukuran map_preview {run.map_preview.shape[:2]} tidak sama dengan "
            f"labels_preview {run.labels_preview.shape[:2]}"
        )

    unknown = sorted(set(run.stages) - set(STAGES))
    if unknown:
        problems.append(f"nama tahap tidak dikenal: {', '.join(unknown)}")
    absent = [s for s in STAGES if s not in run.stages]
    if absent:
        problems.append(f"tahap tidak ada: {', '.join(absent)}")
    for name, st in run.stages.items():
        if st.device not in DEVICES:
            problems.append(
                f"perangkat tahap {name!r} tidak dikenal: {st.device!r} "
                f"(harus salah satu dari {sorted(DEVICES)})"
            )
        if not (math.isfinite(st.time) and st.time >= 0):
            problems.append(f"waktu tahap {name!r} tidak sah: {st.time}")

    if not (math.isfinite(run.total) and run.total > 0):
        problems.append(f"total harus > 0, ditemukan {run.total}")

    return problems


def validate_batch(batch: BatchResult) -> list[str]:
    """Masalah semua wilayah (diberi awalan nama wilayah) + total batch."""
    problems: list[str] = []
    for name, run in zip(batch.names, batch.runs):
        problems.extend(f"[{name}] {p}" for p in validate(run))
    if batch.total is not None and not (math.isfinite(batch.total) and batch.total > 0):
        problems.append(f"total batch harus > 0, ditemukan {batch.total}")
    return problems
