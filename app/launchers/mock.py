"""MockLauncher: menulis folder run kontrak 4.3 dengan data TIRUAN (plan Langkah 3).

Yang nyata: peta kelas. K-Means integer (`reference_kmeans`, aturan sama dengan core)
dijalankan pada pratinjau <= 512 px, jadi peta, centroid, dan counts konsisten satu
sama lain, dan mode `cpu` dan `opt` menghasilkan label IDENTIK.
Yang palsu: waktu per tahap (model biaya sederhana + jitter +-5 %) dan ukuran
citra bila dataset tidak ada. Semuanya bertanda `"mock": true` di meta.json;
bukan hasil ukur dan bukan prediksi.

Citra dipilih dari data/<dataset>/: `preview.jpg`, bila tidak ada dan tanpa
manifest, satu-satunya `*.png`; bila tidak ada, citra sintetis (noise halus menjadi
air, vegetasi, tanah, batuan) yang HANYA bergantung pada nama dataset. Ukuran
penuh dari manifest.json bila ada; bila tidak, ukuran citra; bila tidak ada citra,
DEFAULT_SIZE (satu tile Sentinel-2).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from landcover.reference_kmeans import init_centroids, kmeans_int

from .base import ProgressCallback, RunParams, run_id

DEFAULT_SIZE = (10980, 10980)        # lebar, tinggi satu tile Sentinel-2 (10 m)
DEFAULT_PIXEL_SIZE_M = 10
PREVIEW_MAX_SIDE = 512               # K-Means tiruan dijalankan pada citra sekecil ini
# Asumsi KHUSUS tiruan untuk `--kmeans auto`; ambang sebenarnya dari Gavra (#10).
AUTO_CUDA_THRESHOLD_PIXELS = 8_000_000
SLEEP_ENV = "LC_MOCK_SLEEP"

STAGE_NAMES = ("baca", "inisialisasi", "kmeans", "pewarnaan", "tulis")

# Model biaya (ns per piksel pada satu inti, kecuali dinyatakan lain). Hanya agar pola
# kasarnya mirip hipotesis I.4 (K-Means CUDA ~20x, total ~5-6x); BUKAN prediksi.
_READ_NS = 70.0
_WRITE_NS = 90.0
_COLOR_NS = 6.0
_ASSIGN_NS_PER_CLUSTER = 1.6         # per piksel, per cluster, per iterasi
_UPDATE_NS = 3.0                     # per piksel per iterasi
_P_IO = 0.70                         # bagian paralel baca/tulis PNG (dipotong-potong)
_P_COLOR = 0.95
_P_KMEANS_OMP = 0.95
_CUDA_KERNEL_SPEEDUP = 25.0          # kernel dibanding K-Means serial
_PCIE_BYTES_PER_MS = 6e6             # ~6 GB/s
_CUDA_FIXED_MS = 5.0                 # konteks + peluncuran kernel
_OMP_FIXED_MS = 0.2
_JITTER = 0.05

_SYNTH_PALETTE = np.array(           # air, vegetasi, tanah/cokelat, batuan/abu-abu
    [[40, 90, 160], [50, 110, 50], [150, 110, 70], [150, 150, 150]], dtype=np.int16
)
_SYNTH_QUANTILES = (0.15, 0.65, 0.90)    # porsi: air 15 %, vegetasi 50 %, tanah 25 %, batuan 10 %


# --------------------------------------------------------------------------- #
# Fungsi murni (diuji langsung)
# --------------------------------------------------------------------------- #

def scale_counts(counts: np.ndarray, n_total: int) -> np.ndarray:
    """Skalakan counts pratinjau ke `n_total` piksel: `c * n_total // n_pratinjau`
    (aritmetika eksak), lalu sisa pembulatan masuk ke cluster terbesar (indeks
    terkecil bila seri) sehingga jumlahnya tepat n_total. Nilai 0 tetap 0."""
    cnt = [int(c) for c in np.asarray(counts).tolist()]
    n_prev = sum(cnt)
    if n_prev <= 0 or any(c < 0 for c in cnt):
        raise ValueError("counts harus tidak negatif dengan jumlah > 0")
    if n_total < 1:
        raise ValueError(f"n_total harus >= 1, ditemukan {n_total}")
    scaled = [c * n_total // n_prev for c in cnt]
    scaled[scaled.index(max(scaled))] += n_total - sum(scaled)
    return np.array(scaled, dtype=np.int64)


def synthetic_image(dataset: str, width: int, height: int) -> np.ndarray:
    """Citra (height, width, 3) uint8 dari noise halus; deterministik dan hanya
    bergantung pada nama dataset (crc32, bukan hash() yang diacak per proses)."""
    rng = np.random.default_rng(zlib.crc32(dataset.encode("utf-8")))
    field = np.zeros((height, width), dtype=np.float32)
    for cells, weight in ((6, 0.65), (16, 0.35)):
        grid = rng.random((max(2, cells * height // max(width, height)),
                           max(2, cells * width // max(width, height)))).astype(np.float32)
        field += weight * np.asarray(Image.fromarray(grid).resize((width, height), Image.BILINEAR))
    field += 0.03 * rng.standard_normal((height, width)).astype(np.float32)
    # Ambang berbasis kuantil: keempat tutupan selalu ada dengan porsi tetap.
    cls = np.digitize(field, np.quantile(field, _SYNTH_QUANTILES))
    jitter = rng.integers(-8, 9, size=(height, width, 3))
    return np.clip(_SYNTH_PALETTE[cls] + jitter, 0, 255).astype(np.uint8)


def resolve_device(p: RunParams, n_pixels: int) -> tuple[str, str | None]:
    """(kmeans_device konkret, kmeans_auto_reason). `auto` TIDAK boleh tertulis di meta."""
    if p.mode == "cpu":
        return "serial", None
    if p.kmeans in ("omp", "cuda"):
        return p.kmeans, None
    juta = f"{n_pixels / 1e6:.1f}".replace(".", ",")
    batas = f"{AUTO_CUDA_THRESHOLD_PIXELS / 1e6:g}".replace(".", ",")
    if n_pixels >= AUTO_CUDA_THRESHOLD_PIXELS:
        return "cuda", f"N = {juta} juta piksel ≥ ambang {batas} juta (ambang tiruan)"
    return "omp", f"N = {juta} juta piksel < ambang {batas} juta (ambang tiruan)"


def _speedup(threads: int, parallel_fraction: float) -> float:
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / threads)


def fake_timing(
    p: RunParams, device: str, width: int, height: int, jitter_seed: str
) -> dict:
    """Isi timing.json (termasuk `devices`) dari model biaya: milidetik, kunci `_ms`
    (kontrak U5). Jitter +-5 % per komponen, deterministik dari `jitter_seed`."""
    n = float(width) * float(height)
    rnd = random.Random(jitter_seed)

    def jit(ms: float) -> float:
        return ms * (1.0 + rnd.uniform(-_JITTER, _JITTER))

    ns_ms = 1e-6                                      # ns -> ms
    cpu_kmeans = n * p.iters * (p.k * _ASSIGN_NS_PER_CLUSTER + _UPDATE_NS) * ns_ms
    read, write, color = n * _READ_NS * ns_ms, n * _WRITE_NS * ns_ms, n * _COLOR_NS * ns_ms
    if p.mode == "opt":
        read /= _speedup(p.threads, _P_IO)
        write /= _speedup(p.threads, _P_IO)
        color /= _speedup(p.threads, _P_COLOR)

    h2d = kernel = d2h = 0.0
    if device == "cuda":
        h2d = jit(3.0 * n / _PCIE_BYTES_PER_MS)       # RGB uint8
        d2h = jit(4.0 * n / _PCIE_BYTES_PER_MS)       # label int32
        kernel = jit(cpu_kmeans / _CUDA_KERNEL_SPEEDUP)
        kmeans = h2d + kernel + d2h + _CUDA_FIXED_MS
    elif device == "omp":
        kmeans = jit(cpu_kmeans / _speedup(p.threads, _P_KMEANS_OMP) + _OMP_FIXED_MS)
    else:
        kmeans = jit(cpu_kmeans)

    t = {
        "unit": "ms",
        "read_ms": jit(read),
        "init_ms": jit(0.4 + 0.05 * p.k),
        "kmeans_ms": kmeans,
        "color_ms": jit(color),
        "write_ms": jit(write),
    }
    t["total_ms"] = t["read_ms"] + t["init_ms"] + t["kmeans_ms"] + t["color_ms"] + t["write_ms"]
    t.update(h2d_ms=h2d, kernel_ms=kernel, d2h_ms=d2h)
    t = {key: (val if key == "unit" else round(val, 3)) for key, val in t.items()}
    io = "serial" if p.mode == "cpu" else "omp"
    t["devices"] = {"read": io, "init": "serial", "kmeans": device, "color": io, "write": io}
    return t


def _parse_sleep(raw: str | None) -> float:
    if raw is None or raw.strip() == "":
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{SLEEP_ENV}={raw!r} harus angka detik (mis. 1 atau 2.5)") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{SLEEP_ENV}={raw!r} harus angka >= 0")
    return value


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Launcher
# --------------------------------------------------------------------------- #

class MockLauncher:
    """Lihat docstring modul.

    data_dir          folder berisi <dataset>/ (bawaan: config.DATA_DIR).
    break_verification  hanya untuk run mode `opt`: ubah sedikit label pratinjau (dan
                      counts, peta) agar lencana "BEDA" terhadap run `cpu` bisa diuji.
    max_side          sisi terpanjang pratinjau K-Means tiruan (plan: <= 512).
    sleep             total detik tidur selama run (bagi rata antar tahap); None =
                      baca env LC_MOCK_SLEEP (kosong/0 = tanpa tidur).
    sleep_fn          pengganti time.sleep (untuk tes).
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        break_verification: bool = False,
        max_side: int = PREVIEW_MAX_SIDE,
        sleep: float | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if data_dir is None:
            from config import DATA_DIR
            data_dir = DATA_DIR
        if max_side < 1:
            raise ValueError(f"max_side harus >= 1, ditemukan {max_side}")
        self.data_dir = Path(data_dir)
        self.break_verification = bool(break_verification)
        self.max_side = int(max_side)
        self._sleep = sleep
        self._sleep_fn = sleep_fn

    # -- dataset ------------------------------------------------------------ #

    def _load_dataset(self, p: RunParams) -> tuple[int, int, float, np.ndarray, str]:
        """(width, height, pixel_size_m, citra kecil uint8, sumber citra)."""
        folder = self.data_dir / p.dataset
        manifest: dict | None = None
        image_path: Path | None = None
        if folder.is_dir():
            mf = folder / "manifest.json"
            if mf.is_file():
                try:
                    manifest = json.loads(mf.read_text(encoding="utf-8-sig"))
                    w, h = manifest["width"], manifest["height"]
                    ok = isinstance(manifest, dict) and all(
                        isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (w, h)
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    ok = False
                if not ok:
                    raise ValueError(
                        f"{mf}: manifest.json harus objek dengan width dan height bilangan bulat > 0"
                    )
            if (folder / "preview.jpg").is_file():
                image_path = folder / "preview.jpg"
            elif manifest is None:
                pngs = sorted(folder.glob("*.png"))
                if len(pngs) == 1:
                    image_path = pngs[0]

        raw_image = None
        if image_path is not None:
            with Image.open(image_path) as im:
                raw_image = im.convert("RGB")
        if manifest is not None:
            width, height = manifest["width"], manifest["height"]
            pixel_size = manifest.get("pixel_size_m", DEFAULT_PIXEL_SIZE_M)
        elif raw_image is not None:
            (width, height), pixel_size = raw_image.size, DEFAULT_PIXEL_SIZE_M
        else:
            (width, height), pixel_size = DEFAULT_SIZE, DEFAULT_PIXEL_SIZE_M

        scale = min(1.0, self.max_side / max(width, height))
        sw, sh = max(1, round(width * scale)), max(1, round(height * scale))
        if raw_image is not None:
            if raw_image.size != (sw, sh):
                raw_image = raw_image.resize((sw, sh), Image.BILINEAR)
            return width, height, pixel_size, np.asarray(raw_image, dtype=np.uint8), image_path.name
        return width, height, pixel_size, synthetic_image(p.dataset, sw, sh), "synthetic"

    # -- run ---------------------------------------------------------------- #

    def run(
        self,
        params: RunParams,
        out_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        say = on_progress or (lambda _msg: None)
        sleep_total = _parse_sleep(os.getenv(SLEEP_ENV)) if self._sleep is None else float(self._sleep)
        rid = run_id(params)

        if self.break_verification and params.mode == "opt" and params.k < 2:
            raise ValueError("break_verification butuh k >= 2 (dengan k = 1 tidak ada label yang bisa diubah)")

        width, height, pixel_size, small, source = self._load_dataset(params)
        sh, sw = small.shape[:2]
        n_pixels = width * height
        device, reason = resolve_device(params, n_pixels)
        say("[mock] DATA TIRUAN: peta dari K-Means nyata pada pratinjau, waktu dari model biaya")

        # K-Means pada pratinjau: tidak bergantung pada mode, kmeans, atau thread.
        pixels = small.reshape(-1, 3)
        init = init_centroids(pixels, params.k, params.seed)
        labels, centroids, counts_small = kmeans_int(pixels, init, params.iters)
        labels = labels.reshape(sh, sw)

        if self.break_verification and params.mode == "opt":
            labels = self._break_labels(labels, params.k, rid)
            counts_small = np.bincount(labels.ravel(), minlength=params.k).astype(np.int64)
        counts = scale_counts(counts_small, n_pixels)
        colors = centroids[labels].astype(np.uint8)

        timing = fake_timing(params, device, width, height, rid)
        pause = sleep_total / len(STAGE_NAMES)
        for stage, field in zip(
            STAGE_NAMES, ("read_ms", "init_ms", "kmeans_ms", "color_ms", "write_ms")
        ):
            if pause > 0:
                self._sleep_fn(pause)
            say(f"[stage] {stage} selesai {timing[field] / 1000.0:.2f} s")

        meta = {
            "contract_version": 1,
            "mock": True,
            "dataset": params.dataset,
            "mode": params.mode,
            "kmeans_device": device,
            **({"kmeans_auto_reason": reason} if reason else {}),
            "width": width,
            "height": height,
            "pixel_size_m": pixel_size,
            "k": params.k,
            "iters": params.iters,
            "seed": params.seed,
            "threads": {"serial": 1, "omp": params.threads, "cuda": 0}[device],
            "init_centroids": init.tolist(),
            "preview_width": sw,
            "preview_height": sh,
            "source_image": source,
            "host": "tiruan (mock): waktu dari model biaya, bukan hasil ukur",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_atomic(out_dir / "timing.json", _dump(timing))
        _write_atomic(out_dir / "centroids.json", _dump(centroids.tolist()))
        _write_atomic(out_dir / "counts.json", _dump(counts.tolist()))
        _write_png(out_dir / "labels_preview.png", labels.astype(np.uint8))
        _write_png(out_dir / "map_preview.png", colors)
        _write_atomic(out_dir / "meta.json", _dump(meta))   # terakhir: tanda run lengkap
        say(f"[mock] folder run: {out_dir}")
        return out_dir

    @staticmethod
    def _break_labels(labels: np.ndarray, k: int, rid: str) -> np.ndarray:
        """Ubah ~1 % label pratinjau (minimal satu) menjadi (l + 1) % k; deterministik."""
        out = labels.copy()
        flat = out.reshape(-1)
        n_flip = max(1, flat.size // 100)
        rng = np.random.default_rng(zlib.crc32(rid.encode("utf-8")))
        idx = rng.choice(flat.size, size=n_flip, replace=False)
        flat[idx] = (flat[idx].astype(np.int64) + 1) % k
        return out


def _dump(data) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def _write_png(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(array).save(tmp, format="PNG")
    os.replace(tmp, path)
