"""K-Means integer referensi di NumPy (plan Langkah 3).

Port persis aturan core C++ (docs/KONTRAK_TEKNIS.md butir 3, `tests/ref_kmeans.cpp`,
`src/common/init.cpp`), supaya keluarannya bisa dibandingkan bit demi bit dengan
mode CPU/OpenMP/CUDA pada citra kecil dan dipakai MockLauncher:

  * jarak kuadrat integer, tie-break ke indeks cluster terkecil;
  * jumlah channel dalam int64, centroid baru = sum // count (DIBULATKAN KE BAWAH,
    butir 3.4; di laporan tulis "dibulatkan ke bawah");
  * cluster kosong mempertahankan centroid lama;
  * iterasi tetap `iters` kali, tanpa penghentian dini;
  * label keluaran = hasil assign iterasi TERAKHIR (sebelum update terakhir),
    centroid keluaran = hasil update terakhir. Jadi label tidak dihitung ulang
    terhadap centroid akhir;
  * inisialisasi = splitmix64 + `idx = next() % n` dengan penolakan indeks kembar.

Piksel selalu dibaca row-major: citra (H, W, 3) menjadi `reshape(-1, 3)`.
"""

from __future__ import annotations

import operator

import numpy as np

KMEANS_MAX_K = 16          # batas atas K menurut kontrak (kmeans.h)
MAX_SEED = 0xFFFFFFFF      # seed core bertipe uint32_t

_MASK64 = 0xFFFFFFFFFFFFFFFF
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_INT32_MAX = 0x7FFFFFFF
_CHUNK = 1 << 18           # piksel per potongan: memori O(potongan), bukan O(N*K)


class SplitMix64:
    """splitmix64 persis `init.cpp`. Python int biasa + mask 64 bit (bukan np.uint64,
    supaya tidak ada upcast atau peringatan overflow)."""

    def __init__(self, seed: int):
        self.state = int(seed) & _MASK64

    def next(self) -> int:
        self.state = (self.state + _GOLDEN) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * _MIX1) & _MASK64
        z = ((z ^ (z >> 27)) * _MIX2) & _MASK64
        return z ^ (z >> 31)


# --------------------------------------------------------------------------- #
# Validasi argumen (core menolak dengan kode kembalian != 0; di sini ValueError)
# --------------------------------------------------------------------------- #

def _as_pixels(pixels) -> np.ndarray:
    px = np.asarray(pixels)
    if px.dtype != np.uint8 or px.ndim != 2 or px.shape[1] != 3:
        raise ValueError(
            f"pixels harus array uint8 berbentuk (N, 3), ditemukan {px.dtype} {px.shape}"
        )
    if len(px) == 0:
        raise ValueError("pixels tidak boleh kosong (N = 0)")
    return px


def _check_k(k) -> int:
    try:
        k = operator.index(k)
    except TypeError:
        raise ValueError(f"k harus bilangan bulat, ditemukan {k!r}") from None
    if not 1 <= k <= KMEANS_MAX_K:
        raise ValueError(f"k harus 1..{KMEANS_MAX_K}, ditemukan {k}")
    return k


def _check_seed(seed) -> int:
    try:
        seed = operator.index(seed)
    except TypeError:
        raise ValueError(f"seed harus bilangan bulat, ditemukan {seed!r}") from None
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed harus 0..{MAX_SEED} (uint32), ditemukan {seed}")
    return seed


# --------------------------------------------------------------------------- #
# Inisialisasi dan K-Means
# --------------------------------------------------------------------------- #

def init_centroids(pixels: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Centroid awal (K, 3) int64: port `kmeans_init_centroids` di src/common/init.cpp.

    Untuk c = 0..k-1: `idx = next() % n`; bila idx sudah terpilih DAN jumlah indeks
    berbeda yang sudah terpilih masih < n, buang dan ambil ulang. Bila semua n piksel
    sudah terpilih (k > n, mis. citra 1x1 dengan K=16), duplikat diterima.
    Indeks berbeda belum tentu warna berbeda.
    """
    px = _as_pixels(pixels)
    k = _check_k(k)
    seed = _check_seed(seed)
    n = len(px)

    rng = SplitMix64(seed)
    chosen: list[int] = []
    distinct = 0                      # banyaknya indeks berbeda yang sudah terpilih
    for _ in range(k):
        while True:
            idx = rng.next() % n
            if idx not in chosen:     # `chosen` baru berisi cluster sebelumnya
                distinct += 1
                break
            if distinct >= n:         # semua piksel sudah terpilih: duplikat diizinkan
                break
        chosen.append(idx)
    return px[chosen].astype(np.int64)


def _assign(r: np.ndarray, g: np.ndarray, b: np.ndarray, cen: np.ndarray) -> np.ndarray:
    """Label terdekat per piksel. Channel int32 (jangan uint8: selisihnya wraparound),
    jarak int32, dan perbandingan `<` ketat terhadap yang terbaik sejauh ini, jadi
    seri jatuh ke indeks terkecil, persis kode C++."""
    best = np.zeros(len(r), dtype=np.intp)
    best_d = np.full(len(r), _INT32_MAX, dtype=np.int32)
    for c in range(len(cen)):
        dr = r - np.int32(cen[c, 0])
        dg = g - np.int32(cen[c, 1])
        db = b - np.int32(cen[c, 2])
        d = dr * dr + dg * dg + db * db            # maks 3 * 255^2 = 195075
        better = d < best_d
        best[better] = c
        best_d[better] = d[better]
    return best


def kmeans_int(
    pixels: np.ndarray,   # (N, 3) uint8
    init: np.ndarray,     # (K, 3) int64
    iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K-Means integer. Mengembalikan (labels uint8 (N,), centroids int64 (K,3),
    counts int64 (K,)); counts = jumlah piksel per label keluaran (bisa 0)."""
    px = _as_pixels(pixels)
    cen = np.array(init, dtype=np.int64)
    if cen.ndim != 2 or cen.shape[1] != 3:
        raise ValueError(f"init harus berbentuk (K, 3), ditemukan {cen.shape}")
    k = _check_k(len(cen))
    if cen.min() < 0 or cen.max() > 255:
        raise ValueError(f"nilai init harus 0..255, ditemukan {cen.min()}..{cen.max()}")
    try:
        iters = operator.index(iters)
    except TypeError:
        raise ValueError(f"iters harus bilangan bulat, ditemukan {iters!r}") from None
    if iters < 1:
        raise ValueError(f"iters harus >= 1, ditemukan {iters}")

    n = len(px)
    labels = np.empty(n, dtype=np.uint8)
    for _ in range(iters):
        sums = np.zeros((k, 3), dtype=np.int64)
        counts = np.zeros(k, dtype=np.int64)
        for start in range(0, n, _CHUNK):
            chunk = px[start:start + _CHUNK].astype(np.int32)
            ch = [np.ascontiguousarray(chunk[:, i]) for i in range(3)]
            lab = _assign(ch[0], ch[1], ch[2], cen)
            labels[start:start + len(lab)] = lab
            counts += np.bincount(lab, minlength=k)
            for i in range(3):
                # bincount berbobot = float64: eksak selama jumlah < 2^53 (di sini <= 255 * 2^18).
                s = np.bincount(lab, weights=ch[i], minlength=k)
                sums[:, i] += np.rint(s).astype(np.int64)
        nonempty = counts > 0                        # cluster kosong: centroid tetap
        cen[nonempty] = sums[nonempty] // counts[nonempty][:, None]
    return labels, cen, np.bincount(labels, minlength=k).astype(np.int64)


def nodata_stats(pixels: np.ndarray, labels: np.ndarray) -> tuple[int, int]:
    """Hitung piksel nodata `(0, 0, 0)` dan id cluster-nya (`-1` bila tidak ada).

    Memastikan semua piksel `(0, 0, 0)` mendapat label yang sama (aturan determinisme U7).
    """
    px = _as_pixels(pixels)
    lab = np.asarray(labels)
    if lab.ndim != 1 or len(lab) != len(px):
        raise ValueError(f"labels harus 1-D sepanjang {len(px)}, ditemukan {lab.shape}")
    mask = (px[:, 0] == 0) & (px[:, 1] == 0) & (px[:, 2] == 0)
    nodata_pixels = int(mask.sum())
    if nodata_pixels == 0:
        return 0, -1
    nodata_labels = lab[mask]
    lo, hi = int(nodata_labels.min()), int(nodata_labels.max())
    if lo != hi:
        raise ValueError(f"piksel nodata (0,0,0) mendapat label berbeda: {lo} vs {hi}")
    return nodata_pixels, lo
