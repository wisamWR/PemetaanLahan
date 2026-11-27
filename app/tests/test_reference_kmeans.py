"""Tes K-Means referensi (plan Langkah 3). Kasus kecil dihitung tangan; komentar
menunjukkan hitungannya. Meniru tests/test_ref.cpp: determinisme, dua warna, tepi."""

from __future__ import annotations

import numpy as np
import pytest

from landcover.reference_kmeans import (
    KMEANS_MAX_K,
    SplitMix64,
    init_centroids,
    kmeans_int,
)


def reds(*values: int) -> np.ndarray:
    """Piksel (N, 3) uint8 dengan hanya channel merah terisi."""
    return np.array([[v, 0, 0] for v in values], dtype=np.uint8)


def cen(*values: int) -> np.ndarray:
    return np.array([[v, 0, 0] for v in values], dtype=np.int64)


# --------------------------------------------------------------------------- #
# splitmix64 dan inisialisasi
# --------------------------------------------------------------------------- #

def test_splitmix64_vektor_baku_seed_0():
    # Vektor keluaran splitmix64 baku (Vigna) untuk state awal 0.
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [
        16294208416658607535, 7960286522194355700, 487617019471545679,
    ]


def test_splitmix64_tetap_dalam_64_bit_dan_seed_uint32():
    rng = SplitMix64(0xFFFFFFFF)
    vals = [rng.next() for _ in range(1000)]
    assert all(0 <= v < 2**64 for v in vals)
    assert len(set(vals)) == 1000


def test_init_mengambil_piksel_pada_indeks_urutan_prng():
    # n besar: tabrakan indeks praktis mustahil, jadi idx_c = next() % n berurutan.
    n, k, seed = 1_000_003, 8, 42
    px = np.arange(n * 3, dtype=np.uint64).astype(np.uint8).reshape(n, 3)
    rng = SplitMix64(seed)
    idx = [rng.next() % n for _ in range(k)]
    assert len(set(idx)) == k
    assert np.array_equal(init_centroids(px, k, seed), px[idx].astype(np.int64))


@pytest.mark.parametrize("n", [2, 3, 5, 16])
def test_init_k_sama_n_memilih_n_piksel_berbeda(n):
    px = reds(*[i * 10 for i in range(n)])         # n warna unik
    got = init_centroids(px, n, 42)
    assert sorted(got[:, 0].tolist()) == [i * 10 for i in range(n)]


def test_init_citra_1x1_dengan_k_16_menerima_duplikat():
    got = init_centroids(np.array([[1, 2, 3]], dtype=np.uint8), 16, 42)
    assert got.shape == (16, 3) and (got == [1, 2, 3]).all()


def test_init_deterministik_dan_bergantung_seed():
    rng = np.random.default_rng(5)
    px = rng.integers(0, 256, size=(5000, 3), dtype=np.uint8)
    a, b = init_centroids(px, 8, 42), init_centroids(px, 8, 42)
    assert np.array_equal(a, b) and a.dtype == np.int64
    assert not np.array_equal(a, init_centroids(px, 8, 43))


def test_init_menolak_argumen_tidak_sah():
    px = reds(1, 2, 3, 4)
    for bad in (
        lambda: init_centroids(px, 0, 42),
        lambda: init_centroids(px, KMEANS_MAX_K + 1, 42),
        lambda: init_centroids(px, 2, -1),
        lambda: init_centroids(px, 2, 2**32),
        lambda: init_centroids(px[:0], 2, 42),                       # n = 0
        lambda: init_centroids(px.astype(np.int64), 2, 42),          # bukan uint8
        lambda: init_centroids(px.reshape(2, 2, 3), 2, 42),          # harus (N, 3), bukan (H, W, 3)
        lambda: init_centroids(px, 2.0, 42),
    ):
        with pytest.raises(ValueError):
            bad()


# --------------------------------------------------------------------------- #
# Aturan K-Means (kasus tangan)
# --------------------------------------------------------------------------- #

def test_centroid_dibulatkan_ke_bawah():
    # Satu cluster {0, 1, 1}: rata-rata 2/3. Dibulatkan = 1, dipotong = 0; kontrak 3.4 -> 0.
    labels, c, counts = kmeans_int(reds(0, 1, 1), cen(0), 1)
    assert c.tolist() == [[0, 0, 0]]
    assert counts.tolist() == [3] and labels.tolist() == [0, 0, 0]


def test_tie_break_ke_indeks_terkecil():
    # Piksel 1 berjarak 1 ke centroid 0 dan 1 ke centroid 2 -> label 0 (bukan 1).
    labels, c, counts = kmeans_int(reds(1), cen(0, 2), 1)
    assert labels.tolist() == [0]
    assert c.tolist() == [[1, 0, 0], [2, 0, 0]]      # c0 pindah ke 1, c1 kosong tetap 2
    assert counts.tolist() == [1, 0]


def test_tie_break_juga_di_tiga_channel():
    # (1,1,1) berjarak 3 ke (2,2,2) dan 3 ke (0,0,0): seri -> indeks 0, yaitu (2,2,2),
    # walaupun centroid indeks 1 sama dekatnya.
    px = np.array([[1, 1, 1]], dtype=np.uint8)
    labels, _, counts = kmeans_int(px, np.array([[2, 2, 2], [0, 0, 0]]), 1)
    assert labels.tolist() == [0] and counts.tolist() == [1, 0]


def test_cluster_kosong_mempertahankan_centroid_lama():
    px = reds(10, 10, 10, 10)
    labels, c, counts = kmeans_int(px, cen(10, 250), 3)
    assert counts.tolist() == [4, 0]
    assert c.tolist() == [[10, 0, 0], [250, 0, 0]]   # cluster 1 tidak pernah berubah
    assert labels.tolist() == [0, 0, 0, 0]


def test_centroid_awal_kembar_indeks_besar_kosong_pada_assign_pertama():
    # Dua centroid awal sama (5) -> seri -> semua piksel ke indeks 0. Update: c0 = 19 // 3 = 6,
    # c1 kosong tetap 5.
    px = reds(5, 5, 9)
    labels, c, counts = kmeans_int(px, cen(5, 5), 1)
    assert labels.tolist() == [0, 0, 0] and counts.tolist() == [3, 0]
    assert c[:, 0].tolist() == [6, 5]
    # Setelah c0 bergeser, c1 (5) bisa menarik piksel: iterasi 2 -> 5 ke c1 (jarak 0 lawan 1),
    # 9 ke c0 (9 lawan 16); update c0 = 9, c1 = 5. Jadi "selalu kosong" TIDAK berlaku umum.
    labels2, c2, counts2 = kmeans_int(px, cen(5, 5), 2)
    assert labels2.tolist() == [1, 1, 0] and counts2.tolist() == [1, 2]
    assert c2[:, 0].tolist() == [9, 5]


def test_centroid_awal_kembar_pada_citra_seragam_tetap_kosong():
    labels, c, counts = kmeans_int(reds(5, 5, 5), cen(5, 5), 4)
    assert labels.tolist() == [0, 0, 0] and counts.tolist() == [3, 0]
    assert c[:, 0].tolist() == [5, 5]


def test_dua_cluster_beberapa_iterasi():
    # Awal [0, 12]: {0,1,2} -> c0, {9,10,12} -> c1 (9: jarak 81 ke 0, 9 ke 12).
    # Update: c0 = 3 // 3 = 1; c1 = 31 // 3 = 10 (dipotong dari 10,33). Iterasi 2 sama.
    labels, c, counts = kmeans_int(reds(0, 1, 2, 9, 10, 12), cen(0, 12), 2)
    assert labels.tolist() == [0, 0, 0, 1, 1, 1]
    assert c[:, 0].tolist() == [1, 10] and counts.tolist() == [3, 3]


def test_label_dari_assign_terakhir_bukan_dihitung_ulang_terhadap_centroid_akhir():
    px = reds(0, 1, 5, 6, 100)
    # 1 iterasi dari [0, 1]: assign [0,1,1,1,1]; update c0 = 0, c1 = 112 // 4 = 28.
    l1, c1, n1 = kmeans_int(px, cen(0, 1), 1)
    assert l1.tolist() == [0, 1, 1, 1, 1]
    assert c1[:, 0].tolist() == [0, 28] and n1.tolist() == [1, 4]
    # 2 iterasi: assign terhadap [0, 28] -> [0,0,0,0,1]; update c0 = 12 // 4 = 3, c1 = 100.
    l2, c2, n2 = kmeans_int(px, cen(0, 1), 2)
    assert l2.tolist() == [0, 0, 0, 0, 1]
    assert c2[:, 0].tolist() == [3, 100] and n2.tolist() == [4, 1]


def test_tidak_ada_penghentian_dini():
    # Sudah konvergen setelah iterasi 1, hasil tetap sama untuk 1 dan 50 iterasi.
    px = reds(0, 1, 2, 9, 10, 12)
    a = kmeans_int(px, cen(0, 12), 1)
    b = kmeans_int(px, cen(0, 12), 50)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_tanpa_wraparound_uint8():
    # 250 - 5 pada uint8 akan wraparound; integer 32 bit benar: terdekat = centroid 250.
    labels, _, _ = kmeans_int(reds(250), cen(5, 251), 1)
    assert labels.tolist() == [1]


def test_dua_warna_k2_centroid_tepat_dua_warna():
    # Setara test_two_colors di tests/test_ref.cpp.
    w, h = 40, 10
    left = np.arange(w * h) % w < w // 2
    px = np.where(left[:, None], [10, 120, 30], [250, 240, 5]).astype(np.uint8)
    init = init_centroids(px, 2, 42)
    labels, c, counts = kmeans_int(px, init, 20)
    assert sorted(c.tolist()) == [[10, 120, 30], [250, 240, 5]]
    assert (c[labels] == px).all()
    assert sorted(counts.tolist()) == [200, 200]


def test_citra_1x1_dengan_k_16_semua_label_0():
    px = np.array([[1, 2, 3]], dtype=np.uint8)
    init = init_centroids(px, 16, 42)
    labels, c, counts = kmeans_int(px, init, 20)
    assert labels.tolist() == [0]
    assert (c == [1, 2, 3]).all()
    assert counts.tolist() == [1] + [0] * 15


@pytest.mark.parametrize("seed_img,k", [(7, 8), (11, 16), (3, 1), (5, 2)])
def test_determinisme_dan_bentuk_keluaran(seed_img, k):
    img = np.random.default_rng(seed_img).integers(0, 256, size=(131 * 20, 3), dtype=np.uint8)
    init = init_centroids(img, k, 42)
    a = kmeans_int(img, init, 20)
    b = kmeans_int(img, init, 20)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    labels, c, counts = a
    assert labels.dtype == np.uint8 and labels.shape == (len(img),) and labels.max() < k
    assert c.dtype == np.int64 and c.shape == (k, 3) and c.min() >= 0 and c.max() <= 255
    assert counts.dtype == np.int64 and counts.sum() == len(img)
    assert np.array_equal(counts, np.bincount(labels, minlength=k))


def test_hasil_tidak_bergantung_ukuran_potongan(monkeypatch):
    import landcover.reference_kmeans as rk

    img = np.random.default_rng(1).integers(0, 256, size=(1000, 3), dtype=np.uint8)
    init = init_centroids(img, 8, 42)
    whole = kmeans_int(img, init, 5)
    monkeypatch.setattr(rk, "_CHUNK", 37)
    pieces = kmeans_int(img, init, 5)
    assert all(np.array_equal(x, y) for x, y in zip(whole, pieces))


def test_cocok_dengan_perhitungan_naif():
    # Pembanding naif (loop Python murni, aturan kontrak ditulis langsung) pada citra kecil.
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(300, 3), dtype=np.uint8)
    k, iters = 5, 6
    c = init_centroids(img, k, 42).tolist()
    naive_labels = []
    for _ in range(iters):
        sums = [[0, 0, 0] for _ in range(k)]
        cnt = [0] * k
        naive_labels = []
        for r, g, b in img.tolist():
            best, best_d = 0, 2**31 - 1
            for i in range(k):
                d = (r - c[i][0]) ** 2 + (g - c[i][1]) ** 2 + (b - c[i][2]) ** 2
                if d < best_d:
                    best, best_d = i, d
            naive_labels.append(best)
            cnt[best] += 1
            for ch, v in enumerate((r, g, b)):
                sums[best][ch] += v
        for i in range(k):
            if cnt[i]:
                c[i] = [s // cnt[i] for s in sums[i]]
    labels, cents, counts = kmeans_int(img, init_centroids(img, k, 42), iters)
    assert labels.tolist() == naive_labels
    assert cents.tolist() == c
    assert counts.tolist() == [naive_labels.count(i) for i in range(k)]


def test_menolak_argumen_tidak_sah():
    px = reds(1, 2, 3, 4)
    for bad in (
        lambda: kmeans_int(px, cen(1), 0),                            # iters < 1
        lambda: kmeans_int(px, cen(*range(KMEANS_MAX_K + 1)), 1),     # K = 17
        lambda: kmeans_int(px, np.zeros((0, 3), dtype=np.int64), 1),  # K = 0
        lambda: kmeans_int(px, np.array([[300, 0, 0]]), 1),           # di luar 0..255
        lambda: kmeans_int(px, np.array([[1, 2]]), 1),                # bukan (K, 3)
        lambda: kmeans_int(px[:0], cen(1), 1),                        # n = 0
        lambda: kmeans_int(px.astype(np.int32), cen(1), 1),           # bukan uint8
    ):
        with pytest.raises(ValueError):
            bad()
