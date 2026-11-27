# Ambang pemilihan mode otomatis (`--kmeans auto`)

Pemilik: Gavra. Kode: `kAutoCudaThresholdWork` di `src/cli/landcover.cpp`.
Penurunan yang bisa diulang: `python3 bench/fit_auto_threshold.py`.

## Aturan
CUDA dipilih bila **N × K ≥ 70 juta** (N = jumlah piksel, K = jumlah cluster). Selain itu OpenMP.
Bila CUDA tidak terhubung pada build, jatuh ke OpenMP; bila OpenMP juga tidak ada, ke serial.

## Dasar
- K-Means omp tumbuh sebanding N×K (±5 ns per piksel-cluster pada 8 thread), sedangkan K-Means
  CUDA hampir tidak bergantung K (±4,6 ns per piksel). Karena itu titik impas bergantung N×K, bukan N saja.
- Tiap run UI/CLI menjalankan proses `landcover` baru, jadi inisialisasi konteks CUDA (±200-300 ms,
  tidak masuk `total_ms`) dirasakan pengguna. Yang dibandingkan adalah waktu dinding proses
  (kolom "Proses" di `docs/BENCH_tabel.md`), bukan `total_ms`.
- Data: `bench/results/ambang_summary.csv`, 21 konfigurasi (N 0,1-12 juta, K 4/8/16), omp terbaik vs cuda terbaik.

| Aturan | Tercepat (Proses) | Rugi waktu | Tercepat (Total) |
|---|---|---|---|
| lama: N ≥ 100.000 px | 5/21 | 2585 ms | 21/21 |
| **baru: N×K ≥ 70 juta** | **21/21** | **0 ms** | 5/21 |
| selalu omp | 16/21 | 1427 ms | 0/21 |

Ambang lama benar menurut `total_ms` (tanpa pemanasan CUDA), tetapi salah menurut waktu yang
dirasakan pengguna: untuk 0,1-6 juta piksel pada K 4/8, omp lebih cepat 100-300 ms.
Setiap T antara 65 dan 72 juta memberi hasil sama pada data ini; dipilih 70 juta.

## Keterbatasan
- Satu mesin (RTX 3050 Laptop + i7-13620H), citra sintetis, 21 titik. Ukur ulang di mesin uji lain dan dengan Sentinel-2 asli.
- Margin tipis di dekat ambang (selisih omp vs cuda 78-84 ms pada 64 dan 72 juta).
- K di bawah 4 dan N di atas 12 juta pada K<8 belum diukur; CUDA diperkirakan unggul pada N×K besar (7680×4320, K=8 = 265 juta).
- Bila aplikasi suatu saat memakai proses CUDA yang menetap (pemanasan sekali), ambang ikut `total_ms` dan harus dihitung ulang.
- `--info` tetap mengeluarkan `auto_cuda_threshold_pixels` (kini 8.750.000 = ambang pada K bawaan 8) dan menambah `auto_cuda_threshold_work`.
