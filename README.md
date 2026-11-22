# Pemetaan Cepat Tutupan Lahan (K-Means CPU / OpenMP / CUDA)

Proyek kelompok mata kuliah Arsitektur dan Infrastruktur Berkinerja Tinggi (UGM).
K-Means mengelompokkan piksel citra Sentinel-2 menurut warna RGB menjadi K kelas sebagai
segmentasi awal tanpa pengawasan (bukan klasifikasi akurat). Semua mode (CPU, OpenMP, CUDA)
harus menghasilkan label dan centroid yang identik bit demi bit, mengikuti kontrak teknis
yang disepakati tim (dokumen internal, tidak disimpan di repo ini).

## Lingkungan build

Dipilih: **WSL 2, Ubuntu 24.04**. Windows native tidak dipakai karena `nvcc` di Windows butuh
MSVC (`cl.exe`), dan MSVC tidak terpasang di mesin ini.

Hasil pengecekan 18 November 2026 (mesin Wisam):

| Komponen | Terpasang | Proposal | Catatan |
|---|---|---|---|
| GPU | RTX 3050 6 GB Laptop, CC 8.6, driver 576.93 | RTX 3050 6 GB | sesuai, `-arch=sm_86` |
| CUDA Toolkit (WSL) | 12.9 (V12.9.86) di `/usr/local/cuda-12.9` | 12.8 | beda versi minor; `nvcc` belum ada di `PATH`, Makefile memakai `CUDA_HOME=/usr/local/cuda` |
| GCC (WSL) | 13.3.0 | | didukung CUDA 12.9 |
| GNU Make (WSL) | 4.3 | | |
| `compute-sanitizer` | ada | | |
| `nsys` / `ncu` | **tidak ada** | | perlu dipasang untuk profil T4/T5 |
| CPU | Intel Core i7-13620H | i7-13650HX | **beda dari proposal**, perlu dikoreksi di laporan |
| RAM yang terlihat di WSL | sekitar 7,6 GB | 20 GB | batas bawaan WSL; bisa dinaikkan lewat `.wslconfig` |

Di Windows ada juga MinGW g++ 13.2, hanya dipakai untuk mengecek bahwa referensi CPU
deterministik lintas kompilator.

## Cara build dan uji

Dari terminal WSL, di folder repo (`/mnt/c/Users/wisam/Documents/All_tugas/Sem5/AIBT/PemetaanLahan`):

```bash
make cuda        # bangun objek CUDA dari src/cuda/*.cu
make test        # uji referensi CPU, lalu uji CUDA vs referensi (harus identik)
make test-large  # tambah 7680x4320 dan 10980x10980 (K=8)
make sanitize    # compute-sanitizer memcheck pada citra kecil
make bench       # benchmark versi utama (T4), CSV ditambahkan ke bench/results/cuda_shared.csv
make bench-compare  # benchmark T3 (cuda_simple.csv) dan T4 (cuda_shared.csv) pada citra yang sama
make probe       # analisis bottleneck: assign saja vs T3 vs T4 vs batas bandwidth memori
make clean
```

Parameter benchmark bisa diganti, contoh `make bench BENCH_W=1000 BENCH_H=1000 BENCH_K=4`.

Status saat ini (T4): ada dua varian dengan hasil identik.

- `kmeans_cuda` (versi utama, T4): akumulasi per blok di shared memory, digabung ke global sekali per blok.
- `kmeans_cuda_simple` (T3): `atomicAdd` langsung ke memori global. Dipertahankan sebagai pembanding dan
  cadangan; deklarasinya ada di `src/cuda/kmeans_cuda_variants.h`, bukan di header kontrak.

Keduanya menghasilkan label dan centroid identik dengan referensi CPU di semua kasus `make test` dan
`make test-large`. Hasil benchmark ada di `bench/results/`.

Catatan pengukuran: di WSL 2, durasi dari `cudaEvent` (`kmeans_ms`) terukur sekitar 1 sampai 1,5 %
lebih besar daripada waktu dinding host (`total_ms`) untuk panggilan yang sama. Penyebabnya belum
diketahui; anggap selisih sebesar itu sebagai batas ketelitian saat membandingkan angka.

## Struktur folder

```
src/common/   kmeans.h (antarmuka bersama, kompatibel C), init.cpp (inisialisasi centroid bersama)
src/cpu/      kmeans_cpu, baseline L0 (Azka)
src/omp/      kmeans_omp, L2 (Gavra)
src/cuda/     kmeans_cuda, L1 dan L3 (Wisam)
tests/        ref_kmeans (referensi CPU sementara), test_ref, test_cuda_vs_ref, bench_cuda (Wisam)
bench/        skrip benchmark (Gavra), hasil CSV di bench/results/
data/         citra (tidak di-commit)
```

## Branch

Satu branch per modul: `feature/cpu`, `feature/omp`, `feature/cuda`, `feature/app`.
Merge ke `main` hanya setelah verifikasi label identik terhadap L0 lolos.
