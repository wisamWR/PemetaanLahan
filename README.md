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

Status saat ini (T5): ada tiga varian dengan hasil identik.

- `kmeans_cuda` (versi utama, T4): akumulasi per blok di shared memory, digabung ke global sekali per blok.
- `kmeans_cuda_simple` (T3): `atomicAdd` langsung ke memori global. Dipertahankan sebagai pembanding dan
  cadangan; deklarasinya ada di `src/cuda/kmeans_cuda_variants.h`, bukan di header kontrak.

- `kmeans_cuda_streams` (L3, T5, mode `cuda_streams` di CSV): kernel sama dengan T4, tetapi citra dikirim
  dan label diambil per potongan lewat memori pinned dan dua stream. Pada pengukuran 24 Nov 2026 varian ini
  **tidak lebih cepat** daripada T4 (alokasi pinned per panggilan lebih mahal daripada yang dihemat dari
  tumpang-tindih), jadi T4 tetap versi utama.

Ketiganya menghasilkan label dan centroid identik dengan referensi CPU dan `kmeans_cpu` (L0) di semua kasus
`make test` dan `make test-large`. Hasil benchmark ada di `bench/results/` (`cuda_simple.csv`,
`cuda_shared.csv`, `cuda_streams.csv`).

Catatan pengukuran: di WSL 2, durasi dari `cudaEvent` (`kmeans_ms`) terukur sekitar 1 sampai 1,5 %
lebih besar daripada waktu dinding host (`total_ms`) untuk panggilan yang sama. Penyebabnya belum
diketahui; anggap selisih sebesar itu sebagai batas ketelitian saat membandingkan angka.

## Menjalankan aplikasi

Aplikasi berupa UI web (React) dengan API FastAPI yang berjalan di Windows dan memanggil program inti
`landcover` di WSL. Halaman **Pemetaan** membuat peta kelas, pelabelan, dan tabel luas; halaman **Kinerja**
membandingkan mode CPU saja dengan mode Optimalisasi beserta verifikasi hasil identik.

Prasyarat (sekali):

1. **WSL:** bangun `landcover` di filesystem Linux WSL (bukan `/mnt/*`). CUDA ikut tertaut bila `nvcc` ada.

   ```bash
   git clone https://github.com/wisamWR/PemetaanLahan.git ~/PemetaanLahan
   cd ~/PemetaanLahan && make landcover
   ./build/landcover --info     # periksa: omp dan cuda terhubung, ambang mode otomatis
   ```

   Bila biner ada di tempat lain, setel `LC_WSL_BIN` (path di dalam WSL) di Windows sebelum menjalankan
   aplikasi. Bawaannya `$HOME/PemetaanLahan/build/landcover`.
2. **Windows, dari root repo:** Python 3.12 atau lebih baru dan Node.js.

   ```powershell
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r app\requirements.txt
   npm --prefix web install
   ```

Menjalankan (satu perintah, dari root repo di Windows):

```powershell
app\tools\run_web.cmd --backend cli --open
```

Perintah ini membangun frontend bila perlu, lalu menyajikan API dan UI di http://127.0.0.1:8000. Citra PNG
dibaca dari `data/` (tidak di-commit; ganti dengan `--data-dir`) atau diunggah dari halaman Pemetaan. Tanpa
`--backend cli`, aplikasi memakai backend tiruan dan semua hasil bertanda DATA TIRUAN.

Rincian: opsi `run_web.cmd`, pembungkus WSL `lc_wsl.cmd`, dan catatan pengukuran di `app/tools/README.md`;
mode pengembangan frontend di `web/README.md`.

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
