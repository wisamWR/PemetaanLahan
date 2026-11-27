# Profiling L0 (CPU serial) dan Dokumentasi Bottleneck

Pemilik: Gavra. Bahan untuk penempatan tahap L1 (CUDA), L2 (OpenMP), dan L3 (CPU+GPU streams).

## 1. Tujuan
Mengukur waktu tiap tahap pipeline pada baseline L0 (`kmeans_cpu`, 1 thread) untuk mengetahui
tahap mana yang mendominasi, sebagai data "sebelum" optimasi dan dasar keputusan tahap mana yang
layak dipercepat (OpenMP, CUDA, streams).

## 2. Metode
- Program: `build/landcover --mode cpu` (CLI Ridlo), engine `kmeans_cpu` serial, `--threads 1`.
- Pengukuran: `timing.json` dari CLI. Aturan waktu: `read` (baca PNG sampai buffer RGB penuh),
  `init`, `kmeans` (dari StageTimes), `color` (label ke warna centroid), `write` (map.png resolusi
  penuh), `total = read + waktu panggil kmeans + color + write`. `aux_ms` (counts, pratinjau) tidak
  masuk total.
- Protokol: tiap konfigurasi dijalankan 1 kali sebagai pemanasan cache disk (dibuang), lalu N
  pengulangan; yang dilaporkan **median**.
- Parameter tetap: iterasi 20, seed 42, K sesuai tabel.
- Mesin:
  - CPU: Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz (4 Physical Cores / 8 Logical Processors)
  - RAM: 16 GB (~15.77 GB usable)
  - OS: Windows 11 Home / WSL 2 (Kernel 6.18.33.2-2, Ubuntu Linux)
  - Compiler: `g++` (Ubuntu 11.4.0 / C++17)
  - Tanggal pengujian: 27 November 2026
- Citra: Citra sintetis lanskap (`sintetis_lanskap_<lebar>x<tinggi>`) resolusi 1000x1000 (1 MP), 2000x2000 (4 MP), 4000x3000 (12 MP), dan 7680x4320 (8K / 33.2 MP).
- Cara menjalankan (akar repo, WSL, setelah `make landcover`):
  ```bash
  python3 bench/profile_l0.py --sizes 1000x1000,2000x2000,4000x3000,7680x4320 --k 8 --reps 5
  python3 bench/profile_l0.py --sizes 4000x3000 --k 4,8,16 --reps 5
  ```
  Baris mentah masuk `bench/results/l0_profile.csv`; tabel Markdown ke `docs/PROFIL_L0_tabel.md`.

## 3. Hasil
Waktu L0 (kmeans_cpu, 1 thread), median dari 5 pengulangan, 20 iterasi, ms (persen dari total):

| Citra | Ukuran | K | Baca | Init | K-Means | Pewarnaan | Tulis | Total | Bottleneck |
|---|---|---|---|---|---|---|---|---|---|
| sintetis_lanskap_1000x1000 | 1000x1000 | 8 | 19.0 (5.1%) | 0.0 (0.0%) | 313.9 (83.7%) | 0.9 (0.2%) | 42.8 (11.4%) | 375.1 | K-Means |
| sintetis_lanskap_2000x2000 | 2000x2000 | 8 | 67.0 (5.1%) | 0.0 (0.0%) | 1038.5 (78.5%) | 3.3 (0.2%) | 214.0 (16.2%) | 1322.8 | K-Means |
| sintetis_lanskap_4000x3000 | 4000x3000 | 8 | 201.0 (5.3%) | 0.0 (0.0%) | 3084.7 (80.6%) | 10.1 (0.3%) | 521.2 (13.6%) | 3827.2 | K-Means |
| sintetis_lanskap_7680x4320 | 7680x4320 | 8 | 569.4 (5.5%) | 0.0 (0.0%) | 8483.4 (82.7%) | 29.1 (0.3%) | 1198.9 (11.7%) | 10263.9 | K-Means |
| sintetis_lanskap_4000x3000 | 4000x3000 | 4 | 206.4 (10.3%) | 0.0 (0.0%) | 1514.7 (75.6%) | 9.9 (0.5%) | 267.4 (13.4%) | 2002.5 | K-Means |
| sintetis_lanskap_4000x3000 | 4000x3000 | 16 | 208.6 (2.6%) | 0.0 (0.0%) | 6556.9 (83.3%) | 10.8 (0.1%) | 1073.6 (13.6%) | 7871.4 | K-Means |

Pembacaan yang perlu dijawab dari tabel:
1. **Tahap apa yang terbesar pada setiap ukuran citra, dan berapa persen dari total?**
   Tahap terbesar pada seluruh konfigurasi dan ukuran citra adalah **K-Means**, yang berkontribusi sebesar **75.6% hingga 83.7%** dari total waktu eksekusi pipeline.
2. **Bagaimana K-Means tumbuh terhadap jumlah piksel (linear?) dan terhadap K?**
   - **Terhadap jumlah piksel:** K-Means tumbuh secara linier $O(N)$. Pada $K=8$: dari 1 MP (313.9 ms) ke 4 MP (1038.5 ms, ~3.3x), ke 12 MP (3084.7 ms, ~9.8x), dan ke 33.2 MP (8483.4 ms, ~27.0x).
   - **Terhadap K:** K-Means tumbuh secara linier $O(K)$. Pada ukuran 4000x3000: $K=4$ (1514.7 ms) ke $K=8$ (3084.7 ms, ~2.04x) ke $K=16$ (6556.9 ms, ~2.12x dari K=8 atau ~4.33x dari K=4).
3. **Apakah baca/tulis PNG tumbuh sebanding dengan piksel, atau terpengaruh isi citra (kompresi)?**
   - Tahap **baca** tumbuh sebanding dengan jumlah piksel (~19 ms pada 1 MP hingga ~569 ms pada 33.2 MP, konsisten ~5% dari total). Nilai baca tidak terpengaruh oleh $K$ (~201-208 ms konstan untuk citra 4000x3000).
   - Tahap **tulis** PNG terpengaruh oleh kompresi Deflate/PNG yang bergantung pada hasil label clustering: saat $K$ meningkat dari 4 ke 16 pada resolusi yang sama, waktu tulis melonjak dari 267.4 ms menjadi 1073.6 ms (~4x) karena kompleksitas pola visual dan entropi kompresi meningkat seiring bertambahnya jumlah klaster.
4. **Berapa varians antar pengulangan (min-maks)? Jika besar, tambah `--reps`.**
   Varians antar pengulangan tergolong sangat kecil dan stabil:
   - 1000x1000 ($K=8$): 369.3 - 435.2 ms (rentang ~65.9 ms)
   - 2000x2000 ($K=8$): 1302.7 - 1400.3 ms (rentang ~97.6 ms, ~7.5%)
   - 4000x3000 ($K=8$): 3718.2 - 3912.7 ms (rentang ~194.4 ms, ~5.2%)
   - 7680x4320 ($K=8$): 10156.1 - 10591.3 ms (rentang ~435.2 ms, ~4.3%)
   - 4000x3000 ($K=4$): 1914.4 - 2005.5 ms (rentang ~91.1 ms, ~4.8%)
   - 4000x3000 ($K=16$): 7634.2 - 8102.1 ms (rentang ~467.9 ms, ~6.1%)
   Karena varians relatif rendah (rata-rata 4-7%), 5 pengulangan sudah mencukupi untuk memperoleh median yang representatif.

## 4. Analisis bottleneck
Tulis kesimpulan dengan angka dari tabelmu:
- **Bottleneck utama:** **K-Means**, menyumbang **75.6% - 83.7%** (rata-rata ~80.6% pada citra 4000x3000 $K=8$) dari total waktu pipeline.
- **Batas speedup (hukum Amdahl):** bila bagian K-Means = $p$ dari total dan hanya K-Means yang dipercepat, speedup total maksimum = $1 / (1 - p)$.
  - Untuk konfigurasi standar 4000x3000 ($K=8$), porsi K-Means $p = 0.806$ (80.6%).
  - Batas speedup teoritis maksimum:
    $$S_{\max} = \frac{1}{1 - 0.806} = \frac{1}{0.194} \approx 5.15\times$$
  - Bahkan jika K-Means dipercepat hingga mendekati 0 detik menggunakan akselerasi GPU masif, speedup keseluruhan aplikasi tidak akan melampaui ~5.15x jika tahapan I/O dan pewarnaan tetap serial.
- **Tahap berikutnya setelah K-Means dipercepat:**
  Urutan waktu tahap di luar K-Means:
  1. **Tulis PNG (`write_ms`):** menyumbang 11.4% - 16.2% dari total (521.2 ms pada 4000x3000 $K=8$).
  2. **Baca PNG (`read_ms`):** menyumbang 2.6% - 10.3% dari total (201.0 ms pada 4000x3000 $K=8$).
  3. **Pewarnaan (`color_ms`):** menyumbang 0.1% - 0.5% dari total (10.1 ms pada 4000x3000 $K=8$).
  4. **Inisialisasi (`init_ms`):** mendekati 0.0 ms (<0.01% dari total).
- **Implikasi penempatan tahap:**
  - **L1 CUDA dan L2 OpenMP menyasar K-Means:** K-Means adalah tahap komputasi murni yang sangat dominan ($p \approx 80\%$). Mempercepat K-Means adalah prioritas pertama yang paling berdampak langsung.
  - **OpenMP/paralelisasi untuk baca/tulis dan pewarnaan:** Setelah K-Means dipangkas mendekati nol, tahap I/O (terutama encode/decode PNG dengan stb) akan menjadi bottleneck baru (~19.4% sisa waktu). Paralelisasi atau optimasi I/O akan membuka potensi speedup lebih lanjut.
  - **L3 (CPU+GPU streams):** Menumpuk (overlap) transfer memori (H2D/D2H), baca/tulis disk, dan komputasi kernel sangat relevan karena waktu I/O (baca + tulis = ~722 ms pada 12 MP) signifikan dibandingkan durasi kernel GPU setelah dipercepat.

## 5. Keterbatasan
- Citra sintetis: waktu baca/tulis PNG bergantung isi citra (kompresi); jangan dipakai sebagai satu-satunya
  dasar angka baca/tulis. Gunakan citra Sentinel-2 asli jika tersedia.
- Single-run per mesin; cache disk dihangatkan lewat 1 run pemanasan.
- Hasil dari satu mesin; spesifikasi perangkat keras tertera di Bab 2.

## 6. Reproduksibilitas
- Commit repo: `8786440`
- Perintah persis yang dijalankan:
  ```bash
  python3 bench/profile_l0.py --sizes 1000x1000,2000x2000,4000x3000,7680x4320 --k 8 --reps 5
  python3 bench/profile_l0.py --sizes 4000x3000 --k 4,8,16 --reps 5
  ```
