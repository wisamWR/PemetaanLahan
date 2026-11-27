Benchmark landcover: median dari 5 pengulangan (setelah 1 pemanasan), 20 iterasi, seed 42, ms.
Speedup = total L0 / total konfigurasi (citra dan K sama). Efisiensi strong = speedup omp xT terhadap omp x1, dibagi T; efisiensi weak = waktu omp x1 / waktu omp xT pada beban per thread tetap. "belum diukur" = tidak terukur. Pemeriksaan identik: centroid dan counts sama dengan acuan.

| Konfigurasi | Citra | Ukuran | K | Baca | Init | K-Means | Warna | Tulis | Total | Speedup | Efisiensi | Identik |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | sintetis_lanskap_1000x1000 | 1000x1000 | 8 | 18.6 | 0.0 | 367.7 | 0.9 | 23.5 | 414.0 | 1.00x | belum diukur | ya |
| L2 omp x1 | sintetis_lanskap_1000x1000 | 1000x1000 | 8 | 21.1 | 0.0 | 339.8 | 1.0 | 24.7 | 386.3 | 1.07x | 1.00 | ya |
| L2 omp x2 | sintetis_lanskap_1000x1000 | 1000x1000 | 8 | 17.0 | 0.0 | 196.4 | 0.4 | 16.8 | 230.0 | 1.80x | 0.84 | ya |
| L2 omp x4 | sintetis_lanskap_1000x1000 | 1000x1000 | 8 | 17.2 | 0.0 | 131.0 | 0.4 | 18.2 | 167.5 | 2.47x | 0.58 | ya |
| L2 omp x8 | sintetis_lanskap_1000x1000 | 1000x1000 | 8 | 19.1 | 0.0 | 183.8 | 2.9 | 27.1 | 231.1 | 1.79x | 0.21 | ya |
| L0 | sintetis_lanskap_2000x2000 | 2000x2000 | 8 | 69.5 | 0.0 | 1121.4 | 3.6 | 112.9 | 1311.5 | 1.00x | belum diukur | ya |
| L2 omp x1 | sintetis_lanskap_2000x2000 | 2000x2000 | 8 | 73.4 | 0.0 | 1107.8 | 3.4 | 114.4 | 1298.5 | 1.01x | 1.00 | ya |
| L2 omp x2 | sintetis_lanskap_2000x2000 | 2000x2000 | 8 | 70.9 | 0.0 | 726.1 | 2.4 | 68.2 | 866.6 | 1.51x | 0.75 | ya |
| L2 omp x4 | sintetis_lanskap_2000x2000 | 2000x2000 | 8 | 85.5 | 0.0 | 505.3 | 1.8 | 58.7 | 646.0 | 2.03x | 0.50 | ya |
| L2 omp x8 | sintetis_lanskap_2000x2000 | 2000x2000 | 8 | 73.8 | 0.0 | 485.8 | 5.6 | 43.8 | 627.1 | 2.09x | 0.26 | ya |
| L0 | sintetis_lanskap_4000x3000 | 4000x3000 | 8 | 207.1 | 0.0 | 3314.0 | 10.4 | 281.5 | 3830.1 | 1.00x | belum diukur | ya |
| L2 omp x1 | sintetis_lanskap_4000x3000 | 4000x3000 | 8 | 217.6 | 0.0 | 3332.7 | 10.6 | 291.4 | 3861.5 | 0.99x | 1.00 | ya |
| L2 omp x2 | sintetis_lanskap_4000x3000 | 4000x3000 | 8 | 223.3 | 0.0 | 2068.0 | 5.6 | 190.9 | 2483.8 | 1.54x | 0.78 | ya |
| L2 omp x4 | sintetis_lanskap_4000x3000 | 4000x3000 | 8 | 211.3 | 0.0 | 1434.4 | 5.2 | 126.6 | 1761.5 | 2.17x | 0.55 | ya |
| L2 omp x8 | sintetis_lanskap_4000x3000 | 4000x3000 | 8 | 233.3 | 0.0 | 1220.4 | 13.6 | 120.1 | 1589.7 | 2.41x | 0.30 | ya |
| L0 | sintetis_lanskap_7680x4320 | 7680x4320 | 8 | 567.4 | 0.0 | 8351.6 | 27.4 | 541.0 | 9499.0 | 1.00x | belum diukur | ya |
| L2 omp x1 | sintetis_lanskap_7680x4320 | 7680x4320 | 8 | 512.7 | 0.0 | 8361.4 | 26.2 | 534.3 | 9469.6 | 1.00x | 1.00 | ya |
| L2 omp x2 | sintetis_lanskap_7680x4320 | 7680x4320 | 8 | 521.5 | 0.0 | 4708.6 | 14.2 | 311.7 | 5597.0 | 1.70x | 0.85 | ya |
| L2 omp x4 | sintetis_lanskap_7680x4320 | 7680x4320 | 8 | 529.4 | 0.0 | 3262.2 | 13.3 | 203.5 | 3995.4 | 2.38x | 0.59 | ya |
| L2 omp x8 | sintetis_lanskap_7680x4320 | 7680x4320 | 8 | 523.4 | 0.0 | 2465.5 | 10.7 | 153.0 | 3162.3 | 3.00x | 0.37 | ya |

### Dilewati (belum diukur)

- mesin cuda: tidak terhubung pada build ini (landcover --info)
