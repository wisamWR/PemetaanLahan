Benchmark landcover: median dari 5 pengulangan (setelah 1 pemanasan), 20 iterasi, seed 42, ms.
Speedup = total L0 / total konfigurasi (citra dan K sama). Efisiensi strong = speedup omp xT terhadap omp x1, dibagi T; efisiensi weak = waktu omp x1 / waktu omp xT pada beban per thread tetap. "belum diukur" = tidak terukur. Pemeriksaan identik: centroid dan counts sama dengan acuan.
Warmup = inisialisasi konteks CUDA: TIDAK masuk Total, tetapi dibayar di setiap proses landcover (aplikasi memanggil landcover sebagai proses baru tiap run). Proses = waktu dinding seluruh proses.

| Konfigurasi | Citra | Ukuran | K | Baca | Init | K-Means | Warna | Tulis | Total | Warmup | Proses | Speedup | Efisiensi | Identik |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L2 omp x8 | sintetis_lanskap_320x320 | 320x320 | 4 | 2.0 | 0.0 | 3.4 | 0.1 | 1.4 | 7.0 | 0.0 | 40.6 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_320x320 | 320x320 | 4 | 1.8 | 0.0 | 1.2 | 0.1 | 0.6 | 4.6 | 223.1 | 291.3 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_320x320 | 320x320 | 4 | 2.4 | 0.0 | 1.6 | 0.0 | 1.1 | 6.0 | 212.2 | 292.7 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_320x320 | 320x320 | 8 | 2.0 | 0.0 | 5.1 | 0.1 | 2.9 | 10.2 | 0.0 | 39.4 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_320x320 | 320x320 | 8 | 2.0 | 0.0 | 1.1 | 0.1 | 2.4 | 5.9 | 195.7 | 251.2 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_320x320 | 320x320 | 8 | 1.9 | 0.0 | 1.0 | 0.0 | 3.1 | 6.9 | 198.9 | 278.9 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_320x320 | 320x320 | 16 | 1.9 | 0.0 | 9.1 | 0.0 | 3.2 | 14.3 | 0.0 | 36.0 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_320x320 | 320x320 | 16 | 1.8 | 0.0 | 1.0 | 0.1 | 2.6 | 5.8 | 181.5 | 230.6 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_320x320 | 320x320 | 16 | 2.0 | 0.0 | 1.1 | 0.0 | 3.5 | 7.6 | 206.5 | 280.6 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_640x640 | 640x640 | 4 | 6.2 | 0.0 | 9.6 | 0.2 | 2.7 | 20.2 | 0.0 | 43.9 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_640x640 | 640x640 | 4 | 6.0 | 0.0 | 2.4 | 0.2 | 1.6 | 11.2 | 185.0 | 268.3 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_640x640 | 640x640 | 4 | 7.4 | 0.0 | 2.5 | 0.2 | 6.7 | 16.5 | 178.9 | 270.1 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_640x640 | 640x640 | 8 | 7.0 | 0.0 | 17.9 | 0.2 | 10.5 | 34.9 | 0.0 | 71.1 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_640x640 | 640x640 | 8 | 6.0 | 0.0 | 2.5 | 0.3 | 9.4 | 19.3 | 195.7 | 278.8 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_640x640 | 640x640 | 8 | 6.8 | 0.0 | 2.6 | 0.2 | 12.1 | 22.0 | 201.1 | 337.4 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_640x640 | 640x640 | 16 | 6.2 | 0.0 | 33.7 | 0.1 | 15.7 | 56.6 | 0.0 | 100.4 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_640x640 | 640x640 | 16 | 6.4 | 0.0 | 2.4 | 0.3 | 12.8 | 23.0 | 200.1 | 296.1 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_640x640 | 640x640 | 16 | 6.1 | 0.0 | 2.7 | 0.1 | 15.4 | 26.7 | 213.2 | 318.0 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_1000x1000 | 1000x1000 | 4 | 14.6 | 0.0 | 21.7 | 0.4 | 3.8 | 40.9 | 0.0 | 68.1 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_1000x1000 | 1000x1000 | 4 | 14.3 | 0.0 | 5.1 | 0.7 | 4.0 | 26.1 | 176.4 | 265.1 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_1000x1000 | 1000x1000 | 4 | 14.6 | 0.0 | 5.0 | 0.5 | 5.6 | 27.7 | 216.3 | 310.3 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_1000x1000 | 1000x1000 | 8 | 13.3 | 0.0 | 42.4 | 0.4 | 14.3 | 70.7 | 0.0 | 112.1 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_1000x1000 | 1000x1000 | 8 | 13.7 | 0.0 | 5.0 | 0.8 | 16.5 | 37.6 | 178.3 | 303.6 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_1000x1000 | 1000x1000 | 8 | 14.3 | 0.0 | 5.1 | 0.4 | 22.1 | 45.1 | 218.7 | 399.7 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_1000x1000 | 1000x1000 | 16 | 13.8 | 0.0 | 84.0 | 0.4 | 32.7 | 131.0 | 0.0 | 203.3 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_1000x1000 | 1000x1000 | 16 | 13.1 | 0.0 | 4.3 | 0.7 | 38.7 | 58.0 | 153.3 | 319.3 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_1000x1000 | 1000x1000 | 16 | 13.5 | 0.0 | 4.8 | 0.4 | 30.6 | 50.8 | 182.9 | 368.3 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_2000x2000 | 2000x2000 | 4 | 60.2 | 0.0 | 98.0 | 2.3 | 8.1 | 168.0 | 0.0 | 207.0 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_2000x2000 | 2000x2000 | 4 | 53.5 | 0.0 | 18.2 | 2.9 | 15.5 | 98.1 | 195.6 | 351.9 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_2000x2000 | 2000x2000 | 4 | 54.3 | 0.0 | 18.9 | 2.0 | 8.8 | 87.4 | 178.6 | 370.9 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_2000x2000 | 2000x2000 | 8 | 54.9 | 0.0 | 159.0 | 2.0 | 25.9 | 242.6 | 0.0 | 298.7 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_2000x2000 | 2000x2000 | 8 | 52.5 | 0.0 | 18.0 | 3.0 | 82.1 | 160.8 | 230.2 | 472.9 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_2000x2000 | 2000x2000 | 8 | 53.5 | 0.0 | 17.8 | 2.2 | 29.4 | 105.8 | 174.6 | 398.7 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_2000x2000 | 2000x2000 | 16 | 52.7 | 0.0 | 319.2 | 1.8 | 46.8 | 421.3 | 0.0 | 496.0 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_2000x2000 | 2000x2000 | 16 | 59.9 | 0.0 | 20.5 | 3.7 | 157.4 | 244.1 | 245.5 | 618.2 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_2000x2000 | 2000x2000 | 16 | 55.0 | 0.0 | 17.2 | 2.1 | 44.8 | 124.2 | 283.2 | 573.8 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_2500x2500 | 2500x2500 | 4 | 83.6 | 0.0 | 130.6 | 3.2 | 10.2 | 231.0 | 0.0 | 267.1 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_2500x2500 | 2500x2500 | 4 | 88.3 | 0.0 | 30.3 | 4.9 | 25.0 | 152.4 | 340.9 | 670.1 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_2500x2500 | 2500x2500 | 4 | 90.8 | 0.0 | 30.9 | 3.1 | 9.3 | 138.3 | 331.6 | 554.8 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_2500x2500 | 2500x2500 | 8 | 81.0 | 0.0 | 253.2 | 3.0 | 30.6 | 371.6 | 0.0 | 421.3 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_2500x2500 | 2500x2500 | 8 | 84.7 | 0.0 | 27.1 | 4.4 | 104.2 | 224.1 | 191.6 | 513.8 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_2500x2500 | 2500x2500 | 8 | 83.0 | 0.0 | 27.7 | 3.0 | 32.6 | 153.9 | 186.0 | 444.6 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_2500x2500 | 2500x2500 | 16 | 84.0 | 0.0 | 467.4 | 3.0 | 52.0 | 627.4 | 0.0 | 696.7 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_2500x2500 | 2500x2500 | 16 | 82.6 | 0.0 | 26.9 | 4.5 | 170.3 | 287.9 | 180.8 | 568.9 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_2500x2500 | 2500x2500 | 16 | 82.7 | 0.0 | 27.9 | 3.5 | 56.0 | 171.2 | 185.3 | 506.3 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_3000x3000 | 3000x3000 | 4 | 124.1 | 0.0 | 191.1 | 4.4 | 12.3 | 333.8 | 0.0 | 370.1 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_3000x3000 | 3000x3000 | 4 | 117.7 | 0.0 | 40.3 | 6.6 | 35.3 | 202.4 | 172.0 | 447.9 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_3000x3000 | 3000x3000 | 4 | 118.3 | 0.0 | 40.7 | 4.5 | 20.3 | 187.0 | 186.8 | 478.4 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_3000x3000 | 3000x3000 | 8 | 118.2 | 0.0 | 370.0 | 4.3 | 45.8 | 538.3 | 0.0 | 601.7 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_3000x3000 | 3000x3000 | 8 | 118.6 | 0.0 | 37.2 | 7.6 | 192.8 | 357.0 | 184.7 | 669.4 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_3000x3000 | 3000x3000 | 8 | 119.2 | 0.0 | 40.0 | 4.7 | 51.9 | 218.1 | 177.8 | 517.6 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_3000x3000 | 3000x3000 | 16 | 117.2 | 0.0 | 690.6 | 4.4 | 86.9 | 896.6 | 0.0 | 983.5 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_3000x3000 | 3000x3000 | 16 | 120.1 | 0.0 | 36.8 | 6.4 | 336.0 | 503.7 | 183.0 | 849.2 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_3000x3000 | 3000x3000 | 16 | 119.2 | 0.0 | 37.0 | 5.0 | 83.8 | 247.7 | 179.6 | 563.0 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_4000x3000 | 4000x3000 | 4 | 165.7 | 0.0 | 253.6 | 5.8 | 14.9 | 432.5 | 0.0 | 469.9 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_4000x3000 | 4000x3000 | 4 | 162.1 | 0.0 | 53.4 | 9.6 | 50.2 | 278.4 | 190.8 | 554.0 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_4000x3000 | 4000x3000 | 4 | 157.1 | 0.0 | 55.8 | 6.8 | 19.7 | 248.1 | 304.7 | 683.5 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_4000x3000 | 4000x3000 | 8 | 171.8 | 0.0 | 477.8 | 7.2 | 55.5 | 703.5 | 0.0 | 765.0 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_4000x3000 | 4000x3000 | 8 | 173.0 | 0.0 | 60.4 | 9.4 | 208.1 | 459.2 | 241.2 | 810.6 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_4000x3000 | 4000x3000 | 8 | 163.1 | 0.0 | 52.1 | 5.7 | 54.5 | 276.1 | 211.6 | 599.7 | belum diukur | belum diukur | ya |
| L2 omp x8 | sintetis_lanskap_4000x3000 | 4000x3000 | 16 | 155.1 | 0.0 | 895.3 | 5.4 | 94.3 | 1151.2 | 0.0 | 1224.7 | belum diukur | belum diukur | ya |
| L1 cuda | sintetis_lanskap_4000x3000 | 4000x3000 | 16 | 161.6 | 0.0 | 50.6 | 9.2 | 375.2 | 601.9 | 187.4 | 951.3 | belum diukur | belum diukur | ya |
| L1+L2 cuda x8 | sintetis_lanskap_4000x3000 | 4000x3000 | 16 | 157.3 | 0.0 | 50.1 | 5.6 | 91.0 | 305.5 | 203.7 | 658.3 | belum diukur | belum diukur | ya |

### Pemenang omp vs cuda (konfigurasi terbaik tiap mesin)

Dasar penentuan ambang mode otomatis (`kAutoCudaThresholdPixels` di landcover.cpp). Dua kriteria: Total (tanpa warmup CUDA) dan Proses (waktu dinding, dengan warmup CUDA). Yang dirasakan pengguna adalah Proses.

| Citra | Piksel (juta) | K | omp: Total | cuda: Total | Menang (Total) | omp: Proses | cuda: Proses | Menang (Proses) |
|---|---|---|---|---|---|---|---|---|
| sintetis_lanskap_320x320 | 0.10 | 4 | 7.0 (L2 omp x8) | 4.6 (L1 cuda) | cuda | 40.6 (L2 omp x8) | 291.3 (L1 cuda) | omp |
| sintetis_lanskap_640x640 | 0.41 | 4 | 20.2 (L2 omp x8) | 11.2 (L1 cuda) | cuda | 43.9 (L2 omp x8) | 268.3 (L1 cuda) | omp |
| sintetis_lanskap_1000x1000 | 1.00 | 4 | 40.9 (L2 omp x8) | 26.1 (L1 cuda) | cuda | 68.1 (L2 omp x8) | 265.1 (L1 cuda) | omp |
| sintetis_lanskap_2000x2000 | 4.00 | 4 | 168.0 (L2 omp x8) | 87.4 (L1+L2 cuda x8) | cuda | 207.0 (L2 omp x8) | 351.9 (L1 cuda) | omp |
| sintetis_lanskap_2500x2500 | 6.25 | 4 | 231.0 (L2 omp x8) | 138.3 (L1+L2 cuda x8) | cuda | 267.1 (L2 omp x8) | 554.8 (L1+L2 cuda x8) | omp |
| sintetis_lanskap_3000x3000 | 9.00 | 4 | 333.8 (L2 omp x8) | 187.0 (L1+L2 cuda x8) | cuda | 370.1 (L2 omp x8) | 447.9 (L1 cuda) | omp |
| sintetis_lanskap_4000x3000 | 12.00 | 4 | 432.5 (L2 omp x8) | 248.1 (L1+L2 cuda x8) | cuda | 469.9 (L2 omp x8) | 554.0 (L1 cuda) | omp |
| sintetis_lanskap_320x320 | 0.10 | 8 | 10.2 (L2 omp x8) | 5.9 (L1 cuda) | cuda | 39.4 (L2 omp x8) | 251.2 (L1 cuda) | omp |
| sintetis_lanskap_640x640 | 0.41 | 8 | 34.9 (L2 omp x8) | 19.3 (L1 cuda) | cuda | 71.1 (L2 omp x8) | 278.8 (L1 cuda) | omp |
| sintetis_lanskap_1000x1000 | 1.00 | 8 | 70.7 (L2 omp x8) | 37.6 (L1 cuda) | cuda | 112.1 (L2 omp x8) | 303.6 (L1 cuda) | omp |
| sintetis_lanskap_2000x2000 | 4.00 | 8 | 242.6 (L2 omp x8) | 105.8 (L1+L2 cuda x8) | cuda | 298.7 (L2 omp x8) | 398.7 (L1+L2 cuda x8) | omp |
| sintetis_lanskap_2500x2500 | 6.25 | 8 | 371.6 (L2 omp x8) | 153.9 (L1+L2 cuda x8) | cuda | 421.3 (L2 omp x8) | 444.6 (L1+L2 cuda x8) | omp |
| sintetis_lanskap_3000x3000 | 9.00 | 8 | 538.3 (L2 omp x8) | 218.1 (L1+L2 cuda x8) | cuda | 601.7 (L2 omp x8) | 517.6 (L1+L2 cuda x8) | cuda |
| sintetis_lanskap_4000x3000 | 12.00 | 8 | 703.5 (L2 omp x8) | 276.1 (L1+L2 cuda x8) | cuda | 765.0 (L2 omp x8) | 599.7 (L1+L2 cuda x8) | cuda |
| sintetis_lanskap_320x320 | 0.10 | 16 | 14.3 (L2 omp x8) | 5.8 (L1 cuda) | cuda | 36.0 (L2 omp x8) | 230.6 (L1 cuda) | omp |
| sintetis_lanskap_640x640 | 0.41 | 16 | 56.6 (L2 omp x8) | 23.0 (L1 cuda) | cuda | 100.4 (L2 omp x8) | 296.1 (L1 cuda) | omp |
| sintetis_lanskap_1000x1000 | 1.00 | 16 | 131.0 (L2 omp x8) | 50.8 (L1+L2 cuda x8) | cuda | 203.3 (L2 omp x8) | 319.3 (L1 cuda) | omp |
| sintetis_lanskap_2000x2000 | 4.00 | 16 | 421.3 (L2 omp x8) | 124.2 (L1+L2 cuda x8) | cuda | 496.0 (L2 omp x8) | 573.8 (L1+L2 cuda x8) | omp |
| sintetis_lanskap_2500x2500 | 6.25 | 16 | 627.4 (L2 omp x8) | 171.2 (L1+L2 cuda x8) | cuda | 696.7 (L2 omp x8) | 506.3 (L1+L2 cuda x8) | cuda |
| sintetis_lanskap_3000x3000 | 9.00 | 16 | 896.6 (L2 omp x8) | 247.7 (L1+L2 cuda x8) | cuda | 983.5 (L2 omp x8) | 563.0 (L1+L2 cuda x8) | cuda |
| sintetis_lanskap_4000x3000 | 12.00 | 16 | 1151.2 (L2 omp x8) | 305.5 (L1+L2 cuda x8) | cuda | 1224.7 (L2 omp x8) | 658.3 (L1+L2 cuda x8) | cuda |
