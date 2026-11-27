# Benchmark di mesin GPU (pelengkap #10)

Dijalankan dengan `bench/run_bench.py` pada 27 Nov 2026, untuk mengisi angka CUDA yang di hasil awal #10 masih
"belum diukur" (build tanpa CUDA).

- Mesin: Intel Core i7-13620H (WSL2 Ubuntu 24.04, 16 thread), NVIDIA RTX 3050 6GB Laptop (sm_86, driver 576.93),
  CUDA 12.9, GCC 13.3. Ini bukan mesin di proposal (i7-13650HX), jadi angka di sini tidak boleh dicampur
  dengan angka mesin lain tanpa keterangan.
- Data dan folder run di filesystem Linux WSL (bukan `/mnt/c`).
- Median dari 5 pengulangan setelah 1 pemanasan, K=8, 20 iterasi, seed 42, citra sintetis lanskap.
- Semua konfigurasi `landcover` identik dengan L0 (centroid dan counts).

Perintah:

```
python3 bench/run_bench.py --sizes 1000x1000,2000x2000,4000x3000,7680x4320 --k 8 \
    --cuda-variants simple,shared,streams --tag bench_gpu
python3 bench/run_bench.py --sizes 200x200,320x320,450x450,640x640,800x800 --k 8 --threads 4,8 --tag sweep_gpu
```

Berkas: `results/bench_gpu_raw.csv`, `results/bench_gpu_summary.csv`, `results/sweep_gpu_raw.csv`,
`results/sweep_gpu_summary.csv`.

## Total waktu (ms) dan speedup terhadap L0

| Ukuran | L0 | OMP terbaik | L1 CUDA | L1+L2 CUDA terbaik |
|---|---|---|---|---|
| 1000x1000 | 237 | 74 (x8, 3,2x) | 37 (6,4x) | 35 (x2, 6,9x) |
| 2000x2000 | 922 | 245 (x8, 3,8x) | 160 (5,8x) | 113 (x4, 8,2x) |
| 4000x3000 | 2.603 | 690 (x8, 3,8x) | 420 (6,2x) | 281 (x8, 9,3x) |
| 7680x4320 | 6.766 | 1.876 (x8, 3,6x) | 1.033 (6,6x) | 684 (x16, 9,9x) |

Tahap K-Means saja pada 7680x4320: 5.871 ms (L0) menjadi 139 ms (CUDA).

## Varian CUDA (`bench_cuda`, tahap K-Means saja, citra acak, ms)

| Ukuran | T3 simple | T4 shared | T5 streams |
|---|---|---|---|
| 1000x1000 | 39,4 | 2,8 | 3,4 |
| 2000x2000 | 154,1 | 9,7 | 11,6 |
| 4000x3000 | 462,7 | 32,7 | 34,5 |
| 7680x4320 | 1.282,4 | 83,4 | 91,1 |

## Ambang mode otomatis (#17)

Total terbaik tiap mesin (ms) pada citra kecil:

| Ukuran | Piksel (juta) | OMP terbaik | CUDA terbaik |
|---|---|---|---|
| 200x200 | 0,04 | 5,0 | 4,1 |
| 320x320 | 0,10 | 11,2 | 5,7 |
| 450x450 | 0,20 | 17,3 | 9,0 |
| 640x640 | 0,41 | 35,6 | 19,0 |
| 800x800 | 0,64 | 58,1 | 26,5 |

Ambang `kAutoCudaThresholdPixels` di `src/cli/landcover.cpp` diganti dari 8.000.000 (sementara) menjadi
100.000 piksel: mulai titik ini CUDA unggul sekitar 2x. Pada 0,04 juta piksel selisihnya sekitar 1 ms, masih dalam
derau. Di bawah ambang belum diukur rinci, jadi tetap omp.

## Catatan

1. OMP x16 lebih lambat daripada x8 di semua ukuran. Kemungkinan 16 thread WSL adalah hyperthread di atas inti
   fisik yang lebih sedikit. Belum diselidiki lebih jauh.
2. Setelah K-Means di GPU, tahap baca PNG (serial, sekitar 440 ms pada 7680x4320) menjadi tahap terbesar,
   sekitar 64 % dari total L1+L2 terbaik.
3. Weak scaling (`--weak`) dan citra Sentinel asli (`--image`) belum diukur di mesin ini.
