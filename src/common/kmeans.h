/*
 * kmeans.h - antarmuka bersama semua mode K-Means (cpu, omp, cuda).
 *
 * Sumber aturan: docs/KONTRAK_TEKNIS.md. Jangan ubah tipe atau deklarasi di sini
 * tanpa persetujuan tim (tulis usulan di bagian "Usulan perubahan" kontrak).
 *
 * Header ini harus tetap kompatibel dengan C (tanpa fitur khusus C++).
 */
#ifndef KMEANS_H
#define KMEANS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Batas atas K menurut kontrak (k = 1..16). */
#define KMEANS_MAX_K 16

typedef struct {
    int      width, height;   /* piksel */
    int      k;               /* jumlah cluster, 1..16 */
    int      max_iter;        /* jumlah iterasi tetap, default 20 */
    uint32_t seed;            /* seed inisialisasi, default 42 */
    int      num_threads;     /* hanya dipakai mode OpenMP */
} KMeansParams;

typedef struct {
    double read_ms, init_ms, kmeans_ms, color_ms, write_ms, total_ms;
    double h2d_ms, kernel_ms, d2h_ms;   /* hanya mode CUDA, selain itu 0 */
} StageTimes;

/*
 * Inisialisasi centroid bersama (kontrak butir 3.1). Dipakai SEMUA mode.
 *   rgb : citra RGB interleaved, 3*n byte
 *   n   : jumlah piksel (width*height)
 *   k   : jumlah cluster, 1..KMEANS_MAX_K
 *   seed: seed PRNG
 *   out : k*3 byte centroid (R,G,B per cluster), dialokasikan pemanggil
 * Kembalian 0 = sukses, selain 0 = galat (argumen tidak sah).
 * Algoritma lengkap dijelaskan di src/common/init.cpp.
 */
int kmeans_init_centroids(const uint8_t* rgb, size_t n, int k, uint32_t seed,
                          uint8_t* out);

/* Kembalian 0 = sukses, selain 0 = galat. labels dan centroids dialokasikan pemanggil. */
int kmeans_cpu (const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
                const KMeansParams* p, StageTimes* t);
int kmeans_omp (const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
                const KMeansParams* p, StageTimes* t);
int kmeans_cuda(const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
                const KMeansParams* p, StageTimes* t);

#ifdef __cplusplus
}
#endif

#endif /* KMEANS_H */
