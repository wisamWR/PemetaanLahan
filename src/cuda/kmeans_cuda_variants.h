// kmeans_cuda_variants.h - varian internal modul CUDA (bukan bagian kontrak bersama).
//
//   kmeans_cuda        (kmeans.h) : versi utama, T4, akumulasi di shared memory per blok
//   kmeans_cuda_simple (di sini)  : versi T3, atomicAdd langsung ke memori global.
//                                   Dipertahankan sebagai pembanding dan cadangan.
// Keduanya punya argumen dan semantik yang sama persis dengan kmeans_cuda di kontrak.
#ifndef KMEANS_CUDA_VARIANTS_H
#define KMEANS_CUDA_VARIANTS_H

#include "kmeans.h"

#ifdef __cplusplus
extern "C" {
#endif

int kmeans_cuda_simple(const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
                       const KMeansParams* p, StageTimes* t);

// L3 (T5), mode "cuda_streams": kernel sama dengan kmeans_cuda, tetapi citra dikirim dan label
// diambil per potongan lewat memori pinned dan dua stream, sehingga transfer tumpang-tindih dengan
// assign iterasi pertama dan terakhir. Hasil identik dengan kmeans_cuda.
// Arti StageTimes berbeda karena tahapnya tumpang-tindih:
//   h2d_ms    = iterasi pertama (kirim citra per potongan + assign + update)
//   kernel_ms = iterasi tengah (2 .. max_iter-1)
//   d2h_ms    = iterasi terakhir (assign + ambil label per potongan + update), 0 bila max_iter == 1
int kmeans_cuda_streams(const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
                        const KMeansParams* p, StageTimes* t);

// Sama seperti kmeans_cuda_streams dengan ukuran potongan (piksel) ditentukan sendiri.
// Untuk uji dan penyetelan; chunk_pixels = 0 memakai nilai bawaan.
int kmeans_cuda_streams_chunked(const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
                                const KMeansParams* p, StageTimes* t, size_t chunk_pixels);

#ifdef __cplusplus
}
#endif

#endif  // KMEANS_CUDA_VARIANTS_H
