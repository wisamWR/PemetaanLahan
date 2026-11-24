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

#ifdef __cplusplus
}
#endif

#endif  // KMEANS_CUDA_VARIANTS_H
