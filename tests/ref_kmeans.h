// ref_kmeans.h - referensi CPU SEMENTARA milik Wisam untuk menguji mode CUDA.
// Bukan pengganti kmeans_cpu (src/cpu, milik Azka). Setelah modul Azka masuk, berkas ini
// dibuang atau dipakai sebagai pembanding cadangan.
#ifndef REF_KMEANS_H
#define REF_KMEANS_H

#include "kmeans.h"

#ifdef __cplusplus
extern "C" {
#endif

// Semantik sama persis dengan kmeans_cpu di kontrak. t boleh NULL.
int ref_kmeans(const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
               const KMeansParams* p, StageTimes* t);

#ifdef __cplusplus
}
#endif

#endif  // REF_KMEANS_H
