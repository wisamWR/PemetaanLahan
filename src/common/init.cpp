// init.cpp - inisialisasi centroid bersama untuk semua mode (kontrak butir 3.1).
//
// Algoritma (harus identik di semua platform dan kompilator):
//
// 1. PRNG: splitmix64 yang ditulis sendiri (bukan rand()).
//      state awal = (uint64_t)seed
//      next():  state += 0x9E3779B97F4A7C15
//               z = state
//               z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
//               z = (z ^ (z >> 27)) * 0x94D049BB133111EB
//               return z ^ (z >> 31)
//    Semua operasi pada uint64_t tak bertanda (overflow terdefinisi, modulo 2^64).
//
// 2. Pemilihan indeks: untuk c = 0, 1, ..., k-1 berurutan:
//      idx = next() % n
//      Bila idx sudah pernah terpilih untuk cluster sebelumnya DAN banyaknya indeks
//      berbeda yang sudah terpilih masih < n, buang idx lalu ambil ulang (next() lagi).
//      Bila semua n piksel sudah terpilih (hanya terjadi bila k > n, mis. citra 1x1 dengan
//      K=16), idx diterima apa adanya walaupun berulang.
//    Bias modulo (2^64 tidak habis dibagi n) diabaikan; tetap deterministik.
//
// 3. Centroid awal: out[c*3 + ch] = rgb[idx*3 + ch] untuk ch = 0 (R), 1 (G), 2 (B).
//
// Catatan: indeks berbeda belum tentu warnanya berbeda. Bila dua centroid awal berwarna
// sama, tie-break "indeks terkecil" membuat cluster berindeks lebih besar selalu kosong
// dan centroidnya tidak berubah (kontrak butir 3.3 dan 3.5). Ini tetap deterministik.

#include "kmeans.h"

namespace {

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t s) : state(s) {}
    uint64_t next() {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
};

}  // namespace

extern "C" int kmeans_init_centroids(const uint8_t* rgb, size_t n, int k,
                                     uint32_t seed, uint8_t* out) {
    if (rgb == nullptr || out == nullptr) return 1;
    if (n == 0) return 1;
    if (k < 1 || k > KMEANS_MAX_K) return 1;

    SplitMix64 rng(static_cast<uint64_t>(seed));
    const uint64_t n64 = static_cast<uint64_t>(n);

    uint64_t chosen[KMEANS_MAX_K];
    uint64_t distinct = 0;  // banyaknya indeks berbeda yang sudah terpilih

    for (int c = 0; c < k; ++c) {
        uint64_t idx;
        for (;;) {
            idx = rng.next() % n64;
            bool seen = false;
            for (int j = 0; j < c; ++j) {
                if (chosen[j] == idx) { seen = true; break; }
            }
            if (!seen) { ++distinct; break; }
            if (distinct >= n64) break;  // semua piksel sudah terpilih: duplikat diizinkan
        }
        chosen[c] = idx;
        const size_t off = static_cast<size_t>(idx) * 3;
        out[c * 3 + 0] = rgb[off + 0];
        out[c * 3 + 1] = rgb[off + 1];
        out[c * 3 + 2] = rgb[off + 2];
    }
    return 0;
}
