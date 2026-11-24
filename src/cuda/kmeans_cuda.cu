// kmeans_cuda.cu - K-Means L1 di CUDA, dua varian dengan hasil identik:
//
//   kmeans_cuda_simple (T3): tiap thread atomicAdd langsung ke sums/counts global (64-bit).
//                            Dengan K cluster, jutaan thread menulis ke paling banyak 4K alamat
//                            yang sama, sehingga atomik global saling antre.
//   kmeans_cuda        (T4): tiap blok mengakumulasi di shared memory (32-bit), lalu SEKALI per
//                            blok menggabungkan ke akumulator global 64-bit.
//
// Alur bersama (mengikuti kontrak butir 3, identik dengan referensi CPU):
//   host : validasi argumen, kmeans_init_centroids (src/common), salin citra + centroid ke device
//   loop tepat max_iter kali, seluruhnya di device (centroid tidak bolak-balik ke host):
//     1. cudaMemset sums/counts = 0
//     2. assign + akumulasi: jarak kuadrat int32 ke K centroid, tie-break indeks terkecil,
//        tulis label int32, tambahkan piksel ke cluster terpilih
//     3. update_centroids: centroid = sum / count (pembagian bilangan bulat), cluster kosong tetap
//   host : salin label dan centroid ke host
// Penjumlahan bilangan bulat asosiatif, jadi urutan akumulasi tidak memengaruhi hasil.
//
// Waktu (cudaEvent): h2d_ms = salin citra + centroid, kernel_ms = seluruh loop iterasi,
// d2h_ms = salin label + centroid. kmeans_ms = h2d + loop + d2h. init_ms dan total_ms dari jam host
// (total_ms termasuk alokasi dan pembebasan memori device).

#include <chrono>
#include <climits>
#include <cstring>

#include "cuda_utils.h"
#include "kmeans.h"
#include "kmeans_cuda_variants.h"

namespace {

constexpr int kBlockSize = 256;

// Batas piksel per blok agar akumulator shared 32-bit tidak meluap:
// sum per kanal <= piksel x 255 <= 2^32 - 1  ->  piksel <= 16.843.009.
constexpr unsigned long long kMaxPixelsPerBlock = 0xFFFFFFFFull / 255ull;

enum class Variant { Simple, Shared };

// Jarak kuadrat int32 ke K centroid; '<' ketat memberi tie-break ke indeks terkecil.
__device__ __forceinline__ int nearest(int r, int g, int b, const int* s_cent, int k) {
    int best = 0;
    int best_d = INT_MAX;
    for (int c = 0; c < k; ++c) {
        const int dr = r - s_cent[c * 3 + 0];
        const int dg = g - s_cent[c * 3 + 1];
        const int db = b - s_cent[c * 3 + 2];
        const int d = dr * dr + dg * dg + db * db;
        if (d < best_d) { best_d = d; best = c; }
    }
    return best;
}

// ---- T3: atomicAdd global per piksel ---------------------------------------------------------
// Satu thread per piksel dengan grid-stride loop. Indeks size_t (n bisa 120 juta, offset RGB
// 3*i sampai sekitar 360 juta).
__global__ void assign_accumulate_global(const uint8_t* __restrict__ rgb, size_t n,
                                         const uint8_t* __restrict__ centroids, int k,
                                         int32_t* __restrict__ labels,
                                         unsigned long long* __restrict__ sums,
                                         unsigned long long* __restrict__ counts) {
    __shared__ int s_cent[KMEANS_MAX_K * 3];
    for (int i = threadIdx.x; i < k * 3; i += blockDim.x) s_cent[i] = centroids[i];
    __syncthreads();

    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const size_t off = i * 3;
        const int r = rgb[off + 0];
        const int g = rgb[off + 1];
        const int b = rgb[off + 2];
        const int best = nearest(r, g, b, s_cent, k);
        labels[i] = best;
        atomicAdd(&sums[best * 3 + 0], static_cast<unsigned long long>(r));
        atomicAdd(&sums[best * 3 + 1], static_cast<unsigned long long>(g));
        atomicAdd(&sums[best * 3 + 2], static_cast<unsigned long long>(b));
        atomicAdd(&counts[best], 1ULL);
    }
}

// ---- T4: akumulasi shared memory per blok ----------------------------------------------------
// Blok memproses banyak piksel lewat grid-stride loop. Pemanggil menjamin piksel per blok
// <= kMaxPixelsPerBlock sehingga akumulator 32-bit di shared memory aman.
__global__ void assign_accumulate_shared(const uint8_t* __restrict__ rgb, size_t n,
                                         const uint8_t* __restrict__ centroids, int k,
                                         int32_t* __restrict__ labels,
                                         unsigned long long* __restrict__ sums,
                                         unsigned long long* __restrict__ counts) {
    __shared__ int s_cent[KMEANS_MAX_K * 3];
    __shared__ unsigned int s_sum[KMEANS_MAX_K * 3];
    __shared__ unsigned int s_cnt[KMEANS_MAX_K];
    for (int i = threadIdx.x; i < KMEANS_MAX_K * 3; i += blockDim.x) {
        s_cent[i] = i < k * 3 ? centroids[i] : 0;
        s_sum[i] = 0;
    }
    for (int i = threadIdx.x; i < KMEANS_MAX_K; i += blockDim.x) s_cnt[i] = 0;
    __syncthreads();

    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const size_t off = i * 3;
        const int r = rgb[off + 0];
        const int g = rgb[off + 1];
        const int b = rgb[off + 2];
        const int best = nearest(r, g, b, s_cent, k);
        labels[i] = best;
        atomicAdd(&s_sum[best * 3 + 0], static_cast<unsigned int>(r));
        atomicAdd(&s_sum[best * 3 + 1], static_cast<unsigned int>(g));
        atomicAdd(&s_sum[best * 3 + 2], static_cast<unsigned int>(b));
        atomicAdd(&s_cnt[best], 1u);
    }
    __syncthreads();

    // Gabungkan ke global sekali per blok; lewati nol agar tidak menambah lalu lintas atomik.
    for (int i = threadIdx.x; i < k * 3; i += blockDim.x) {
        if (s_sum[i] != 0) atomicAdd(&sums[i], static_cast<unsigned long long>(s_sum[i]));
    }
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        if (s_cnt[i] != 0) atomicAdd(&counts[i], static_cast<unsigned long long>(s_cnt[i]));
    }
}

__global__ void update_centroids(uint8_t* centroids, int k, const unsigned long long* sums,
                                 const unsigned long long* counts) {
    const int c = threadIdx.x;
    if (c >= k) return;
    const unsigned long long cnt = counts[c];
    if (cnt == 0) return;  // cluster kosong: centroid tetap
    for (int ch = 0; ch < 3; ++ch) {
        centroids[c * 3 + ch] = static_cast<uint8_t>(sums[c * 3 + ch] / cnt);
    }
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Ukuran grid varian T3: satu thread per piksel (dibatasi batas gridDim.x; sisanya grid-stride).
unsigned int grid_simple(size_t n) {
    const size_t want = (n + kBlockSize - 1) / kBlockSize;
    return static_cast<unsigned int>(want < (1u << 30) ? want : (1u << 30));
}

// Ukuran grid varian T4: secukupnya untuk mengisi GPU (blok aktif per SM x jumlah SM), sehingga
// tiap blok memproses banyak piksel dan penggabungan global jarang terjadi. Bila piksel per blok
// melampaui batas 32-bit, jumlah blok dinaikkan.
int grid_shared(size_t n, unsigned int* blocks_out) {
    int device = 0, sms = 0, per_sm = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, assign_accumulate_shared,
                                                             kBlockSize, 0));
    size_t blocks = static_cast<size_t>(sms) * static_cast<size_t>(per_sm > 0 ? per_sm : 1);
    const size_t max_useful = (n + kBlockSize - 1) / kBlockSize;
    if (blocks > max_useful) blocks = max_useful;
    for (;;) {
        const size_t stride = blocks * kBlockSize;
        const size_t per_thread = (n + stride - 1) / stride;
        if (per_thread * kBlockSize <= kMaxPixelsPerBlock) break;
        blocks *= 2;
    }
    *blocks_out = static_cast<unsigned int>(blocks);
    return 0;
}

int run(Variant variant, const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
        const KMeansParams* p, StageTimes* t) {
    if (rgb == nullptr || labels == nullptr || centroids == nullptr || p == nullptr) return 1;
    if (p->width < 1 || p->height < 1) return 1;
    if (p->k < 1 || p->k > KMEANS_MAX_K) return 1;
    if (p->max_iter < 1) return 1;

    const auto t_start = std::chrono::steady_clock::now();
    const size_t n = static_cast<size_t>(p->width) * static_cast<size_t>(p->height);
    const int k = p->k;

    // Inisialisasi bersama di host (kontrak butir 3.1). Mode CUDA tidak punya inisialisasi sendiri.
    uint8_t init_cent[KMEANS_MAX_K * 3];
    if (kmeans_init_centroids(rgb, n, k, p->seed, init_cent) != 0) return 1;
    const double init_ms = ms_since(t_start);

    unsigned int blocks = 0;
    if (variant == Variant::Simple) {
        blocks = grid_simple(n);
    } else {
        const int rc = grid_shared(n, &blocks);
        if (rc != 0) return rc;
    }

    DeviceBuffer<uint8_t> d_rgb, d_cent;
    DeviceBuffer<int32_t> d_labels;
    DeviceBuffer<unsigned long long> d_sums, d_counts;
    CUDA_CHECK(d_rgb.alloc(n * 3));
    CUDA_CHECK(d_labels.alloc(n));
    CUDA_CHECK(d_cent.alloc(KMEANS_MAX_K * 3));
    CUDA_CHECK(d_sums.alloc(KMEANS_MAX_K * 3));
    CUDA_CHECK(d_counts.alloc(KMEANS_MAX_K));

    DeviceEvent e_h2d0, e_h2d1, e_loop1, e_d2h1;
    CUDA_CHECK(e_h2d0.create());
    CUDA_CHECK(e_h2d1.create());
    CUDA_CHECK(e_loop1.create());
    CUDA_CHECK(e_d2h1.create());

    CUDA_CHECK(cudaEventRecord(e_h2d0.ev));
    CUDA_CHECK(cudaMemcpy(d_rgb.ptr, rgb, n * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cent.ptr, init_cent, static_cast<size_t>(k) * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e_h2d1.ev));

    for (int it = 0; it < p->max_iter; ++it) {
        CUDA_CHECK(cudaMemsetAsync(d_sums.ptr, 0, KMEANS_MAX_K * 3 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemsetAsync(d_counts.ptr, 0, KMEANS_MAX_K * sizeof(unsigned long long)));
        if (variant == Variant::Simple) {
            assign_accumulate_global<<<blocks, kBlockSize>>>(d_rgb.ptr, n, d_cent.ptr, k,
                                                             d_labels.ptr, d_sums.ptr, d_counts.ptr);
        } else {
            assign_accumulate_shared<<<blocks, kBlockSize>>>(d_rgb.ptr, n, d_cent.ptr, k,
                                                             d_labels.ptr, d_sums.ptr, d_counts.ptr);
        }
        CUDA_CHECK_LAUNCH();
        update_centroids<<<1, 32>>>(d_cent.ptr, k, d_sums.ptr, d_counts.ptr);
        CUDA_CHECK_LAUNCH();
    }
    CUDA_CHECK(cudaEventRecord(e_loop1.ev));

    CUDA_CHECK(cudaMemcpy(labels, d_labels.ptr, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(centroids, d_cent.ptr, static_cast<size_t>(k) * 3, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e_d2h1.ev));
    CUDA_CHECK(cudaEventSynchronize(e_d2h1.ev));

    float h2d = 0.f, loop = 0.f, d2h = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d, e_h2d0.ev, e_h2d1.ev));
    CUDA_CHECK(cudaEventElapsedTime(&loop, e_h2d1.ev, e_loop1.ev));
    CUDA_CHECK(cudaEventElapsedTime(&d2h, e_loop1.ev, e_d2h1.ev));

    if (t != nullptr) {
        std::memset(t, 0, sizeof(*t));
        t->init_ms = init_ms;
        t->h2d_ms = h2d;
        t->kernel_ms = loop;
        t->d2h_ms = d2h;
        t->kmeans_ms = static_cast<double>(h2d) + loop + d2h;
    }
    // total_ms diisi setelah buffer device dibebaskan (lihat timed()).
    return 0;
}

int timed(Variant variant, const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
          const KMeansParams* p, StageTimes* t) {
    // Inisialisasi konteks CUDA di luar pengukuran (panggilan pertama bisa ratusan ms).
    CUDA_CHECK(cudaFree(nullptr));

    const auto t0 = std::chrono::steady_clock::now();
    const int rc = run(variant, rgb, labels, centroids, p, t);  // buffer dibebaskan saat run() selesai
    if (rc == 0 && t != nullptr) t->total_ms = ms_since(t0);
    return rc;
}

}  // namespace

extern "C" int kmeans_cuda(const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
                           const KMeansParams* p, StageTimes* t) {
    return timed(Variant::Shared, rgb, labels, centroids, p, t);
}

extern "C" int kmeans_cuda_simple(const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
                                  const KMeansParams* p, StageTimes* t) {
    return timed(Variant::Simple, rgb, labels, centroids, p, t);
}
