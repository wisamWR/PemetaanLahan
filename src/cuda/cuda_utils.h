// cuda_utils.h - pemeriksaan galat CUDA dan pembungkus memori device (RAII).
//
// Aturan proyek: setiap panggilan CUDA diperiksa. Pustaka tidak boleh memanggil exit();
// galat dicetak (berkas, baris, pesan) lalu dikembalikan sebagai kode bukan nol.
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>

// Cetak galat dan kembalikan kodenya (int, selalu != 0) dari fungsi pemanggil.
#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        const cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "[CUDA] %s:%d: %s gagal: %s (%s)\n", __FILE__,       \
                         __LINE__, #call, cudaGetErrorString(err_),                   \
                         cudaGetErrorName(err_));                                     \
            return static_cast<int>(err_);                                            \
        }                                                                             \
    } while (0)

// Periksa galat peluncuran kernel (konfigurasi tidak sah, dsb.).
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

// Untuk tempat yang tidak bisa mengembalikan nilai (destruktor): cetak saja.
#define CUDA_WARN(call)                                                               \
    do {                                                                              \
        const cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "[CUDA] %s:%d: %s gagal: %s\n", __FILE__, __LINE__,  \
                         #call, cudaGetErrorString(err_));                            \
        }                                                                             \
    } while (0)

// Buffer device yang otomatis dibebaskan saat keluar cakupan (juga pada jalur galat).
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() {
        if (ptr != nullptr) CUDA_WARN(cudaFree(ptr));
    }
    cudaError_t alloc(size_t count) { return cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)); }
};

// Event CUDA yang otomatis dihancurkan.
struct DeviceEvent {
    cudaEvent_t ev = nullptr;
    DeviceEvent() = default;
    DeviceEvent(const DeviceEvent&) = delete;
    DeviceEvent& operator=(const DeviceEvent&) = delete;
    ~DeviceEvent() {
        if (ev != nullptr) CUDA_WARN(cudaEventDestroy(ev));
    }
    cudaError_t create() { return cudaEventCreate(&ev); }
};

#endif  // CUDA_UTILS_H
