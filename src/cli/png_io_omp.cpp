// png_io_omp.cpp - backend png_io.h dengan penulis PNG paralel OpenMP (L2, Gavra).
//
// Pengganti src/cli/png_io_stb.cpp tanpa mengubah landcover.cpp:
//   make landcover LC_PNG_SRC=src/cli/png_io_omp.cpp
//
// Baca : sama dengan png_io_stb.cpp (stb_image, serial; satu aliran deflate tidak bisa dipecah).
//        Untuk baca paralel, beri masukan berupa folder potongan + manifest (sudah didukung landcover).
// Tulis: encoder PNG sendiri, gaya pigz. Citra dibagi menjadi strip baris tetap (kPngStripRows),
//        tiap strip difilter (Sub), dimampatkan (LZ77 + Huffman tetap) dan diberi alamat adler32 oleh
//        satu thread; strip disambung berurutan dengan blok "stored" kosong sebagai penyelaras byte.
//        Tiap strip dimampatkan mandiri, jadi KELUARAN IDENTIK BYTE DEMI BYTE untuk berapa pun jumlah
//        thread (termasuk 1). Piksel hasil dekode sama persis dengan masukan (tanpa kehilangan).
//
// Catatan: berkas PNG tidak identik dengan keluaran stb (kompresor berbeda), tetapi pikselnya sama.
// Tidak ada pengecualian yang keluar dari fungsi publik (aman dipanggil dari daerah paralel).

#include "png_io.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef LC_PNG_FILTER
#define LC_PNG_FILTER 1  // 0 none, 1 sub (sama dengan png_io_stb.cpp)
#endif
#ifndef LC_PNG_STRIP_BYTES
#define LC_PNG_STRIP_BYTES (1 << 21)  // ~2 MiB data tersaring per strip
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace lc {

namespace {

// ---------------------------------------------------------------------------------------
// Baca (sama dengan png_io_stb.cpp)
// ---------------------------------------------------------------------------------------

std::string reason() {
    const char* r = stbi_failure_reason();
    return r ? r : "alasan tidak diketahui";
}

// ---------------------------------------------------------------------------------------
// CRC32 (slicing-by-8) dan Adler32
// ---------------------------------------------------------------------------------------

struct CrcTables {
    uint32_t t[8][256];
    CrcTables() {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[0][i] = c;
        }
        for (uint32_t i = 0; i < 256; ++i)
            for (int s = 1; s < 8; ++s) t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xFF];
    }
};

const CrcTables& crc_tables() {
    static const CrcTables tabs;
    return tabs;
}

// crc = nilai internal (belum di-XOR akhir); mulai dari 0xFFFFFFFF.
uint32_t crc_update(uint32_t crc, const uint8_t* p, size_t n) {
    const CrcTables& T = crc_tables();
    while (n >= 8) {
        uint32_t a = crc ^ (static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 |
                            static_cast<uint32_t>(p[2]) << 16 | static_cast<uint32_t>(p[3]) << 24);
        crc = T.t[7][a & 0xFF] ^ T.t[6][(a >> 8) & 0xFF] ^ T.t[5][(a >> 16) & 0xFF] ^ T.t[4][a >> 24] ^
              T.t[3][p[4]] ^ T.t[2][p[5]] ^ T.t[1][p[6]] ^ T.t[0][p[7]];
        p += 8;
        n -= 8;
    }
    while (n--) crc = T.t[0][(crc ^ *p++) & 0xFF] ^ (crc >> 8);
    return crc;
}

constexpr uint32_t kAdlerBase = 65521;

uint32_t adler_of(const uint8_t* p, size_t n) {
    uint32_t a = 1, b = 0;
    while (n > 0) {
        const size_t m = std::min<size_t>(n, 5552);  // NMAX zlib: tanpa overflow 32-bit
        for (size_t i = 0; i < m; ++i) {
            a += p[i];
            b += a;
        }
        a %= kAdlerBase;
        b %= kAdlerBase;
        p += m;
        n -= m;
    }
    return a | (b << 16);
}

// Sama dengan adler32_combine milik zlib.
uint32_t adler_combine(uint32_t a1, uint32_t a2, size_t len2) {
    const uint32_t rem = static_cast<uint32_t>(len2 % kAdlerBase);
    uint32_t sum1 = a1 & 0xFFFF;
    uint32_t sum2 = (rem * sum1) % kAdlerBase;
    sum1 += (a2 & 0xFFFF) + kAdlerBase - 1;
    sum2 += ((a1 >> 16) & 0xFFFF) + ((a2 >> 16) & 0xFFFF) + kAdlerBase - rem;
    if (sum1 >= kAdlerBase) sum1 -= kAdlerBase;
    if (sum1 >= kAdlerBase) sum1 -= kAdlerBase;
    if (sum2 >= (kAdlerBase << 1)) sum2 -= (kAdlerBase << 1);
    if (sum2 >= kAdlerBase) sum2 -= kAdlerBase;
    return sum1 | (sum2 << 16);
}

// ---------------------------------------------------------------------------------------
// Deflate: LZ77 + Huffman tetap (RFC 1951, blok tipe 01)
// ---------------------------------------------------------------------------------------

struct DeflateTables {
    uint16_t lit_code[288];  // kode sudah dibalik bit (siap ditulis LSB dulu)
    uint8_t lit_len[288];
    uint16_t len_sym[259];   // panjang 3..258 -> simbol 257..285
    uint8_t len_xbits[259];
    uint16_t len_xval[259];
    DeflateTables() {
        auto rev = [](uint32_t code, int n) {
            uint32_t r = 0;
            for (int i = 0; i < n; ++i) r |= ((code >> i) & 1u) << (n - 1 - i);
            return r;
        };
        for (int s = 0; s < 288; ++s) {
            uint32_t code;
            int n;
            if (s <= 143) { code = 0x30 + s; n = 8; }
            else if (s <= 255) { code = 0x190 + (s - 144); n = 9; }
            else if (s <= 279) { code = s - 256; n = 7; }
            else { code = 0xC0 + (s - 280); n = 8; }
            lit_code[s] = static_cast<uint16_t>(rev(code, n));
            lit_len[s] = static_cast<uint8_t>(n);
        }
        static const int base[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
                                     35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
        static const int xb[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                   3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
        for (int len = 3; len <= 258; ++len) {
            int c = 28;
            if (len != 258) {
                c = 27;
                while (base[c] > len) --c;
            }
            len_sym[len] = static_cast<uint16_t>(257 + c);
            len_xbits[len] = static_cast<uint8_t>(xb[c]);
            len_xval[len] = static_cast<uint16_t>(len - base[c]);
        }
    }
};

const DeflateTables& dtab() {
    static const DeflateTables t;
    return t;
}

// Kode jarak 0..29 untuk jarak 1..32768.
inline int dist_code(uint32_t d, int& xbits, uint32_t& xval) {
    static const uint32_t base[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385,
                                      513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
    static const int xb[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    int c = 29;
    while (base[c] > d) --c;
    xbits = xb[c];
    xval = d - base[c];
    return c;
}

struct BitWriter {
    std::vector<uint8_t>& out;
    uint64_t acc = 0;
    int n = 0;
    explicit BitWriter(std::vector<uint8_t>& o) : out(o) {}
    inline void put(uint32_t bits, int cnt) {
        acc |= static_cast<uint64_t>(bits) << n;
        n += cnt;
        while (n >= 8) {
            out.push_back(static_cast<uint8_t>(acc & 0xFF));
            acc >>= 8;
            n -= 8;
        }
    }
    inline void align() {
        if (n > 0) {
            out.push_back(static_cast<uint8_t>(acc & 0xFF));
            acc = 0;
            n = 0;
        }
    }
};

constexpr int kHashBits = 15;
constexpr uint32_t kWindow = 32768;
constexpr int kMaxChain = 8;

// Mampatkan `d[0..len)` menjadi satu blok Huffman tetap (BFINAL = last). Bila bukan strip terakhir,
// ditutup blok stored kosong agar keluaran berakhir tepat di batas byte dan bisa disambung.
void deflate_strip(const uint8_t* d, size_t len, bool last, std::vector<uint8_t>& out,
                   std::vector<int32_t>& head, std::vector<int32_t>& prev) {
    const DeflateTables& T = dtab();
    BitWriter bw(out);
    bw.put(last ? 1u : 0u, 1);
    bw.put(1u, 2);  // BTYPE = 01 (Huffman tetap)

    std::fill(head.begin(), head.end(), -1);
    auto hash3 = [&](size_t p) {
        const uint32_t v = static_cast<uint32_t>(d[p]) | static_cast<uint32_t>(d[p + 1]) << 8 |
                           static_cast<uint32_t>(d[p + 2]) << 16;
        return (v * 2654435761u) >> (32 - kHashBits);
    };
    auto insert = [&](size_t p) {
        if (p + 3 > len) return;
        const uint32_t h = hash3(p);
        prev[p & (kWindow - 1)] = head[h];
        head[h] = static_cast<int32_t>(p);
    };

    size_t i = 0;
    while (i < len) {
        size_t best_len = 0;
        uint32_t best_dist = 0;
        if (i + 3 <= len) {
            const size_t maxl = std::min<size_t>(258, len - i);
            int32_t cand = head[hash3(i)];
            int tries = kMaxChain;
            while (cand >= 0 && i - static_cast<size_t>(cand) <= kWindow && tries-- > 0) {
                const uint8_t* a = d + i;
                const uint8_t* b = d + cand;
                size_t l = 0;
                while (l < maxl && a[l] == b[l]) ++l;
                if (l > best_len) {
                    best_len = l;
                    best_dist = static_cast<uint32_t>(i - static_cast<size_t>(cand));
                    if (l >= maxl) break;
                }
                const int32_t nxt = prev[static_cast<size_t>(cand) & (kWindow - 1)];
                if (nxt >= cand) break;  // rantai basi (tertimpa jendela)
                cand = nxt;
            }
        }
        if (best_len >= 3) {
            const uint32_t L = static_cast<uint32_t>(best_len);
            bw.put(T.lit_code[T.len_sym[L]], T.lit_len[T.len_sym[L]]);
            if (T.len_xbits[L]) bw.put(T.len_xval[L], T.len_xbits[L]);
            int xb;
            uint32_t xv;
            const int dc = dist_code(best_dist, xb, xv);
            // kode jarak tetap 5 bit, MSB dulu
            uint32_t r = 0;
            for (int k = 0; k < 5; ++k) r |= ((static_cast<uint32_t>(dc) >> k) & 1u) << (4 - k);
            bw.put(r, 5);
            if (xb) bw.put(xv, xb);
            const size_t ins = std::min<size_t>(best_len, 16);  // sisanya tidak diindeks (cepat)
            for (size_t k = 0; k < ins; ++k) insert(i + k);
            i += best_len;
        } else {
            bw.put(T.lit_code[d[i]], T.lit_len[d[i]]);
            insert(i);
            ++i;
        }
    }
    bw.put(T.lit_code[256], T.lit_len[256]);  // akhir blok
    if (!last) {
        bw.put(0u, 1);  // blok stored kosong (BFINAL = 0): menyelaraskan ke batas byte
        bw.put(0u, 2);
        bw.align();
        const uint8_t sync[4] = {0x00, 0x00, 0xFF, 0xFF};
        out.insert(out.end(), sync, sync + 4);
    } else {
        bw.align();
    }
}

// ---------------------------------------------------------------------------------------
// Penulis PNG
// ---------------------------------------------------------------------------------------

inline void put_be32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v >> 24);
    p[1] = static_cast<uint8_t>(v >> 16);
    p[2] = static_cast<uint8_t>(v >> 8);
    p[3] = static_cast<uint8_t>(v);
}

bool write_chunk(std::FILE* f, const char type[4], const uint8_t* data, size_t n) {
    uint8_t hdr[8];
    put_be32(hdr, static_cast<uint32_t>(n));
    std::memcpy(hdr + 4, type, 4);
    uint32_t crc = crc_update(0xFFFFFFFFu, hdr + 4, 4);
    if (n) crc = crc_update(crc, data, n);
    uint8_t tail[4];
    put_be32(tail, crc ^ 0xFFFFFFFFu);
    return std::fwrite(hdr, 1, 8, f) == 8 && (n == 0 || std::fwrite(data, 1, n, f) == n) &&
           std::fwrite(tail, 1, 4, f) == 4;
}

struct Piece {
    const uint8_t* p;
    size_t n;
};

bool write_png(const std::string& path, const uint8_t* px, int w, int h, int comp, std::string& err) {
    if (w < 1 || h < 1 || px == nullptr || (comp != 1 && comp != 3)) {
        err = path + ": ukuran atau data PNG tidak sah";
        return false;
    }
    const size_t W = static_cast<size_t>(w), H = static_cast<size_t>(h), C = static_cast<size_t>(comp);
    const size_t stride = W * C;
    const size_t row_bytes = stride + 1;
    const size_t strip_rows = std::max<size_t>(1, static_cast<size_t>(LC_PNG_STRIP_BYTES) / row_bytes);
    const size_t nstrips = (H + strip_rows - 1) / strip_rows;

    std::vector<std::vector<uint8_t>> parts(nstrips);
    std::vector<uint32_t> adler(nstrips, 1);
    std::vector<size_t> raw_len(nstrips, 0);
    std::atomic<bool> failed{false};

    // Seluruh pekerjaan berat (filter + adler + deflate) paralel per strip. Hasil tiap strip hanya
    // bergantung pada datanya sendiri, jadi keluaran tidak bergantung pada jumlah thread.
#pragma omp parallel
    {
        std::vector<uint8_t> filt;
        std::vector<int32_t> head, prev;
        try {
            head.assign(size_t{1} << kHashBits, -1);
            prev.assign(kWindow, -1);
            filt.resize(strip_rows * row_bytes);
        } catch (...) {
failed.store(true);
        }
#pragma omp for schedule(dynamic, 1)
        for (long long s = 0; s < static_cast<long long>(nstrips); ++s) {
            if (failed.load()) continue;
            try {
                const size_t y0 = static_cast<size_t>(s) * strip_rows;
                const size_t y1 = std::min(H, y0 + strip_rows);
                uint8_t* o = filt.data();
                for (size_t y = y0; y < y1; ++y) {
                    const uint8_t* src = px + y * stride;
                    *o++ = static_cast<uint8_t>(LC_PNG_FILTER == 1 ? 1 : 0);
#if LC_PNG_FILTER == 1
                    for (size_t x = 0; x < C && x < stride; ++x) o[x] = src[x];
                    for (size_t x = C; x < stride; ++x) o[x] = static_cast<uint8_t>(src[x] - src[x - C]);
#else
                    std::memcpy(o, src, stride);
#endif
                    o += stride;
                }
                const size_t len = static_cast<size_t>(o - filt.data());
                raw_len[static_cast<size_t>(s)] = len;
                adler[static_cast<size_t>(s)] = adler_of(filt.data(), len);
                std::vector<uint8_t>& out = parts[static_cast<size_t>(s)];
                out.reserve(len / 4 + 64);
                deflate_strip(filt.data(), len, static_cast<size_t>(s) + 1 == nstrips, out, head, prev);
            } catch (...) {
failed.store(true);
            }
        }
    }
    if (failed.load()) {
        err = path + ": memori tidak cukup untuk menulis PNG";
        return false;
    }

    // Sambung berurutan (serial): header zlib, strip, adler32 gabungan.
    uint32_t ad = adler[0];
    for (size_t s = 1; s < nstrips; ++s) ad = adler_combine(ad, adler[s], raw_len[s]);
    const uint8_t zhdr[2] = {0x78, 0x01};
    uint8_t ztail[4];
    put_be32(ztail, ad);
    std::vector<Piece> pieces;
    pieces.push_back({zhdr, 2});
    for (auto& p : parts) pieces.push_back({p.data(), p.size()});
    pieces.push_back({ztail, 4});
    size_t total = 0;
    for (auto& pc : pieces) total += pc.n;

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        err = path + ": gagal menulis PNG";
        return false;
    }
    bool ok = true;
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    ok = std::fwrite(sig, 1, 8, f) == 8;
    uint8_t ihdr[13];
    put_be32(ihdr, static_cast<uint32_t>(w));
    put_be32(ihdr + 4, static_cast<uint32_t>(h));
    ihdr[8] = 8;
    ihdr[9] = comp == 1 ? 0 : 2;
    ihdr[10] = ihdr[11] = ihdr[12] = 0;
    ok = ok && write_chunk(f, "IHDR", ihdr, 13);

    // IDAT: satu aliran zlib, dipecah per <= 1 GiB agar panjang chunk tetap sah.
    constexpr size_t kMaxIdat = size_t{1} << 30;
    std::vector<uint8_t> buf;
    size_t pi = 0, po = 0, remaining = total;
    while (ok && remaining > 0) {
        const size_t cl = std::min(remaining, kMaxIdat);
        try {
            buf.resize(cl);
        } catch (...) {
            ok = false;
            break;
        }
        size_t filled = 0;
        while (filled < cl) {
            const size_t take = std::min(cl - filled, pieces[pi].n - po);
            std::memcpy(buf.data() + filled, pieces[pi].p + po, take);
            filled += take;
            po += take;
            if (po == pieces[pi].n) { ++pi; po = 0; }
        }
        ok = write_chunk(f, "IDAT", buf.data(), cl);
        remaining -= cl;
    }
    ok = ok && write_chunk(f, "IEND", nullptr, 0);
    const bool closed = std::fclose(f) == 0;
    if (!ok || !closed) {
        err = path + ": gagal menulis PNG";
        return false;
    }
    return true;
}

}  // namespace

bool png_info(const std::string& path, int& width, int& height, std::string& err) {
    int comp = 0;
    if (!stbi_info(path.c_str(), &width, &height, &comp)) {
        err = path + ": bukan PNG yang bisa dibaca (" + reason() + ")";
        return false;
    }
    if (width < 1 || height < 1) {
        err = path + ": ukuran PNG tidak sah";
        return false;
    }
    return true;
}

bool png_read_rgb8(const std::string& path, Rgb8& out, std::string& err) {
    int w = 0, h = 0, comp = 0;
    uint8_t* px = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (px == nullptr) {
        err = path + ": gagal membaca PNG (" + reason() + ")";
        return false;
    }
    out.width = w;
    out.height = h;
    out.data.reset(px);  // stb memakai malloc, jadi std::free benar
    return true;
}

bool png_write_gray8(const std::string& path, const uint8_t* px, int width, int height, std::string& err) {
    return write_png(path, px, width, height, 1, err);
}

bool png_write_rgb8(const std::string& path, const uint8_t* px, int width, int height, std::string& err) {
    return write_png(path, px, width, height, 3, err);
}

}  // namespace lc
