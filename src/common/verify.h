// verify.h - pembanding hasil K-Means bit demi bit (kontrak butir 3, target 100% identik).
#ifndef VERIFY_H
#define VERIFY_H

#include <cstddef>
#include <cstdint>
#include <cstdio>

// Mengembalikan jumlah selisih (0 = identik). Melaporkan selisih pertama ke `log` jika tidak null.
long verify_identical(const int32_t* lab_ref, const uint8_t* cen_ref,
                      const int32_t* lab_test, const uint8_t* cen_test,
                      size_t n, int k, FILE* log);

#endif  // VERIFY_H
