// verify.cpp - implementasi pembanding label dan centroid.
#include "verify.h"

long verify_identical(const int32_t* lab_ref, const uint8_t* cen_ref,
                      const int32_t* lab_test, const uint8_t* cen_test,
                      size_t n, int k, FILE* log) {
    long diff = 0;
    for (size_t i = 0; i < n; i++) {
        if (lab_ref[i] != lab_test[i]) {
            if (diff == 0 && log)
                std::fprintf(log, "label pertama beda di piksel %zu: ref=%d test=%d\n",
                             i, lab_ref[i], lab_test[i]);
            diff++;
        }
    }
    for (int i = 0; i < 3 * k; i++) {
        if (cen_ref[i] != cen_test[i]) {
            if (log) std::fprintf(log, "centroid beda di byte %d\n", i);
            diff++;
        }
    }
    return diff;
}
