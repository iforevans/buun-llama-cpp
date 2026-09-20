#pragma once

#include "ggml.h"

#include <array>
#include <cstdint>
#include <limits>

// This is ladder provenance, not an error estimate. Legacy Turbo codecs are
// deliberately incomparable with TCQ; equal-type copies remain meaningful.
constexpr std::array<ggml_type, 6> VBR_TURBO_PRECISION_LADDER = {
    GGML_TYPE_F16, GGML_TYPE_TURBO8_0, GGML_TYPE_TURBO4_0,
    GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO1_TCQ,
};

inline int vbr_precision_rank(int32_t type, bool classic = false) noexcept {
    if (type == GGML_TYPE_F16) {
        return 0;
    }
    if (classic) {
        return type == GGML_TYPE_Q8_0 ? 1 : type == GGML_TYPE_Q4_0 ? 2 : -1;
    }
    for (size_t i = 0; i < VBR_TURBO_PRECISION_LADDER.size(); ++i) {
        if (type == VBR_TURBO_PRECISION_LADDER[i]) { return int(i); }
    }
    return -1;
}

// Unknown stays unknown. Promotion never erases a previous precision loss.
inline int32_t vbr_precision_merge(int32_t prior, int32_t encoded) noexcept {
    if (prior == encoded) {
        return prior;
    }
    const bool classic = prior == GGML_TYPE_Q8_0 || prior == GGML_TYPE_Q4_0 ||
                         encoded == GGML_TYPE_Q8_0 || encoded == GGML_TYPE_Q4_0;
    const int a = vbr_precision_rank(prior, classic);
    const int b = vbr_precision_rank(encoded, classic);
    return a < 0 || b < 0 ? -1 : a >= b ? prior : encoded;
}

// Initial conservative host-restore policy: at most one rung worse in any
// unit, and one quarter of a rung on average, weighted by row width. Higher
// precision in another unit never cancels a deficit. This is a layout-distance
// heuristic, not an MSE bound; keep it independent of matched token percentage.
struct vbr_precision_admission {
    uint64_t weight = 0;
    uint64_t deficit = 0;
    uint32_t worst_steps = 0;
    bool known = true;

    void add(int32_t effective, int32_t desired, uint64_t columns) noexcept {
        const bool classic = effective == GGML_TYPE_Q8_0 || effective == GGML_TYPE_Q4_0 ||
                             desired == GGML_TYPE_Q8_0 || desired == GGML_TYPE_Q4_0;
        const int a = vbr_precision_rank(effective, classic);
        const int b = vbr_precision_rank(desired, classic);
        if (columns == 0 || effective < 0 || desired < 0 ||
            effective >= GGML_TYPE_COUNT || desired >= GGML_TYPE_COUNT ||
            (effective != desired && (a < 0 || b < 0))) {
            known = false;
            return;
        }
        const uint32_t steps = a > b ? uint32_t(a-b) : 0;
        worst_steps = steps > worst_steps ? steps : worst_steps;
        const auto max = std::numeric_limits<uint64_t>::max();
        if (columns > max-weight || (steps && columns > (max-deficit)/steps)) {
            known = false;
            return;
        }
        weight += columns;
        deficit += columns*steps;
    }

    bool allowed() const noexcept {
        return known && weight != 0 && worst_steps <= 1 && deficit <= weight/4;
    }
};
