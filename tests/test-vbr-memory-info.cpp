#include "ggml-cuda.h"

#include <cstdio>

// Run on an otherwise idle GPU. Small VMM mappings must not consume many times
// their physical size in the free-memory reading used by fit and VBR.
int main() {
    const auto * be = ggml_backend_cuda_vbr_iface();
    constexpr size_t mib = 1024 * 1024;
    constexpr size_t bytes = 2048 * mib;
    if (!be || !be->get_device_count() || !be->vmm_available(0)) {
        return 77;
    }
    const size_t gran = be->vmm_granularity(0);
    if (!gran || bytes % gran) {
        return 77;
    }
    auto * backend = be->backend_init(0);
    if (!backend) {
        return 1;
    }
    size_t before, total;
    be->get_device_memory(0, &before, &total);
    if (before < 6144 * mib) {
        std::puts("SKIP: memory accounting gate requires 6 GiB free on an idle GPU");
        ggml_backend_free(backend);
        return 77;
    }
    auto * pool = be->vmm_pool_init(0, bytes);
    if (!pool || !be->vmm_pool_map(pool, 0, bytes)) {
        if (pool) { be->vmm_pool_free(pool); }
        ggml_backend_free(backend);
        return 1;
    }
    be->sync_device(0);
    size_t mapped_free, fit_free, fit_total;
    be->get_device_memory(0, &mapped_free, &total);
    ggml_backend_dev_memory(ggml_backend_get_device(backend), &fit_free, &fit_total);
    // Allow lazy runtime workspaces and page rounding, but not ROCm's former
    // 8x charge (2 GiB at 256 KiB granularity could report zero free on 16 GiB).
    bool ok = mapped_free + bytes + 512 * mib >= before &&
              fit_free + bytes + 512 * mib >= before && fit_total == total;
    std::printf("VMM memory info: before=%zu mapped=%zu fit=%zu live=%zu gran=%zu\n",
            before, mapped_free, fit_free, be->vmm_pool_mapped(pool), gran);

    // Physical pressure must still be visible through both public query doors.
    auto * pressure = ggml_backend_buft_alloc_buffer(be->buffer_type(0), 512 * mib);
    if (!pressure) {
        ok = false;
    } else {
        ggml_backend_buffer_clear(pressure, 0);
        be->sync_device(0);
        size_t pressured;
        be->get_device_memory(0, &pressured, &total);
        ggml_backend_dev_memory(ggml_backend_get_device(backend), &fit_free, &fit_total);
        ok = pressured + 384 * mib <= mapped_free && fit_free + 384 * mib <= mapped_free && ok;
        std::printf("physical pressure: controller=%zu fit=%zu\n", pressured, fit_free);
        ggml_backend_buffer_free(pressure);
    }
    be->vmm_pool_free(pool);
    ggml_backend_free(backend);
    std::puts(ok ? "PASS: physical VRAM accounting" : "FAIL: physical VRAM accounting");
    return ok ? 0 : 1;
}
