#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <vector>

// DKQ=DV=256, GQA=4, and 3/4 queries select the 16-column Ampere tile.
// Swizzled K occupies 64*128 half2s, but combine metadata uses 64*132:
// without a barrier, the final metadata rows can overwrite live V reads.
static bool test_shape(ggml_backend_t backend, int queries, int kv) {
    constexpr int d = 256;
    constexpr int heads_kv = 4;
    constexpr int heads_q = 4*heads_kv;
    ggml_context * ctx = ggml_init({4*1024*1024, nullptr, true});
    if (!ctx) {
        return false;
    }
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, queries, heads_q, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kv, heads_kv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kv, heads_kv, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, queries);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/16.0f, 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }

    std::vector<float> q_data(ggml_nelements(q), 0.0f);
    std::vector<ggml_fp16_t> kv_data(ggml_nelements(k));
    std::vector<ggml_fp16_t> mask_data(ggml_nelements(mask), ggml_fp32_to_fp16(0.0f));
    constexpr float values[] = {-0.5f, -0.25f, 0.25f, 0.5f};
    for (size_t i = 0; i < kv_data.size(); ++i) {
        kv_data[i] = ggml_fp32_to_fp16(values[i % 4]);
    }
    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, kv_data.data(), 0, ggml_nbytes(k));
    // Zero V is an exact oracle even when softmax uses approximate exp/reciprocal:
    // a racing float metadata write changes its half2 payload to nonzero data.
    std::vector<ggml_fp16_t> v_data(ggml_nelements(v), ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(v, v_data.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask));

    bool ok = true;
    std::vector<float> result(ggml_nelements(out));
    for (int repeat = 0; repeat < 128 && ok; ++repeat) {
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        if (!ok) {
            break;
        }
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        // Finite Q/K and zero V must produce exact zero for every output.
        // Unlike a nonzero constant, this does not assume exact softmax math.
        for (size_t i = 0; i < result.size(); ++i) {
            if (!std::isfinite(result[i]) || result[i] != 0.0f) {
                std::fprintf(stderr, "FA combine mismatch queries=%d kv=%d repeat=%d index=%zu actual=%a expected=%a\n",
                             queries, kv, repeat, i, result[i], 0.0);
                ok = false;
                break;
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name(GGML_CUDA_NAME "0");
    if (!device) {
        std::puts("SKIP: no CUDA/HIP device");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) {
        return 1;
    }
    bool ok = true;
    for (int queries : {3, 4}) {
        for (int kv : {256, 512, 1024, 2048}) {
            ok = test_shape(backend, queries, kv) && ok;
        }
    }
    ggml_backend_free(backend);
    if (ok) {
        std::puts("PASS: FA combine metadata lifetime, 8 shapes x 128 repetitions");
    }
    return ok ? 0 : 1;
}
