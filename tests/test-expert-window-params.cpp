#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend);
    ggml_backend_cpu_set_n_threads(backend, 2);
    constexpr int k = 32, m = 4, experts = 3, used = 4, tokens = 2;
    for (bool window : {false, true}) {
        for (auto precision : {GGML_PREC_UNDEFINED, GGML_PREC_Q8}) {
            ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true});
            GGML_ASSERT(ctx);
            auto * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, m, experts);
            auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, used, tokens);
            auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
            auto * y = ggml_mul_mat_id(ctx, w, x, ids);
            if (window) ggml_mul_mat_id_set_expert_window(y, 7, experts);
            GGML_ASSERT(ggml_prec_set_src(y, precision, 1));
            GGML_ASSERT(ggml_prec_set_acc(y, GGML_PREC_F32));
            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, y);
            auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            GGML_ASSERT(buffer);

            const std::vector<int32_t> routed = window
                ? std::vector<int32_t>{6, 7, 9, 10, -1, 8, 7, 10}
                : std::vector<int32_t>{-1, 0, 2, -1, -1, 1, 0, -1};
            std::vector<float> weights(k * m * experts), acts(k * used * tokens);
            for (int e = 0; e < experts; ++e)
                for (int j = 0; j < m; ++j)
                    for (int i = 0; i < k; ++i)
                        weights[(e*m+j)*k+i] = (i%5-2)*0.125f + e*0.25f + j*0.5f;
            for (size_t i = 0; i < acts.size(); ++i) acts[i] = (i%3)*0.25f;
            ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
            ggml_backend_tensor_set(x, acts.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(ids, routed.data(), 0, ggml_nbytes(ids));
            ggml_backend_tensor_memset(y, 0xff, 0, ggml_nbytes(y));
            GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            std::vector<float> out(m * used * tokens);
            ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
            for (int row = 0; row < used*tokens; ++row) {
                const int expert = routed[row] - (window ? 7 : 0);
                for (int j = 0; j < m; ++j) {
                    float expected = 0.0f;
                    if (expert >= 0 && expert < experts)
                        for (int i = 0; i < k; ++i)
                            expected += weights[(expert*m+j)*k+i] * acts[row*k+i];
                    GGML_ASSERT(std::isfinite(out[row*m+j]));
                    GGML_ASSERT(std::fabs(out[row*m+j] - expected) <= 1e-5f * (1.0f + std::fabs(expected)));
                }
            }
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(backend);
    puts("CPU expert-window precision: 4 explicit-slice cases PASS");
}
