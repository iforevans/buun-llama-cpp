#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

int main() {
    if (!ggml_backend_cuda_get_device_count()) {
        return 77;
    }
    auto * backend = ggml_backend_cuda_init(0);
    GGML_ASSERT(backend);

    // Include the bounds of the fused verification path, both fallback widths,
    // and the production head count. Observable gathers force the control path.
    for (auto [m, h] : {std::pair{1, 4}, {2, 4}, {8, 4}, {13, 4}, {16, 4}, {17, 4}, {8, 48}}) {
        for (bool scatter : {false, true}) {
            std::vector<float> reference;
            for (bool control : {true, false}) {
                auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
                constexpr int d = 128;
                const int size = d * d * h;
                auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, h, m, 1);
                auto * k = ggml_dup_tensor(ctx, q);
                auto * v = ggml_dup_tensor(ctx, q);
                auto * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, h, m, 1);
                auto * gate = ggml_dup_tensor(ctx, beta);
                auto * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, size, 32);
                auto * index = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
                auto * gather = ggml_get_rows(ctx, cache, index);
                if (control) {
                    ggml_set_output(gather);
                }
                auto * state = ggml_reshape_4d(ctx, gather, d, d, h, 1);
                auto * result = ggml_gated_delta_net(ctx, q, k, v, gate, beta, state, m);
                auto * scores = ggml_view_1d(ctx, result, d * h * m, 0);
                auto * graph = ggml_new_graph_custom(ctx, 128, false);
                if (scatter) {
                    const size_t stride = size * sizeof(float);
                    auto * src = ggml_view_3d(ctx, result, size, 1, m, stride, stride, d * h * m * sizeof(float));
                    auto * dst = ggml_view_3d(ctx, cache, size, 1, m, stride, stride, stride);
                    ggml_build_forward_expand(graph, ggml_cpy(ctx, src, dst));
                }
                ggml_build_forward_expand(graph, scores);
                auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                GGML_ASSERT(buffer);
                ggml_backend_buffer_clear(buffer, 0);
                for (auto * tensor : {q, k, v, beta, gate}) {
                    std::vector<float> data(ggml_nelements(tensor));
                    for (size_t i = 0; i < data.size(); ++i) {
                        data[i] = tensor == gate ? -0.2f : tensor == beta ? 0.3f : float(int(i * 37 % 101) - 50) * 0.001f;
                    }
                    ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
                }
                std::vector<float> initial(size * 32);
                for (size_t i = 0; i < initial.size(); ++i) {
                    initial[i] = float(int(i * 37 % 101) - 50) * 0.001f;
                }
                size_t step = 0;
                for (int32_t row : {0, 2, 1}) {
                    // Change the device index across graph replays. Restore the
                    // source, including rows overlapping the snapshot destination.
                    ggml_backend_tensor_set(cache, initial.data(), 0, ggml_nbytes(cache));
                    ggml_backend_tensor_set(index, &row, 0, sizeof(row));
                    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                    auto * second = scatter ? cache : result;
                    std::vector<float> output(ggml_nelements(scores) + ggml_nelements(second));
                    ggml_backend_tensor_get(scores, output.data(), 0, ggml_nbytes(scores));
                    ggml_backend_tensor_get(second, output.data() + ggml_nelements(scores), 0, ggml_nbytes(second));
                    for (float x : output) {
                        GGML_ASSERT(std::isfinite(x));
                    }
                    if (control) {
                        reference.insert(reference.end(), output.begin(), output.end());
                    } else {
                        GGML_ASSERT(std::memcmp(reference.data() + step * output.size(), output.data(), output.size() * sizeof(float)) == 0);
                    }
                    ++step;
                }
                ggml_backend_buffer_free(buffer);
                ggml_free(ctx);
            }
            std::printf("PASS m=%d h=%d scatter=%d\n", m, h, scatter);
        }
    }
    ggml_backend_free(backend);
}
