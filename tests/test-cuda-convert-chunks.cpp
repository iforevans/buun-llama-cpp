#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstring>
#include <vector>

static float marker(int64_t row, int column, int batch) {
    return float((row % 7) - 3 + 8*column + 32*batch) / 8.0f;
}

static bool test_shape(ggml_backend_t backend, ggml_type type, int64_t rows, int batches) {
    constexpr int64_t k = 512;
    constexpr int columns = 3;
    ggml_context * ctx = ggml_init({4*1024*1024, nullptr, true});
    if (!ctx) {
        return false;
    }
    ggml_tensor * a = ggml_new_tensor_3d(ctx, type, k, rows, batches);
    // Non-F32 RHS selects the cuBLAS path, not the small-column MMVF path.
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, columns, batches);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "conversion boundary allocation failed\n");
        ggml_free(ctx);
        return false;
    }

    // Approximately 256 MiB stored A, at most 512 MiB converted temporary,
    // and a small output. No multi-gigabyte fixture or CPU reference GEMM.
    std::vector<unsigned char> weights(ggml_nbytes(a), 0);
    for (int batch = 0; batch < batches; ++batch) {
        for (int64_t row = 0; row < rows; ++row) {
            for (int column = 0; column < columns; ++column) {
                const size_t offset = (size_t(batch)*rows*k + row*k + column)*sizeof(ggml_fp16_t);
                const float value = marker(row, column, batch);
                if (type == GGML_TYPE_F16) {
                    const ggml_fp16_t encoded = ggml_fp32_to_fp16(value);
                    std::memcpy(weights.data() + offset, &encoded, sizeof(encoded));
                } else {
                    const ggml_bf16_t encoded = ggml_fp32_to_bf16(value);
                    std::memcpy(weights.data() + offset, &encoded, sizeof(encoded));
                }
            }
        }
    }
    std::vector<ggml_fp16_t> rhs(ggml_nelements(b), ggml_fp32_to_fp16(0.0f));
    for (int batch = 0; batch < batches; ++batch) {
        for (int column = 0; column < columns; ++column) {
            rhs[(batch*columns + column)*k + column] = ggml_fp32_to_fp16(1.0f);
        }
    }
    ggml_backend_tensor_set(a, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(b, rhs.data(), 0, ggml_nbytes(b));
    bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> result(ggml_nelements(out));
    if (ok) {
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        for (int batch = 0; batch < batches && ok; ++batch) {
            for (int column = 0; column < columns && ok; ++column) {
                for (int64_t row = 0; row < rows; ++row) {
                    const float expected = marker(row, column, batch);
                    const float actual = result[(size_t(batch)*columns + column)*rows + row];
                    if (actual != expected) {
                        std::fprintf(stderr, "conversion mismatch type=%s rows=%lld batches=%d batch=%d column=%d row=%lld actual=%a expected=%a\n",
                                     ggml_type_name(type), (long long) rows, batches, batch, column,
                                     (long long) row, actual, expected);
                        ok = false;
                        break;
                    }
                }
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::printf("conversion %s rows=%lld batches=%d: %s\n", ggml_type_name(type),
                (long long) rows, batches, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name("CUDA0");
    if (!device) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) {
        return 1;
    }
    bool ok = true;
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_BF16}) {
        // F32 conversion footprint: 512 MiB minus one row, exactly 512 MiB,
        // and one row over; the last case must keep the full output ldc.
        for (int64_t rows : {262143, 262144, 262145}) {
            ok = test_shape(backend, type, rows, 1) && ok;
        }
        // The conversion budget includes both batches; final chunk has one row.
        // Check both ldc and strideC while src0's batch stride remains padded.
        ok = test_shape(backend, type, 131073, 2) && ok;
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
