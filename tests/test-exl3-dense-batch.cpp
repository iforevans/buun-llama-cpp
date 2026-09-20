#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "test-exl3-gpu-common.h"

#ifdef EXL3_TEST_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// Compare every row of each projection with that row executed alone.
// SM86 also covers padded matrix tiles and wider verification batches.
// Rows deliberately have different scales/outliers: sharing an activation max
// across rows must not accidentally satisfy this test.
static bool check_batch(ggml_backend_t backend, int bits, int k, int n, bool head, int max_m, bool exact_bound = false) {
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * compute = ggml_init({4*1024*1024, nullptr, true});
    auto * w = ggml_new_tensor_2d(ctx, ggml_exl3_type(bits, 2), k, n);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, max_m);
    auto * svh = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n);
    auto * suh = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, k);
    ggml_set_name(w, head ? "output.weight" : "probe.weight");
    auto * graph = ggml_new_graph_custom(compute, 512, false);
    auto project = [&](int m, int row) {
        auto * v = ggml_view_2d(compute, x, k, m, x->nb[1], row*x->nb[1]);
        auto * y = ggml_mul_mat(compute, w, v);
        y->src[2] = svh;
        y->src[3] = suh;
        ggml_set_output(y);
        ggml_build_forward_expand(graph, y);
        return y;
    };
    std::vector<ggml_tensor *> single(max_m), batch(max_m);
    for (int row = 0; row < max_m; ++row) single[row] = project(1, row);
    for (int m = 1; m <= max_m; ++m) batch[m-1] = project(m, 0);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    auto * alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(buffer && ggml_gallocr_alloc_graph(alloc, graph));
    unsigned rng = 1234567;
    auto next = [&]() { rng = rng*1664525u + 1013904223u; return rng; };
    std::vector<unsigned char> weights(ggml_nbytes(w));
    for (auto & v : weights) v = next() >> 24;
    if (exact_bound) std::fill(weights.begin(), weights.end(), 0);
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    std::vector<float> input(k*max_m);
    const float scales[] = {1.0f, 0.0f, 0.001f, 100.0f, 0.1f, 10.0f, 1.0f, 0.01f};
    for (size_t i = 0; i < input.size(); ++i) {
        float v = 0;
        for (int j = 0; j < 6; ++j) v += float(next() >> 8)/16777216.0f - 0.5f;
        input[i] = v*scales[(i/k)%8]*(i%127 == 0 ? 8.0f : 1.0f);
        // A Hadamard impulse gives a constant transformed row, quantized to
        // +/-127. Zero trellis windows give centered weight -512: exercise the
        // F16/F32 executor's worst-case exact-integer accumulation bound.
        if (exact_bound) input[i] = i % 128 == 0 ? 16.f * scales[(i/k)%8] : 0.f;
    }
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    std::vector<ggml_fp16_t> signs(k), output_scales(n);
    for (int i = 0; i < k; ++i) signs[i] = ggml_fp32_to_fp16(exact_bound || i%3 ? 1.0f : -1.0f);
    for (int i = 0; i < n; ++i) {
        output_scales[i] = ggml_fp32_to_fp16((i%7 ? 1.0f : -1.0f)*(0.01f + float(i%13)*0.002f));
    }
    ggml_backend_tensor_set(suh, signs.data(), 0, ggml_nbytes(suh));
    ggml_backend_tensor_set(svh, output_scales.data(), 0, ggml_nbytes(svh));
    std::vector<float> expected(n*max_m), actual(n*max_m), first;
    bool ok = true;
    for (int pass = 0; pass < 4; ++pass) {
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        for (int row = 0; row < max_m; ++row) {
            ggml_backend_tensor_get(single[row], expected.data() + row*n, 0, ggml_nbytes(single[row]));
        }
        for (float v : expected) ok &= std::isfinite(v);
        if (pass == 0) first = expected;
        else ok &= memcmp(first.data(), expected.data(), expected.size()*sizeof(float)) == 0;
        for (int m = 1; m <= max_m; ++m) {
            ggml_backend_tensor_get(batch[m-1], actual.data(), 0, ggml_nbytes(batch[m-1]));
            const bool same = memcmp(expected.data(), actual.data(), ggml_nbytes(batch[m-1])) == 0;
            if (!same) printf("MISMATCH K%d k=%d n=%d head=%d M=%d pass=%d\n", bits, k, n, head, m, pass);
            ok &= same;
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(compute);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    printf("K%d k=%d n=%d head=%d bound=%d: %s\n", bits, k, n, head, exact_bound, ok ? "PASS" : "FAIL");
    return ok;
}

// The joint graph may bundle projections, while each one-node graph cannot.
// Independent signs and unequal output widths catch accidental row concatenation.
static bool check_pair(ggml_backend_t backend, int n0, int n1, int m, bool shared_input, int k = 5120) {
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    auto * x2 = shared_input ? x : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor * w[2], * suh[2], * svh[2], * y[2];
    ggml_cgraph * single[2];
    auto * together = ggml_new_graph_custom(ctx, 16, false);
    for (int i = 0; i < 2; ++i) {
        const int n = i == 0 ? n0 : n1;
        w[i] = ggml_new_tensor_2d(ctx, ggml_exl3_type(4, 2), k, n);
        suh[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, k);
        svh[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n);
        y[i] = ggml_mul_mat(ctx, w[i], i == 0 ? x : x2);
        y[i]->src[2] = svh[i];
        y[i]->src[3] = suh[i];
        ggml_set_output(y[i]);
        single[i] = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(single[i], y[i]);
        ggml_build_forward_expand(together, y[i]);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    unsigned rng = 456789;
    auto next = [&]() { return rng = rng*1664525u + 1013904223u; };
    std::vector<float> input(k*m);
    for (auto & v : input) v = float(int(next() >> 16) - 32768)/32768.f;
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    if (!shared_input) {
        for (auto & v : input) v *= -2.f;
        ggml_backend_tensor_set(x2, input.data(), 0, ggml_nbytes(x2));
    }
    for (int i = 0; i < 2; ++i) {
        std::vector<unsigned char> weights(ggml_nbytes(w[i]));
        for (auto & v : weights) v = next() >> 24;
        ggml_backend_tensor_set(w[i], weights.data(), 0, weights.size());
        std::vector<ggml_fp16_t> signs(k), scales(svh[i]->ne[0]);
        for (auto & v : signs) v = ggml_fp32_to_fp16(next() & 256 ? 1.f : -1.f);
        for (auto & v : scales) v = ggml_fp32_to_fp16(next() & 256 ? .01f : -.02f);
        ggml_backend_tensor_set(suh[i], signs.data(), 0, ggml_nbytes(suh[i]));
        ggml_backend_tensor_set(svh[i], scales.data(), 0, ggml_nbytes(svh[i]));
    }
    std::vector<float> expected[2];
    bool ok = true;
    for (int i = 0; i < 2; ++i) {
        GGML_ASSERT(ggml_backend_graph_compute(backend, single[i]) == GGML_STATUS_SUCCESS);
        expected[i].resize(ggml_nelements(y[i]));
        ggml_backend_tensor_get(y[i], expected[i].data(), 0, ggml_nbytes(y[i]));
        for (float v : expected[i]) ok &= std::isfinite(v);
    }
    for (int pass = 0; pass < 4; ++pass) {
        GGML_ASSERT(ggml_backend_graph_compute(backend, together) == GGML_STATUS_SUCCESS);
        for (int i = 0; i < 2; ++i) {
            std::vector<float> actual(expected[i].size());
            ggml_backend_tensor_get(y[i], actual.data(), 0, ggml_nbytes(y[i]));
            ok &= memcmp(actual.data(), expected[i].data(), ggml_nbytes(y[i])) == 0;
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    printf("pair K%d N%d/%d M%d shared=%d: %s\n", k, n0, n1, m, shared_input, ok ? "PASS" : "FAIL");
    return ok;
}

#ifdef EXL3_TEST_CUDA
bool test_cuda_exl3_sm86_image();
#endif

int main() {
    auto * reg = ggml_backend_cuda_reg();
    if (ggml_backend_reg_dev_count(reg) == 0) return 77;
    auto * backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    GGML_ASSERT(backend);
    if (!exl3_gpu_supported(backend)) {
        ggml_backend_free(backend);
        return 77;
    }
    bool sm86 = false, sm75 = false;
#ifdef EXL3_TEST_CUDA
    cudaDeviceProp props{};
    GGML_ASSERT(cudaGetDeviceProperties(&props, 0) == cudaSuccess);
    sm86 = props.major == 8 && props.minor == 6 && test_cuda_exl3_sm86_image();
    sm75 = props.major == 7 && props.minor == 5;
#endif
    bool ok = true;
    for (int bits = 1; bits <= 8; ++bits) {
        const int max_m = sm86 && bits >= 2 ? (bits <= 4 ? 16 : 13) : 8;
        ok &= check_batch(backend, bits, 128, 128, false, max_m);  // single K split, half a column block
        ok &= check_batch(backend, bits, 5120, 640, false, max_m);  // partial N tile, many K slices
        ok &= check_batch(backend, bits, 2048, 6144, false, max_m);
        ok &= check_batch(backend, bits, 4096, 151936, true, max_m); // cap-limited split, head residual
    }
    for (int m : {1, 4, 5, 6, 7, 8, 13}) {
        if (m > 8 && !sm86) continue;
        ok &= check_pair(backend, 12288, 6144, m, true);
        ok &= check_pair(backend, 640, 128, m, true);
        ok &= check_pair(backend, 17408, 17408, m, true);
        ok &= check_pair(backend, 12288, 6144, m, false);
    }
    if (sm86 || sm75) ok &= check_batch(backend, 4, 5120, 17408, false, 8, true);
    // Thirteen live rows within a sixteen-row tile: the down projection's
    // shared partials must omit padding without changing any reduction order.
    if (sm86) {
        ok &= check_batch(backend, 4, 17408, 5120, false, 16);
        for (int m : {8, 13, 16}) {
            ok &= check_pair(backend, 5120, 6144, m, true, 17408);
            ok &= check_pair(backend, 5120, 4352, m, false, 17408);
        }
    }
    if (sm86) for (int k : {128, 384, 640}) {
        ok &= check_batch(backend, 6, k, 131072, true, 13);
    }
    if (sm86) for (int m : {4, 5, 6, 7, 8, 13}) {
        ok &= check_pair(backend, 6144, 12288, m, true);  // larger second projection
        ok &= check_pair(backend, 4096, 4352, m, true);   // minimum paired width, unequal K slices
        ok &= check_pair(backend, 17408, 16384, m, true); // unequal wide projections
        ok &= check_pair(backend, 17408, 12288, m, true); // different executor shapes fall back
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
