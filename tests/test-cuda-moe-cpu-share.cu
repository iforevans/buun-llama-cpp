#include "../ggml/src/ggml-cuda/moe-cpu-share.cuh"
#include "../ggml/src/ggml-cuda/moe-weighted-reduction.cuh"
#include "../ggml/src/ggml-backend-moe-cache.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Exercise the real doorbell, host workers and merge handoff, including replay
// of graphs captured with different activation limits. All routes are misses
// and CPU_SHARE=0 makes every output come from a host worker (no silent fallback).
static void set_env(const char * key, const char * value) {
#ifdef _WIN32
    GGML_ASSERT(_putenv_s(key, value) == 0);
#else
    GGML_ASSERT(setenv(key, value, 1) == 0);
#endif
}

static bool check(ggml_backend_t gpu, ggml_backend_t cpu, ggml_type type, int tokens, int used) {
    constexpr int width = 256, experts = 8;
    auto * wc = ggml_init({8*ggml_tensor_overhead(), nullptr, true});
    auto * up = ggml_new_tensor_3d(wc, type, width, width, experts);
    auto * gate = ggml_dup_tensor(wc, up);
    auto * down = ggml_dup_tensor(wc, up);
    ggml_set_name(up, "blk.0.ffn_up_exps.weight");
    ggml_set_name(gate, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(down, "blk.0.ffn_down_exps.weight");
    auto * wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, ggml_backend_cuda_host_buffer_type());
    GGML_ASSERT(wb);
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> row(width);
    int matrix = 0;
    for (auto * w : {up, gate, down}) {
        for (int e = 0; e < experts; ++e) {
            for (int r = 0; r < width; ++r) {
                for (int k = 0; k < width; ++k) {
                    row[k] = .12f*std::sin(float(k + 7*r + 11*e + matrix)) +
                        (r%2 ? -.25f : .25f);
                }
                ggml_get_type_traits(type)->from_float_ref(row.data(),
                    (char *) w->data + e*w->nb[2] + r*w->nb[1], width);
            }
        }
        ++matrix;
    }
    auto * gc = ggml_init({32*ggml_tensor_overhead(), nullptr, true});
    auto * x = ggml_new_tensor_3d(gc, GGML_TYPE_F32, width, 1, tokens);
    auto * ids = ggml_new_tensor_2d(gc, GGML_TYPE_I32, used, tokens);
    auto * gu = ggml_mul_mat_id(gc, up, x, ids);
    auto * gg = ggml_mul_mat_id(gc, gate, x, ids);
    auto * gd = ggml_mul_mat_id(gc, down, gu, ids);
    auto * weights = ggml_new_tensor_3d(gc, GGML_TYPE_F32, 1, used, tokens);
    auto * result = ggml_new_tensor_2d(gc, GGML_TYPE_F32, width, tokens);
    auto * gb = ggml_backend_alloc_ctx_tensors(gc, gpu);
    GGML_ASSERT(gb);
    std::vector<float> acts(width*tokens);
    std::vector<int32_t> routes(used*tokens);
    std::vector<float> scales(used*tokens);
    for (size_t i = 0; i < acts.size(); ++i) acts[i] = .8f + .1f*std::sin(float(i));
    for (size_t i = 0; i < routes.size(); ++i) routes[i] = i%experts;
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = 1.0f/(1 + i%used);
    ggml_backend_tensor_set(x, acts.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
    ggml_backend_tensor_set(weights, scales.data(), 0, ggml_nbytes(weights));
    void * backends[] = {gpu, cpu};
    auto * session = ggml_moe_cache.session_create(backends, 2, nullptr);
    GGML_ASSERT(session);
    bool routed = true;
    for (auto * op : {gu, gg, gd}) routed = ggml_moe_cache.route_supported(session, gpu, op) && routed;
    GGML_ASSERT(routed || tokens > 1);
    if (!routed) {
        // past the type's MMVQ batch cap the op takes MMQ, which never reads the route table
        printf("cpu-share type=%s tokens=%d used=%d SKIP: not routed\n", ggml_type_name(type), tokens, used);
        ggml_moe_cache.session_destroy(session);
        ggml_backend_buffer_free(gb);
        ggml_backend_buffer_free(wb);
        ggml_free(gc);
        ggml_free(wc);
        return true;
    }
    auto & ctx = *(ggml_backend_cuda_context *) gpu->context;
    const auto stream = ctx.stream();
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.gate = gate;

    // Independent CPU graph reference, with CPU cache interception disabled.
    auto * rc = ggml_init({32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    auto * rx = ggml_dup_tensor(rc, x);
    auto * ri = ggml_dup_tensor(rc, ids);
    auto * ru = ggml_mul_mat_id(rc, up, rx, ri);
    auto * rg = ggml_mul_mat_id(rc, gate, rx, ri);
    const float limits[] = {0.0f, .2f, 10.0f};
    ggml_tensor * outputs[3];
    for (int j = 0; j < 3; ++j) {
        auto * glu = j ? ggml_swiglu_clamp(rc, rg, ru, limits[j]) : ggml_swiglu_split(rc, rg, ru);
        outputs[j] = ggml_mul_mat_id(rc, down, glu, ri);
    }
    auto * rb = ggml_backend_alloc_ctx_tensors(rc, cpu);
    GGML_ASSERT(rb);
    ggml_backend_tensor_set(rx, acts.data(), 0, ggml_nbytes(rx));
    ggml_backend_tensor_set(ri, routes.data(), 0, ggml_nbytes(ri));
    auto * graph = ggml_new_graph(rc);
    for (auto * out : outputs) ggml_build_forward_expand(graph, out);
    ggml_moe_cache.session_enter(nullptr);
    GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    ggml_moe_cache.session_leave(nullptr);

    cudaGraph_t captured[3]{};
    cudaGraphExec_t exec[3]{};
    auto launch = [&] {
        const auto begin = ggml_moe_cpu_share_begin(up, gate, ids, x, &fusion, stream);
        GGML_ASSERT(begin.skip);
        GGML_ASSERT(ggml_moe_cpu_share_down(down, ids, gd).skip);
        ggml_cuda_op_moe_weighted_reduction(ctx, gd, nullptr, weights, result);
    };
    for (int j = 0; j < 3; ++j) {
        fusion.glu_op = j ? GGML_GLU_OP_SWIGLU_CLAMP : GGML_GLU_OP_SWIGLU;
        fusion.glu_limit = limits[j];
        launch(); // allocate staging before capture
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
        launch();
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured[j]));
        CUDA_CHECK(cudaGraphInstantiate(&exec[j], captured[j], nullptr, nullptr, 0));
    }
    bool ok = true;
    float max_error = 0;
    std::vector<float> actual(width*tokens), expected(width*used*tokens);
    for (int repeat = 0; repeat < 6; ++repeat) {
        for (int j : {2, 0, 1}) {
            // The down matvec is intentionally absent: every row must be
            // supplied by the host, never by uninitialized GPU output.
            CUDA_CHECK(cudaMemsetAsync(gd->data, 0xff, ggml_nbytes(gd), stream));
            CUDA_CHECK(cudaGraphLaunch(exec[j], stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
            ggml_backend_tensor_get(outputs[j], expected.data(), 0, ggml_nbytes(outputs[j]));
            for (size_t i = 0; i < actual.size(); ++i) {
                const size_t t = i/width, col = i%width;
                float sum = 0;
                for (int e = 0; e < used; ++e) sum += scales[t*used+e]*expected[(t*used+e)*width+col];
                float err = std::abs(actual[i] - sum)/(1.0f + std::abs(sum));
                max_error = std::max(max_error, err);
                ok &= std::isfinite(actual[i]) && err < .001f;
            }
        }
    }
    // Other GLUs and invalid limits must not silently use the SwiGLU worker.
    fusion.glu_op = GGML_GLU_OP_SWIGLU_OAI;
    ok &= !ggml_moe_cpu_share_begin(up, gate, ids, x, &fusion, stream).skip;
    fusion.glu_op = GGML_GLU_OP_SWIGLU_CLAMP;
    for (float limit : {0.0f, -1.0f, INFINITY, NAN}) {
        fusion.glu_limit = limit;
        ok &= !ggml_moe_cpu_share_begin(up, gate, ids, x, &fusion, stream).skip;
    }
    printf("cpu-share type=%s tokens=%d used=%d graph-replay=18 max_error=%g %s\n",
        ggml_type_name(type), tokens, used, max_error, ok ? "PASS" : "FAIL");
    for (int j = 0; j < 3; ++j) {
        CUDA_CHECK(cudaGraphExecDestroy(exec[j]));
        CUDA_CHECK(cudaGraphDestroy(captured[j]));
    }
    ggml_moe_cache.session_destroy(session);
    ggml_backend_buffer_free(rb);
    ggml_backend_buffer_free(gb);
    ggml_backend_buffer_free(wb);
    ggml_free(rc);
    ggml_free(gc);
    ggml_free(wc);
    return ok;
}

int main() {
    set_env("GGML_CUDA_MOE_CPU_SHARE", "0");
    set_env("GGML_CUDA_MOE_ROUTE", "1");
    set_env("GGML_CUDA_MOE_CACHE", "1");
    set_env("GGML_CUDA_MOE_CACHE_MODE", "on");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", "1");
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    auto * cpu = ggml_backend_cpu_init();
    auto * gpu = ggml_backend_cuda_init(0);
    GGML_ASSERT(cpu && gpu);
    bool ok = true;
    int tested = 0;
    for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0}) {
        // The share intentionally refuses multi-row-only CPU dot kernels.
        if (ggml_moe_cache_cpu_traits(type)->nrows != 1) {
            printf("SKIP: %s CPU dot kernel is not eligible for sharing\n", ggml_type_name(type));
            continue;
        }
        ++tested;
        for (int used : {4, 6, 8}) {
            for (int tokens : {1, 3, 8}) ok = check(gpu, cpu, type, tokens, used) && ok;
        }
    }
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return !tested ? 77 : ok ? 0 : 1;
}
