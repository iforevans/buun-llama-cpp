#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

static void check(ggml_backend_t backend, int tokens, int copies, int channels, bool dependency, bool inactive = false) {
    constexpr int prefix = 3, sequences = 2, capacity = 4, head = 1;
    const int width = prefix + tokens, row = prefix * channels;
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, channels, sequences);
    auto * states = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row, capacity * copies);
    auto * graph = ggml_new_graph_custom(ctx, 1024, false);
    std::vector<float> data(ggml_nelements(input)), expected(ggml_nelements(states), -123.0f);
    for (size_t i = 0; i < data.size(); ++i) data[i] = float(int(i % 1019) - 503) / 16;
    const uint32_t patterns[] = {0x7fc01234u, 0xff800000u, 0x80000000u};
    for (size_t i = 0; i < 3; ++i) std::memcpy(&data[i], &patterns[i], sizeof(float));
    for (int t = 1; t <= copies; ++t) {
        const int offset = std::max(0, tokens - copies + t), slot = copies - t;
        auto * src = ggml_view_3d(ctx, input, prefix, channels, sequences,
                input->nb[1], input->nb[2], offset * sizeof(float));
        auto * dst = ggml_view_2d(ctx, states, row, sequences, states->nb[1],
                (slot * capacity + head) * row * sizeof(float));
        ggml_build_forward_expand(graph, ggml_cpy(ctx, src, dst));
        for (int s = 0; s < sequences; ++s)
            for (int c = 0; c < channels; ++c)
                for (int p = 0; p < prefix; ++p)
                    expected[(slot * capacity + head + s) * row + c * prefix + p] =
                        data[(s * channels + c) * width + offset + p];
    }
    if (dependency) {
        // Reads a preceding copy's destination; it cannot join that batch.
        auto * src = ggml_view_1d(ctx, states, row, head * row * sizeof(float));
        auto * dst = ggml_view_1d(ctx, states, row, 0);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, src, dst));
        std::copy_n(expected.begin() + head * row, row, expected.begin());
        // An identical source/destination must fall back without changing bytes.
        ggml_build_forward_expand(graph, ggml_cpy(ctx, dst, dst));
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    if (inactive) {
        // The backend skips non-compute nodes, which can be unallocated. A
        // look-ahead matcher must respect that boundary before testing ranges.
        auto * unused_src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        auto * unused_dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        auto * unused_copy = ggml_cpy(ctx, unused_src, unused_dst);
        unused_copy->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
        ggml_graph_add_node(graph, unused_copy);
    }
    ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
    const std::vector<float> initial(expected.size(), -123.0f);
    for (int iteration = 0; iteration < 3; ++iteration) {
        ggml_backend_tensor_set(states, initial.data(), 0, ggml_nbytes(states));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        std::vector<float> actual(expected.size());
        ggml_backend_tensor_get(states, actual.data(), 0, ggml_nbytes(states));
        for (size_t i = 0; i < actual.size(); ++i) {
            if (std::memcmp(&actual[i], &expected[i], sizeof(float))) {
                std::fprintf(stderr, "mismatch tokens=%d copies=%d channels=%d dependency=%d at %zu: %g != %g\n",
                    tokens, copies, channels, dependency, i, actual[i], expected[i]);
                GGML_ABORT("copy batch mismatch");
            }
        }
    }
    std::printf("PASS tokens=%d copies=%d channels=%d dependency=%d\n", tokens, copies, channels, dependency);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

// Exercise distinct four-dimensional layouts, padding, and singleton axes.
// Keep sentinels in the gaps so an incorrect offset cannot silently pass.
static void check_layouts(ggml_backend_t backend, bool singleton) {
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    auto * graph = ggml_new_graph_custom(ctx, 128, false);
    const int64_t src_ne[4] = {3, singleton ? 1 : 5, 7, 2};
    const int64_t dst_ne[4] = {7, 3, 2, singleton ? 1 : 5};
    const size_t src_nb[4] = {4, 20, 120, 1000};
    const size_t dst_nb[4] = {4, 48, 200, 512};
    constexpr int storage = 2048;
    ggml_tensor * input[2], * output[2];
    std::vector<uint32_t> data(storage), expected[2];
    for (int i = 0; i < storage; ++i) data[i] = 0x3f000000u + unsigned(i);
    data[0] = 0x7fc01234u; data[1] = 0x80000000u; data[2] = 0xff800000u;
    auto offset = [](int64_t index, const int64_t * ne, const size_t * nb) {
        size_t result = 0;
        for (int d = 0; d < 4; ++d) {
            result += (index % ne[d]) * nb[d];
            index /= ne[d];
        }
        return result / sizeof(float);
    };
    for (int c = 0; c < 2; ++c) {
        input[c] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, storage);
        output[c] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, storage);
        auto * src = ggml_view_4d(ctx, input[c], src_ne[0], src_ne[1], src_ne[2], src_ne[3],
                                 src_nb[1], src_nb[2], src_nb[3], 0);
        auto * dst = ggml_view_4d(ctx, output[c], dst_ne[0], dst_ne[1], dst_ne[2], dst_ne[3],
                                 dst_nb[1], dst_nb[2], dst_nb[3], 0);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, src, dst));
        expected[c].assign(storage, 0xdeadbeefu);
        for (int64_t i = 0; i < ggml_nelements(src); ++i) {
            expected[c][offset(i, dst_ne, dst_nb)] = data[offset(i, src_ne, src_nb)];
        }
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    const std::vector<uint32_t> initial(storage, 0xdeadbeefu);
    for (int pass = 0; pass < 3; ++pass) {
        for (int c = 0; c < 2; ++c) {
            ggml_backend_tensor_set(input[c], data.data(), 0, ggml_nbytes(input[c]));
            ggml_backend_tensor_set(output[c], initial.data(), 0, ggml_nbytes(output[c]));
        }
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        for (int c = 0; c < 2; ++c) {
            std::vector<uint32_t> actual(storage);
            ggml_backend_tensor_get(output[c], actual.data(), 0, ggml_nbytes(output[c]));
            GGML_ASSERT(actual == expected[c]);
        }
    }
    std::printf("PASS four-dimensional copy layouts singleton=%d\n", singleton);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if (ggml_backend_cuda_get_device_count() == 0) return 77;
    auto * backend = ggml_backend_cuda_init(0);
    if (!backend) return 77;
    if (argc == 2 && std::strcmp(argv[1], "--inactive") == 0) {
        check(backend, 8, 8, 257, false, true);
        ggml_backend_free(backend);
        return 0;
    }
    for (int copies : {1,2,8,16,17}) {
        check(backend, 1, copies, 257, false);
        check(backend, copies, copies, 257, true);
    }
    check(backend, 8, 8, 10240, true);
    check(backend, 8, 8, 257, false, true);
    check_layouts(backend, false);
    check_layouts(backend, true);
    ggml_backend_free(backend);
}
