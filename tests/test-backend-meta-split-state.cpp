// Split-state resolution must not use stack proportional to graph depth, even
// after a late descriptor mismatch invalidates all of the memoized states.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static ggml_backend_meta_split_state split_state(const ggml_tensor * tensor, void * user_data) {
    ++*static_cast<size_t *>(user_data);
    if (tensor->ne[1] == 4) {
        return {GGML_BACKEND_SPLIT_AXIS_0, {2, 2}, {1}, 1};
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

static void allocate(ggml_backend_buffer_t buffer, ggml_tensor * tensor, size_t index, size_t stride) {
    const auto address = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer)) + index * stride);
    GGML_ASSERT(ggml_backend_tensor_alloc(buffer, tensor, address) == GGML_STATUS_SUCCESS);
}

static void test_chain(ggml_backend_buffer_type_t buft, int depth, int repeats) {
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t stride = ((16 * sizeof(float) + alignment - 1) / alignment) * alignment;
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * size_t(depth + 8), nullptr, true}));
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(buft, stride * (depth + 4)));
    GGML_ASSERT(buffer);
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    std::vector<ggml_tensor *> nodes{ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16)};
    for (int i = 0; i < depth; ++i) {
        nodes.push_back(ggml_add(ctx.get(), nodes.back(), nodes.front()));
    }
    const auto allocation_start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < nodes.size(); ++i) {
        ggml_format_name(nodes[i], "node-%zu", i);
        allocate(buffer.get(), nodes[i], i, stride);
    }
    printf("depth=%d us/allocation=%.3f\n", depth,
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - allocation_start).count() / nodes.size());
    float value = 7;
    ggml_backend_tensor_set(nodes.back(), &value, 0, sizeof(value));
    for (bool invalidate : {false, true}) {
        const int count = invalidate ? repeats : repeats * 1000;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < count; ++i) {
            if (invalidate) {
                ggml_format_name(nodes.back(), "renamed-%d", i);
            }
            value = 0;
            ggml_backend_tensor_get(nodes.back(), &value, 0, sizeof(value));
            GGML_ASSERT(value == 7);
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        printf("depth=%d invalidate=%d repeats=%d us/lookup=%.3f\n", depth, invalidate, count, us / count);
    }

    // A later sibling can invalidate the map after an earlier source was read.
    // The resolver must own its source values, not references into that map.
    auto * sibling = ggml_add(ctx.get(), nodes.front(), nodes.front());
    allocate(buffer.get(), sibling, depth + 1, stride);
    auto * joined = ggml_add(ctx.get(), nodes.back(), sibling);
    allocate(buffer.get(), joined, depth + 2, stride);
    for (size_t j = 0; j < 2; ++j) {
        ggml_backend_tensor_set(ggml_backend_meta_buffer_simple_tensor(joined, j), &value, 0, sizeof(value));
    }
    ggml_set_name(sibling, "changed-sibling");
    value = 0;
    ggml_backend_tensor_get(joined, &value, 0, sizeof(value));
    GGML_ASSERT(value == 7);
}

static void test_sync_modes(ggml_backend_buffer_type_t buft, size_t & callback_calls) {
    // Sources live in a different meta buffer. Sharded matmul output is
    // MIRRORED with assume_sync=true (allocation) but PARTIAL with false (I/O).
    ggml_context_ptr weights(ggml_init({ggml_tensor_overhead() * 4, nullptr, true}));
    auto * a = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, 4, 4);
    auto * b = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, 4, 4);
    auto * view = ggml_view_2d(weights.get(), a, 4, 2, a->nb[1], 0);
    ggml_backend_buffer_ptr weights_buffer(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), buft));
    GGML_ASSERT(weights_buffer);
    ggml_backend_buffer_set_usage(weights_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Renaming a view must invalidate its same-buffer source too. Just erasing
    // the stale view would preserve these bytes but skip the source callback.
    const std::vector<float> values(16, 12);
    ggml_backend_tensor_set(a, values.data(), 0, values.size() * sizeof(float));
    float restored = 0;
    ggml_backend_tensor_get(view, &restored, 0, sizeof(restored));
    const size_t calls_before = callback_calls;
    ggml_set_name(view, "renamed-view");
    ggml_backend_tensor_get(view, &restored, 0, sizeof(restored));
    GGML_ASSERT(restored == 12 && callback_calls == calls_before + 1);

    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * 4, nullptr, true}));
    auto * product = ggml_mul_mat(ctx.get(), view, b);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t stride = (ggml_backend_buft_get_alloc_size(buft, product) + alignment - 1) / alignment * alignment;
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(buft, 2 * stride));
    GGML_ASSERT(buffer);
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    allocate(buffer.get(), product, 0, 0);
    const float value = 12;
    for (int i = 0; i < 3; ++i) {
        ggml_format_name(product, "partial-%d", i);
        if (i == 1) {
            // Rebuild the weights cache while a compute-buffer frame waits.
            ggml_set_name(view, "cross-buffer-miss");
        }
        ggml_backend_tensor_set(product, &value, 0, sizeof(value));
        for (size_t j = 0; j < 2; ++j) {
            float shard = 0;
            ggml_backend_tensor_get(ggml_backend_meta_buffer_simple_tensor(product, j), &shard, 0, sizeof(shard));
            GGML_ASSERT(shard == value / 2);
        }
    }
    auto * consumer = ggml_add(ctx.get(), product, product);
    allocate(buffer.get(), consumer, 1, stride);
    ggml_backend_tensor_set(consumer, &value, 0, sizeof(value));
    for (size_t j = 0; j < 2; ++j) {
        float shard = 0;
        ggml_backend_tensor_get(ggml_backend_meta_buffer_simple_tensor(consumer, j), &shard, 0, sizeof(shard));
        GGML_ASSERT(shard == value);
    }
    puts("cross-buffer sources, views and synchronization modes: PASS");
}

int main(int argc, char ** argv) {
    const int depth = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 3;
    GGML_ASSERT(depth > 0 && repeats > 0);
    ggml_backend_load_all();
    auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        fprintf(stderr, "SKIP: requires CPU backend\n");
        return 77;
    }
    // Two independent child buffers; no GPU hardware is needed for this test.
    ggml_backend_dev_t children[] = {cpu, cpu};
    size_t callback_calls = 0;
    auto dev = ggml_backend_meta_device(children, 2, split_state, &callback_calls);
    auto buft = ggml_backend_dev_buffer_type(dev);
    test_chain(buft, depth, repeats);
    test_sync_modes(buft, callback_calls);
    puts("PASS");
}
