#include "../src/models/models.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "llama-adapter.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <vector>

// Exercise both sides of a tied folded head. Checking the embedding separately
// prevents two omitted (mutually cancelling) transforms from passing the test.
static void check_graph(bool folded, bool signed_rows) {
    llama_model_dflash target(llama_model_default_params());
    ggml_context_ptr weights(ggml_init({3*ggml_tensor_overhead(), nullptr, true}));
    target.tok_embd = target.output = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, 4, 3);
    auto * rot = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, 2, 2);
    auto * signs = ggml_new_tensor_1d(weights.get(), GGML_TYPE_F32, 4);
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    auto * weight_buffer = ggml_backend_alloc_ctx_tensors(weights.get(), backend.get());
    GGML_ASSERT(weight_buffer);
    const float table[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const float scale = 1.0f / std::sqrt(2.0f);
    const float rotation[] = {scale, scale, scale, -scale};
    const float sign_values[] = {-1, 1, -1, 1};
    ggml_backend_tensor_set(target.tok_embd, table, 0, sizeof(table));
    ggml_backend_tensor_set(rot, rotation, 0, sizeof(rotation));
    ggml_backend_tensor_set(signs, sign_values, 0, sizeof(sign_values));
    if (folded) {
        const llama_hadamard_transform transform{rot, signed_rows ? signs : nullptr};
        target.hadamard_rotations.emplace(target.output, transform);
        target.hadamard_inverses.emplace(target.tok_embd, transform);
    }
    target.adopt_buffer(std::move(weights), ggml_backend_buffer_ptr(weight_buffer));
    llama_model_dflash draft(llama_model_default_params());
    llama_model_share_tensors(&draft, &target);

    llm_graph_result result(128);
    llama_adapter_loras_ordered loras;
    llm_graph_params params{};
    params.hparams.n_layer_all = 1;
    params.hparams.n_embd = 4;
    params.hparams.n_head_arr[0] = params.hparams.n_head_kv_arr[0] = 1;
    params.hadamard_rotations = &draft.hadamard_rotations;
    params.hadamard_inverses = &draft.hadamard_inverses;
    params.loras = &loras;
    params.res = &result;
    llm_graph_context graph(params);
    auto * ids = ggml_new_tensor_1d(graph.ctx0, GGML_TYPE_I32, 3);
    // Like DFlash's one-graph cycle, skip an injection-prefix token.
    auto * noise = ggml_view_1d(graph.ctx0, ids, 2, sizeof(int32_t));
    auto * embd = graph.build_get_rows_embd(draft.tok_embd, noise);
    auto * out = graph.build_lora_mm(draft.output, embd);
    ggml_build_forward_expand(graph.gf, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(graph.ctx0, backend.get()));
    GGML_ASSERT(buffer);
    const int32_t token_ids[] = {0, 2, 1};
    ggml_backend_tensor_set(ids, token_ids, 0, sizeof(token_ids));
    GGML_ASSERT(ggml_backend_graph_compute(backend.get(), graph.gf) == GGML_STATUS_SUCCESS);
    float embeddings[8], logits[6];
    ggml_backend_tensor_get(embd, embeddings, 0, sizeof(embeddings));
    ggml_backend_tensor_get(out, logits, 0, sizeof(logits));
    for (int n = 0; n < 2; ++n) {
        const auto * row = table + 4 * token_ids[n + 1];
        for (int k = 0; k < 4; ++k) {
            float expected = row[k];
            if (folded) {
                expected = scale * (row[k & ~1] + (k % 2 ? -1 : 1) * row[k | 1]);
                if (signed_rows) expected *= sign_values[k];
            }
            GGML_ASSERT(std::abs(embeddings[4*n + k] - expected) < 1e-4f);
        }
        for (int m = 0; m < 3; ++m) {
            float expected = 0;
            for (int k = 0; k < 4; ++k) expected += row[k] * table[4*m + k];
            GGML_ASSERT(std::abs(logits[3*n + m] - expected) < 1e-3f);
        }
    }
}

static void check(ggml_backend_buffer_type_t buft, bool own_device, bool no_alloc, bool tied,
                  bool host_weights = false, bool folded = false) {
    llama_model_dflash target(llama_model_default_params());
    ggml_context_ptr ctx(ggml_init({4*ggml_tensor_overhead(), nullptr, true}));
    target.tok_embd = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 128, 128);
    target.output = tied ? target.tok_embd : ggml_new_tensor_2d(ctx.get(), ggml_exl3_type(4, 2), 128, 128);
    target.output_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 128);
    target.output_in_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 128);
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
    GGML_ASSERT(buffer);
    uint8_t value = 37;
    for (auto * t : { target.tok_embd, target.output, target.output_s, target.output_in_s }) {
        std::vector<uint8_t> data(ggml_nbytes(t), value++);
        ggml_backend_tensor_set(t, data.data(), 0, data.size());
    }
    target.adopt_buffer(std::move(ctx), ggml_backend_buffer_ptr(buffer));
    if (host_weights) {
        // Only the auxiliaries are foreign: sharing the weight alone must not
        // take the early return and leave the transforms on the wrong device.
        ggml_context_ptr host(ggml_init({2*ggml_tensor_overhead(), nullptr, true}));
        target.tok_embd = ggml_dup_tensor(host.get(), target.tok_embd);
        target.output = tied ? target.tok_embd : ggml_dup_tensor(host.get(), target.output);
        auto * host_buffer = ggml_backend_alloc_ctx_tensors_from_buft(host.get(), ggml_backend_cpu_buffer_type());
        GGML_ASSERT(host_buffer);
        ggml_backend_buffer_clear(host_buffer, 42);
        target.adopt_buffer(std::move(host), ggml_backend_buffer_ptr(host_buffer));
    }

    if (folded) {
        ggml_context_ptr transforms(ggml_init({2*ggml_tensor_overhead(), nullptr, true}));
        auto * rot = ggml_new_tensor_2d(transforms.get(), GGML_TYPE_F32, 128, 128);
        auto * signs = tied ? nullptr : ggml_new_tensor_1d(transforms.get(), GGML_TYPE_F32, 128);
        auto * transform_buffer = ggml_backend_alloc_ctx_tensors_from_buft(transforms.get(), buft);
        GGML_ASSERT(transform_buffer);
        ggml_backend_buffer_clear(transform_buffer, 53);
        target.hadamard_inverses.emplace(target.tok_embd, llama_hadamard_transform{rot, signs});
        target.hadamard_rotations.emplace(target.output, llama_hadamard_transform{rot, signs, 16, 4, 2});
        target.adopt_buffer(std::move(transforms), ggml_backend_buffer_ptr(transform_buffer));
    }

    llama_model_dflash draft(llama_model_default_params());
    draft.hparams.no_alloc = no_alloc;
    if (own_device) {
        draft.devices.push_back({false, ggml_backend_buft_get_device(buft)});
    }
    llama_model_share_tensors(&draft, &target);
    std::vector<const ggml_tensor *> sources = {target.tok_embd, target.output, target.output_s, target.output_in_s};
    std::vector<const ggml_tensor *> dests = {draft.tok_embd, draft.output, draft.output_s, draft.output_in_s};
    if (folded) {
        GGML_ASSERT(draft.hadamard_inverses.size() == 1 && draft.hadamard_rotations.size() == 1);
        const auto & inv = draft.hadamard_inverses.at(draft.tok_embd);
        const auto & rot = draft.hadamard_rotations.at(draft.output);
        const auto & expected = target.hadamard_rotations.at(target.output);
        GGML_ASSERT(rot.perm_hd == expected.perm_hd && rot.perm_nk == expected.perm_nk && rot.perm_rep == expected.perm_rep);
        GGML_ASSERT(inv.perm_rep == 0);
        // The forward and inverse share auxiliaries, including after copying.
        GGML_ASSERT(inv.rot == rot.rot && inv.signs == rot.signs);
        sources.push_back(expected.rot);
        dests.push_back(rot.rot);
        if (expected.signs) {
            sources.push_back(expected.signs);
            dests.push_back(rot.signs);
        } else {
            GGML_ASSERT(rot.signs == nullptr);
        }
    } else {
        GGML_ASSERT(draft.hadamard_inverses.empty() && draft.hadamard_rotations.empty());
    }
    for (size_t i = 0; i < sources.size(); ++i) {
        const bool copied = !own_device && !ggml_backend_buffer_is_host(sources[i]->buffer);
        GGML_ASSERT(dests[i]);
        GGML_ASSERT((dests[i] != sources[i]) == copied);
        GGML_ASSERT(dests[i]->type == sources[i]->type);
        GGML_ASSERT(ggml_are_same_shape(dests[i], sources[i]));
        if (copied && no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_size(dests[i]->buffer) == 0);
        } else {
            std::vector<uint8_t> data(ggml_nbytes(dests[i]));
            std::vector<uint8_t> expected(ggml_nbytes(sources[i]));
            ggml_backend_tensor_get(dests[i], data.data(), 0, data.size());
            ggml_backend_tensor_get(sources[i], expected.data(), 0, expected.size());
            GGML_ASSERT(data == expected);
        }
    }
    GGML_ASSERT((draft.output == draft.tok_embd) == tied);
    // A materialized head or fully borrowed same-device bundle is stable on
    // repeat setup. Auxiliary-only copies are not covered by this assertion.
    if (draft.output != target.output || own_device) {
        llama_model_share_tensors(&draft, &target);
        GGML_ASSERT(draft.output == dests[1] && draft.output_s == dests[2] && draft.output_in_s == dests[3]);
    }

    // Sharing just the embedding must preserve a self-contained output bundle.
    llama_model_dflash independent(llama_model_default_params());
    independent.output = target.output_s;
    independent.output_s = target.output_in_s;
    independent.output_in_s = target.output;
    if (folded) {
        independent.hadamard_rotations.emplace(independent.output, target.hadamard_rotations.at(target.output));
    }
    llama_model_share_tensors(&independent, &target);
    GGML_ASSERT(independent.output == target.output_s);
    GGML_ASSERT(independent.output_s == target.output_in_s);
    GGML_ASSERT(independent.output_in_s == target.output);
    if (folded) {
        // A tied source embedding also carries the source head's forward map.
        GGML_ASSERT(independent.hadamard_rotations.size() == (tied ? 2 : 1));
        GGML_ASSERT(independent.hadamard_rotations.at(independent.output).rot ==
                    target.hadamard_rotations.at(target.output).rot);
        GGML_ASSERT(independent.hadamard_inverses.count(independent.tok_embd) == 1);
    }

    // A plain head must not inherit stale scale fields from the draft shell.
    target.output_s = nullptr;
    target.output_in_s = nullptr;
    llama_model_dflash plain(llama_model_default_params());
    plain.output_s = independent.output_s;
    plain.output_in_s = independent.output_in_s;
    llama_model_share_tensors(&plain, &target);
    GGML_ASSERT(plain.output_s == nullptr && plain.output_in_s == nullptr);
}

int main() {
    ggml_backend_load_all();
    check_graph(false, false);
    check_graph(true, false);
    check_graph(true, true);
    for (bool folded : {false, true}) {
        check(ggml_backend_cpu_buffer_type(), false, false, false, false, folded);
        check(ggml_backend_cpu_buffer_type(), false, true, true, false, folded);
    }
    auto * gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu) {
        for (bool own_device : {false, true}) {
            for (bool no_alloc : {false, true}) {
                for (bool tied : {false, true}) {
                    for (bool folded : {false, true}) {
                        check(ggml_backend_dev_buffer_type(gpu), own_device, no_alloc, tied, false, folded);
                        check(ggml_backend_dev_buffer_type(gpu), own_device, no_alloc, tied, true, folded);
                    }
                }
            }
        }
    } else {
        puts("GPU placement cases skipped (no GPU backend)");
    }
    puts("draft shared tensor tests passed");
}
