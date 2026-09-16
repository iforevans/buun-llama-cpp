#include "../src/llama-model.h"
#include "../src/llama-model-loader.h"
#include "../src/llama-model-saver.h"

#include <array>
#include <cstdio>
#include <stdexcept>

// Exercise the real generic loader and saver without tensors or a model download.
struct metadata_model : llama_model_base {
    metadata_model() : llama_model_base(llama_model_default_params()) { arch = LLM_ARCH_LLAMA; }
    void load_arch_hparams(llama_model_loader & ml) override {
        ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    }
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override { return nullptr; }
};

static void load(metadata_model & model, gguf_context * meta) {
    std::vector<std::string> splits;
    llama_model_loader ml(meta, nullptr, nullptr, nullptr, "", splits, nullptr,
                          LLAMA_LOAD_MODE_NONE, false, true, true, nullptr, nullptr);
    model.load_hparams(ml);
}

template<typename F> static void rejects(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::runtime_error &) { rejected = true; }
    GGML_ASSERT(rejected);
}

int main() {
    const LLM_KV kv(LLM_ARCH_LLAMA);
    gguf_context * meta = gguf_init_empty();
    gguf_set_val_str(meta, "general.architecture", "llama");
    gguf_set_val_u32(meta, kv(LLM_KV_BLOCK_COUNT).c_str(), 4);
    gguf_set_val_u32(meta, kv(LLM_KV_NEXTN_PREDICT_LAYERS).c_str(), 1);
    gguf_set_val_u32(meta, kv(LLM_KV_CONTEXT_LENGTH).c_str(), 128);
    gguf_set_val_u32(meta, kv(LLM_KV_EMBEDDING_LENGTH).c_str(), 32);
    gguf_set_val_u32(meta, kv(LLM_KV_EXPERT_COUNT).c_str(), 8);
    const std::string used_key = kv(LLM_KV_EXPERT_USED_COUNT);
    const std::string width_key = kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH);

    for (const bool heterogeneous : {false, true}) {
        const std::array<uint32_t, 4> used = heterogeneous ? std::array<uint32_t, 4>{2, 0, 4, 6}
                                                          : std::array<uint32_t, 4>{2, 2, 2, 2};
        const std::array<uint32_t, 4> width = heterogeneous ? std::array<uint32_t, 4>{64, 0, 128, 256}
                                                           : std::array<uint32_t, 4>{64, 64, 64, 64};
        if (heterogeneous) {
            gguf_set_arr_data(meta, used_key.c_str(), GGUF_TYPE_UINT32, used.data(), used.size());
            gguf_set_arr_data(meta, width_key.c_str(), GGUF_TYPE_UINT32, width.data(), width.size());
        } else {
            gguf_set_val_u32(meta, used_key.c_str(), used[0]);
            gguf_set_val_u32(meta, width_key.c_str(), width[0]);
        }
        metadata_model model;
        load(model, meta);
        GGML_ASSERT(model.hparams.n_layer() == 3);
        GGML_ASSERT(model.hparams.n_expert_used_max() == (heterogeneous ? 6 : 2));
        llama_model_saver saver(&model);
        saver.add_kv_from_model();
        metadata_model restored;
        load(restored, saver.gguf_ctx);
        for (uint32_t il = 0; il < 4; ++il) {
            GGML_ASSERT(model.hparams.n_expert_used(il) == used[il]);
            GGML_ASSERT(model.hparams.n_ff_exp(il) == width[il]);
            GGML_ASSERT(restored.hparams.n_expert_used(il) == used[il]);
            GGML_ASSERT(restored.hparams.n_ff_exp(il) == width[il]);
        }
        GGML_ASSERT(model.hparams.n_expert_used_arr[4] == 0);
        GGML_ASSERT(model.hparams.n_ff_exp_arr[4] == 0);
    }

    // A capacity-bounded get_arr may read a prefix; per-layer get_key_or_arr
    // must instead require exactly block_count entries, including the MTP tail.
    const uint32_t values[] = {2, 0, 4, 6, 8};
    for (const size_t length : {size_t(0), size_t(3), size_t(5)}) {
        gguf_set_arr_data(meta, used_key.c_str(), GGUF_TYPE_UINT32, values, length);
        rejects([&] { metadata_model model; load(model, meta); });
    }
    const float wrong_type[] = {2, 0, 4, 6};
    gguf_set_arr_data(meta, used_key.c_str(), GGUF_TYPE_FLOAT32, wrong_type, 4);
    rejects([&] { metadata_model model; load(model, meta); });
    gguf_free(meta);
    std::puts("MoE metadata: scalar broadcast, heterogeneous/MTP roundtrip and four rejection cases passed");
    return 0;
}
