#include "../src/llama-model.h"

#include <cstdio>

// Exercise the production placement callback without allocating device storage.
struct split_model : llama_model_base {
    explicit split_model(const llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override { return nullptr; }
};

int main() {
    std::vector<float> fractions(llama_max_devices(), 1.0f);
    auto params = llama_model_default_params();
    params.tensor_split = fractions.data();
    for (const auto arch : {LLM_ARCH_GEMMA4, LLM_ARCH_MIMO2, LLM_ARCH_QWEN35}) {
        for (const bool unequal : {false, true}) {
            for (const bool gated : {false, true}) {
                if (gated && arch != LLM_ARCH_QWEN35) { continue; }
                split_model model(params);
                model.arch = arch;
                model.hparams = {};
                auto & hp = model.hparams;
                hp.n_layer_all = 1;
                hp.n_embd = 64; // Deliberately not the Q projection's width.
                hp.n_head_arr[0] = 8;
                hp.n_head_kv_arr[0] = 4;
                hp.n_embd_head_k_full = 128;
                hp.n_embd_head_v_full = unequal ? 64 : 128;
                const int64_t q = (gated ? 2 : 1) * hp.n_head(0) * hp.n_embd_head_k(0);
                const int64_t k = hp.n_embd_k_gqa(0), v = hp.n_embd_v_gqa(0);
                ggml_context_ptr ctx(ggml_init({16*ggml_tensor_overhead(), nullptr, true}));
                auto * out = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8*hp.n_embd_head_v(0), hp.n_embd);
                ggml_set_name(out, "blk.0.attn_output.weight");
                model.tensors_by_name.emplace_back(out->name, out);
                auto * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hp.n_embd, q+k+v);
                auto * bias = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, q+k+v);
                auto * scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, q+k+v);
                ggml_set_name(weight, "blk.0.attn_qkv.weight");
                ggml_set_name(bias, "blk.0.attn_qkv.bias");
                ggml_set_name(scale, "blk.0.attn_qkv.scale");
                llama_meta_device_get_split_state_userdata ud{2, &model};
                for (auto * tensor : {weight, bias, scale}) {
                    const auto ss = llama_meta_device_get_split_state(tensor, &ud);
                    GGML_ASSERT(ss.axis == (tensor == weight ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0));
                    GGML_ASSERT(ss.n_segments == (unequal ? 3u : 2u));
                    for (int rank = 0; rank < 2; ++rank) {
                        GGML_ASSERT(ss.ne[rank] == q/2 && ss.nr[0] == 1);
                        GGML_ASSERT(ss.ne[2+rank] == k/2 && ss.nr[1] == (unequal ? 1 : 2));
                        if (unequal) { GGML_ASSERT(ss.ne[4+rank] == v/2 && ss.nr[2] == 1); }
                    }
                }
            }
        }
    }
    std::puts("QKV split: unequal heads, gated/ungated Q, bias and channel scales passed");
}
