#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model.h"

#include "ggml.h"
#include "gguf.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static void write_adapter(
        const char * path,
        const llama_model * model,
        const char * tensor_name,
        uint32_t seed) {
    const ggml_tensor * base = model->get_tensor(tensor_name);
    GGML_ASSERT(base != nullptr);
    GGML_ASSERT(ggml_n_dims(base) == 2);

    gguf_context_ptr gguf(gguf_init_empty());
    gguf_set_val_str(gguf.get(), "general.type", "adapter");
    gguf_set_val_str(gguf.get(), "general.architecture", llm_arch_name(model->arch));
    gguf_set_val_str(gguf.get(), "adapter.type", "lora");
    gguf_set_val_f32(gguf.get(), "adapter.lora.alpha", 4.0f);

    const int64_t rank = 4;
    const size_t data_size = size_t(base->ne[0] + base->ne[1])*rank*sizeof(float);
    ggml_init_params params = {
        /*.mem_size   =*/ data_size + 2*ggml_tensor_overhead() + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, base->ne[0], rank);
    ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, rank, base->ne[1]);
    ggml_set_name(a, (std::string(tensor_name) + ".lora_a").c_str());
    ggml_set_name(b, (std::string(tensor_name) + ".lora_b").c_str());

    auto fill = [&](ggml_tensor * tensor) {
        float * data = static_cast<float *>(tensor->data);
        for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
            seed = 1664525u*seed + 1013904223u;
            data[i] = (float(int32_t(seed >> 8) % 2001) - 1000.0f)*1.0e-5f;
        }
    };
    fill(a);
    fill(b);

    gguf_add_tensor(gguf.get(), a);
    gguf_add_tensor(gguf.get(), b);
    gguf_write_to_file(gguf.get(), path, false);
}

static llama_context_ptr make_context(llama_model * model) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 16;
    params.n_batch = 4;
    params.n_ubatch = 4;
    params.n_threads = 1;
    params.n_threads_batch = 1;

    llama_context_ptr ctx(llama_init_from_model(model, params));
    GGML_ASSERT(ctx != nullptr);
    return ctx;
}

static std::vector<float> decode_logits(llama_model * model, llama_context * ctx) {
    std::array<llama_token, 4> tokens = { 1, 2, 3, 4 };
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    GGML_ASSERT(llama_decode(ctx, batch) == 0);

    const float * logits = llama_get_logits_ith(ctx, -1);
    GGML_ASSERT(logits != nullptr);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    return std::vector<float>(logits, logits + n_vocab);
}

static std::vector<float> get_logits(
        llama_model * model,
        const std::vector<llama_adapter_lora *> & adapters,
        const std::vector<float> & scales) {
    llama_context_ptr ctx = make_context(model);
    auto adapters_copy = adapters;
    auto scales_copy = scales;
    GGML_ASSERT(llama_set_adapters_lora(
            ctx.get(),
            adapters_copy.data(),
            adapters_copy.size(),
            scales_copy.data()) == 0);

    return decode_logits(model, ctx.get());
}

static void check_single_adapter_golden(const std::vector<float> & logits) {
    // Captured on the pre-I6 implementation from the generated qwen35-dense fixture (seed 0).
    // A small tolerance makes the golden portable across CPU SIMD variants while still catching
    // graph-scale/order changes; the two-adapter check below is deliberately bitwise.
    static const std::array<float, 8> expected = {
         0.000132657879f,
        -0.00244316482f,
         0.00144490902f,
         0.0000787233657f,
         0.0000348991125f,
        -0.000829864352f,
         0.000591345190f,
        -0.000452886423f,
    };
    static const std::array<size_t, 8> indices = {
        0, 1, 2, 3, 16, 31, 63, 127,
    };

    for (size_t i = 0; i < indices.size(); ++i) {
        if (std::abs(logits[indices[i]] - expected[i]) > 2.0e-7f) {
            fprintf(stderr, "single-adapter golden mismatch at logit %zu: expected %.9g, got %.9g\n",
                    indices[i], expected[i], logits[indices[i]]);
            for (size_t j = 0; j < indices.size(); ++j) {
                fprintf(stderr, "golden[%zu] = %.9g\n", indices[j], logits[indices[j]]);
            }
            GGML_ABORT("single-adapter output regression");
        }
    }
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL ADAPTER_DIR\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model_ptr model(llama_model_load_from_file(argv[1], model_params));
    GGML_ASSERT(model != nullptr);

    const std::string path_a = std::string(argv[2]) + "/lora-determinism-a.gguf";
    const std::string path_b = std::string(argv[2]) + "/lora-determinism-b.gguf";
    write_adapter(path_a.c_str(), model.get(), "output.weight", 0x12345678u);
    write_adapter(path_b.c_str(), model.get(), "output.weight", 0x87654321u);

    llama_adapter_lora * adapter_a = llama_adapter_lora_init(model.get(), path_a.c_str());
    llama_adapter_lora * adapter_b = llama_adapter_lora_init(model.get(), path_b.c_str());
    llama_adapter_lora * adapter_a_copy = llama_adapter_lora_init(model.get(), path_a.c_str());
    GGML_ASSERT(adapter_a != nullptr);
    GGML_ASSERT(adapter_b != nullptr);
    GGML_ASSERT(adapter_a_copy != nullptr);

    std::array<uint8_t, 32> digest_a;
    std::array<uint8_t, 32> digest_b;
    std::array<uint8_t, 32> digest_a_copy;
    llama_adapter_meta_digest(adapter_a, digest_a.data());
    llama_adapter_meta_digest(adapter_b, digest_b.data());
    llama_adapter_meta_digest(adapter_a_copy, digest_a_copy.data());
    static const std::array<uint8_t, 32> expected_digest_a = {
        0x77, 0x8e, 0xf4, 0xea, 0x16, 0x86, 0x48, 0x3c,
        0xcf, 0xf3, 0x4c, 0x77, 0xb0, 0x71, 0xad, 0xb0,
        0x2e, 0x75, 0x31, 0x6c, 0x4d, 0xd9, 0xbd, 0x62,
        0x5f, 0x87, 0x29, 0x9f, 0xa1, 0xdb, 0x68, 0xd4,
    };
    GGML_ASSERT(digest_a == expected_digest_a);
    GGML_ASSERT(digest_a == digest_a_copy);
    GGML_ASSERT(digest_a != digest_b);

    const std::vector<float> logits_ab = get_logits(model.get(), { adapter_a, adapter_b }, { 0.75f, -0.5f });
    const std::vector<float> logits_ba = get_logits(model.get(), { adapter_b, adapter_a }, { -0.5f, 0.75f });
    GGML_ASSERT(logits_ab.size() == logits_ba.size());
    GGML_ASSERT(memcmp(logits_ab.data(), logits_ba.data(), logits_ab.size()*sizeof(float)) == 0);

    const std::vector<float> logits_a = get_logits(model.get(), { adapter_a }, { 1.0f });
    check_single_adapter_golden(logits_a);
    const std::vector<float> logits_a_large = get_logits(model.get(), { adapter_a }, { 100.0f });
    const std::vector<float> logits_aa = get_logits(model.get(), { adapter_a, adapter_a_copy }, { 100.0f, 100.0f });
    GGML_ASSERT(logits_a_large.size() == logits_aa.size());
    GGML_ASSERT(memcmp(logits_a_large.data(), logits_aa.data(), logits_a_large.size()*sizeof(float)) != 0);

    llama_context_ptr ctx_reject = make_context(model.get());
    {
        llama_adapter_lora * adapters[] = { adapter_a };
        float scales[] = { 1.0f };
        GGML_ASSERT(llama_set_adapters_lora(ctx_reject.get(), adapters, 1, scales) == 0);
    }
    for (float invalid_scale : {
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity() }) {
        llama_adapter_lora * adapters[] = { adapter_b };
        GGML_ASSERT(llama_set_adapters_lora(ctx_reject.get(), adapters, 1, &invalid_scale) == -1);
    }
    const std::vector<float> logits_after_reject = decode_logits(model.get(), ctx_reject.get());
    GGML_ASSERT(logits_a.size() == logits_after_reject.size());
    GGML_ASSERT(memcmp(logits_a.data(), logits_after_reject.data(), logits_a.size()*sizeof(float)) == 0);

    ctx_reject.reset();
    llama_adapter_lora_free(adapter_a_copy);
    llama_adapter_lora_free(adapter_b);
    llama_adapter_lora_free(adapter_a);
    model.reset();
    llama_backend_free();
    return 0;
}
