#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <clocale>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

constexpr double NMSE_THRESHOLD = 1e-5;

enum class test_status {
    PASS,
    FAIL,
    SKIP,
};

static const char * test_status_str(test_status status) {
    switch (status) {
        case test_status::PASS: return "\033[1;32mPASS\033[0m";
        case test_status::FAIL: return "\033[1;31mFAIL\033[0m";
        case test_status::SKIP: return "\033[1;33mSKIP\033[0m";
    }
    return "";
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        const float a_i = a[i];
        const float b_i = b[i];
        if (!std::isfinite(a_i) || !std::isfinite(b_i)) {
            return std::numeric_limits<double>::infinity();
        }
        const double d = double(a_i) - b_i;
        mse_a_b += d * d;
        mse_a_0 += (double) a_i * a_i;
    }

    return mse_a_0 == 0.0 ? (mse_a_b == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : mse_a_b / mse_a_0;
}

struct generation_result {
    llama_tokens tokens;
    std::vector<std::vector<float>> logits;

    bool empty() const { return tokens.empty(); }
};

static bool get_current_logits(llama_context * ctx, std::vector<float> & out) {
    const auto * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        return false;
    }
    out.assign(logits, logits + n_vocab);
    return true;
}

struct llama_batch_ptr {
    llama_batch batch;

    llama_batch_ptr(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
        : batch{llama_batch_init(n_tokens, embd, n_seq_max)} {}

    ~llama_batch_ptr() { llama_batch_free(batch); }

    llama_batch_ptr(const llama_batch_ptr &) = delete;
    llama_batch_ptr & operator=(const llama_batch_ptr &) = delete;
    llama_batch_ptr(llama_batch_ptr &&) = default;
    llama_batch_ptr & operator=(llama_batch_ptr &&) = default;

    llama_batch & get() { return batch; }
    const llama_batch & get() const { return batch; }
};

static generation_result generate_tokens(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id) {
    generation_result result;
    llama_batch_ptr batch(1, 0, 1);

    for (int i = 0; i < n_predict; i++) {
        std::vector<float> logits;
        if (!get_current_logits(ctx, logits)) {
            LOG_ERR("\n%s: failed to get logits\n", __func__);
            return {};
        }

        auto next_token = llama_sampler_sample(smpl, ctx, -1);

        // LOG_LEVEL_INFO gate: visible in single-model mode, silenced in --models table mode
        LOGV(LOG_LEVEL_INFO, "%d ", next_token);
        result.tokens.push_back(next_token);
        result.logits.push_back(std::move(logits));

        common_batch_clear(batch.get());
        common_batch_add(batch.get(), next_token, n_past, {seq_id}, true);

        if (llama_decode(ctx, batch.get())) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return {};
        }
        n_past++;
    }

    return result;
}

static bool generate_tokens_compare(
        llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id,
        const generation_result & expected) {
    if (expected.tokens.size() != expected.logits.size() || expected.tokens.size() < (size_t) n_predict) {
        LOG_ERR("\n%s: invalid expected generation\n", __func__);
        return false;
    }

    llama_batch_ptr batch(1, 0, 1);

    for (int i = 0; i < n_predict; i++) {
        std::vector<float> logits;
        if (!get_current_logits(ctx, logits)) {
            LOG_ERR("\n%s: failed to get logits\n", __func__);
            return false;
        }
        if (logits.size() != expected.logits[i].size()) {
            LOG_ERR("\n%s: logits size mismatch at step %d: %zu != %zu\n", __func__, i, logits.size(), expected.logits[i].size());
            return false;
        }

        const double nmse_val = nmse(expected.logits[i], logits);
        LOG_TRC("%s: step %d nmse = %.6e\n", __func__, i, nmse_val);
        if (!(nmse_val <= NMSE_THRESHOLD)) {
            LOG_ERR("\n%s: error: NMSE at step %d is %.6e (threshold %.1e)\n", __func__, i, nmse_val, NMSE_THRESHOLD);
            return false;
        }

        const auto next_token = llama_sampler_sample(smpl, ctx, -1);
        const auto expected_token = expected.tokens[i];

        LOGV(LOG_LEVEL_INFO, "%d ", next_token);
        if (next_token != expected_token) {
            LOG_TRC("%s: sampled token %d differs from expected %d, using expected token\n", __func__, next_token, expected_token);
        }

        common_batch_clear(batch.get());
        common_batch_add(batch.get(), expected_token, n_past, {seq_id}, true);

        if (llama_decode(ctx, batch.get())) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return false;
        }
        n_past++;
    }

    return true;
}

// Test 1: baseline
// - decode all but the last token
// - save state to disk
// - decode the last token
// - generate n_predict tokens
static generation_result test_baseline(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto n_past = 0;
    if (!common_prompt_batch_decode(ctx.get(), tokens, (int)tokens.size(), n_past, params.n_batch, params.out_file, true)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return {};
    }

    LOGV(LOG_LEVEL_INFO, "\n=== Test 1: baseline ===\n");

    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return {};
    }

    LOGV(LOG_LEVEL_INFO, "\n");

    return result;
}


// Test 2: sequence removal isolation
// - decode the same prefix into two sequences
// - remove sequence 0
// - verify that sequence 1 remains unchanged
static bool test_seq_rm_isolated(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;

    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\n=== Test 2: sequence removal isolation ===\n");

    const size_t n_tokens = tokens.size() < 128 ? tokens.size() : 128;
    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        llama_batch_ptr batch(n_tokens, 0, 1);
        for (size_t i = 0; i < n_tokens; ++i) {
            common_batch_add(batch.get(), tokens[i], i, { seq_id }, i == n_tokens - 1);
        }

        if (llama_decode(ctx.get(), batch.get())) {
            LOG_ERR("%s: failed to decode prompt for sequence %d\n", __func__, seq_id);
            return false;
        }
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size(ctx.get(), seq_id);
        if (state_size == 0) {
            LOG_ERR("%s: sequence state is empty\n", __func__);
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), state.data(), state.size(), seq_id);
        if (ncopy != state.size()) {
            LOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                    __func__, ncopy, state.size());
            return false;
        }

        return true;
    };

    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, state_before)) {
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return false;
    }

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, state_after)) {
        return false;
    }

    if (state_before != state_after) {
        LOG_ERR("%s: removing sequence 0 changed sequence 1\n", __func__);
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "PASS\n");
    return true;
}


// Test 3: state load
// - create a new context
// - load state from file
// - replay the last prompt token
// - generate n_predict tokens and compare against expected result
static bool test_state_load(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const generation_result & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOGV(LOG_LEVEL_INFO, "\n=== Test 3: state load ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Generate tokens and compare logits against the baseline
    if (!generate_tokens_compare(ctx.get(), smpl.get(), n_past, params.n_predict, 0, expected_result)) {
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS\n");
    return true;
}


// Test 4: seq copy (host)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the CPU path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_host(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const generation_result & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOGV(LOG_LEVEL_INFO, "\n=== Test 4: seq copy (host) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (CPU path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size(ctx.get(), 0));
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), seq_store.data(), seq_store.size(), 0);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data(ctx.get(), seq_store.data(), seq_store.size(), 1);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens and compare logits against the baseline
    if (!generate_tokens_compare(ctx.get(), smpl.get(), n_past, params.n_predict, 1, expected_result)) {
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS\n");
    return true;
}


// Test 5: seq copy (device)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the on-device path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_device(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const generation_result & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOGV(LOG_LEVEL_INFO, "\n=== Test 5: seq copy (device) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (on-device path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens and compare logits against the baseline
    if (!generate_tokens_compare(ctx.get(), smpl.get(), n_past, params.n_predict, 1, expected_result)) {
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS\n");
    return true;
}

// Test 6: sequence file integrity envelope
// - round-trip a sequence file and compare the exact in-memory sequence state
// - reject truncation, a same-length payload bit flip, v2, and foreign magic
// - verify every rejected file leaves the destination sequence and outputs unchanged
static bool test_seq_file_integrity(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    static constexpr size_t SEQ_FILE_HEADER_SIZE = 24;

    std::array<uint8_t, 32> family = {};
    const bool has_family_identity = llama_model_semantic_family_digest(
        model, family.data());
    const auto family_now = [&]() {
        std::array<uint8_t, 32> value = {};
        if (!llama_model_semantic_family_digest(model, value.data())) {
            value.fill(0);
        }
        return value;
    };
    if (has_family_identity) {
        const std::string saved_name = model->name;
        model->name = saved_name + "-renamed-requantized-finetune";
        bool nonsemantic_stable = family_now() == family;
        model->name = saved_name;

        // Quantization/file metadata and model labels are deliberately not
        // part of the semantic family.  The receipt owns the effective
        // architecture/KV/tokenizer shape below, not model weight identity.
        static constexpr const char * metadata_keys[] = {
            "general.file_type",
            "general.quantization_version",
        };
        for (const char * key : metadata_keys) {
            const auto it = model->gguf_kv.find(key);
            const bool had_value = it != model->gguf_kv.end();
            const std::string saved_value = had_value ? it->second : std::string();
            model->gguf_kv[key] = "different-quant-or-finetune-metadata";
            nonsemantic_stable = nonsemantic_stable && family_now() == family;
            if (had_value) {
                model->gguf_kv[key] = saved_value;
            } else {
                model->gguf_kv.erase(key);
            }
        }
        if (!nonsemantic_stable) {
            LOG_ERR("%s: nonsemantic model/quant metadata changed family identity\n", __func__);
            return false;
        }
    }
    // An appended MTP head is a drafter, not target state: the family covers
    // the trunk only, so a model with and without its MTP head matches.
    const bool mtp_excluded =
        model->hparams.has_mtp() && model->hparams.n_layer() > 0;
    const uint32_t il_last =
        (mtp_excluded ? model->hparams.n_layer() : model->hparams.n_layer_all) - 1;
    if (has_family_identity && mtp_excluded) {
        auto & hp = model->hparams;
        const auto saved_hp = hp;
        hp.n_ff_arr[hp.n_layer_all - 1] += 1;
        bool mtp_stable = family_now() == family;
        hp = saved_hp;
        hp.n_layer_all  -= hp.n_layer_nextn;
        hp.n_layer_nextn = 0;
        mtp_stable = mtp_stable && family_now() == family;
        hp = saved_hp;
        if (!mtp_stable) {
            LOG_ERR("%s: MTP head shape or presence changed family identity\n", __func__);
            return false;
        }
    }
    const auto require_family_mutation = [&](auto & field, auto replacement,
                                              const char * label) {
        const auto saved = field;
        field = replacement;
        const bool changed = family_now() != family;
        field = saved;
        if (!changed) {
            LOG_ERR("%s: %s did not change semantic-family identity\n",
                    __func__, label);
        }
        return changed;
    };
    if (has_family_identity &&
       (!require_family_mutation(model->arch, LLM_ARCH_UNKNOWN, "architecture") ||
        !require_family_mutation(model->hparams.n_embd,
            model->hparams.n_embd + 1, "embedding shape") ||
        !require_family_mutation(model->hparams.n_head_kv_arr[0],
            model->hparams.n_head_kv_arr[0] + 1, "KV layout") ||
        !require_family_mutation(model->hparams.n_ff_arr[0],
            model->hparams.n_ff_arr[0] + 1, "per-layer FFN shape") ||
        !require_family_mutation(model->hparams.n_expert,
            model->hparams.n_expert + 1, "expert count") ||
        !require_family_mutation(model->hparams.n_expert_used_arr[0],
            model->hparams.n_expert_used_arr[0] + 1, "selected expert count") ||
        !require_family_mutation(model->hparams.n_expert_used_arr[il_last],
            model->hparams.n_expert_used_arr[il_last] + 1, "last-block selected expert count") ||
        !require_family_mutation(model->hparams.n_ff_exp_arr[il_last],
            model->hparams.n_ff_exp_arr[il_last] + 1, "last-block expert width") ||
        !require_family_mutation(model->hparams.n_expert_shared,
            model->hparams.n_expert_shared + 1, "shared expert topology") ||
        !require_family_mutation(model->hparams.n_expert_groups,
            model->hparams.n_expert_groups + 1, "expert group topology") ||
        !require_family_mutation(model->hparams.expert_group_scale,
            model->hparams.expert_group_scale + 1.0f, "expert group scale") ||
        !require_family_mutation(model->hparams.expert_gating_func,
            model->hparams.expert_gating_func + 1, "expert gating policy") ||
        !require_family_mutation(model->hparams.f_attn_logit_softcapping,
            model->hparams.f_attn_logit_softcapping + 1.0f,
            "attention logit policy") ||
        !require_family_mutation(model->hparams.f_attention_scale,
            model->hparams.f_attention_scale + 1.0f,
            "attention scale") ||
        !require_family_mutation(model->hparams.rope_finetuned,
            !model->hparams.rope_finetuned, "RoPE fine-tuned policy") ||
        !require_family_mutation(model->hparams.n_rot_full,
            model->hparams.n_rot_full + 1, "RoPE dimensions") ||
        !require_family_mutation(model->hparams.rope_freq_base_train,
            model->hparams.rope_freq_base_train + 1.0f, "RoPE base") ||
        !require_family_mutation(model->hparams.rope_freq_scale_train,
            model->hparams.rope_freq_scale_train + 1.0f, "RoPE scale") ||
        !require_family_mutation(model->hparams.rope_sections[0],
            model->hparams.rope_sections[0] + 1, "RoPE sections") ||
        !require_family_mutation(model->hparams.rope_pattern[0],
            model->hparams.rope_pattern[0] ^ 1u, "RoPE layer pattern") ||
        !require_family_mutation(model->hparams.rope_type,
            static_cast<llama_rope_type>(uint32_t(model->hparams.rope_type) + 1),
            "RoPE type") ||
        !require_family_mutation(model->hparams.rope_scaling_type_train,
            static_cast<llama_rope_scaling_type>(
                uint32_t(model->hparams.rope_scaling_type_train) + 1),
            "RoPE scaling type") ||
        !require_family_mutation(model->hparams.swa_type,
            static_cast<llama_swa_type>(uint32_t(model->hparams.swa_type) + 1),
            "SWA type") ||
        !require_family_mutation(model->hparams.n_swa,
            model->hparams.n_swa + 1, "SWA window") ||
        !require_family_mutation(model->hparams.is_recr_impl[0],
            model->hparams.is_recr_impl[0] ^ 1u, "recurrent layout") ||
        !require_family_mutation(model->hparams.ssm_dt_rank,
            model->hparams.ssm_dt_rank + 1, "SSM rank") ||
        !require_family_mutation(model->hparams.ssm_n_group,
            model->hparams.ssm_n_group + 1, "SSM groups") ||
        !require_family_mutation(model->hparams.ssm_dt_b_c_rms,
            !model->hparams.ssm_dt_b_c_rms, "SSM normalization") ||
        !require_family_mutation(model->hparams.dflash2_conv_kernel_size,
            model->hparams.dflash2_conv_kernel_size + 1,
            "recurrent convolution shape") ||
        !require_family_mutation(model->hparams.is_indexer_full_impl[0],
            model->hparams.is_indexer_full_impl[0] ^ 1u, "index layout"))) {
        return false;
    }
    if (has_family_identity) {
        auto & token0 = const_cast<llama_vocab::token_data &>(
            model->vocab.get_token_data(0));
        const std::string saved_token0 = token0.text;
        token0.text.push_back('\0');
        const bool token_mapping_changed = family_now() != family;
        token0.text = saved_token0;
        if (!token_mapping_changed) {
            LOG_ERR("%s: same-size vocabulary mapping mutation was accepted\n",
                    __func__);
            return false;
        }
    }

    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 1;
    params_ctx.kv_unified = true;

    auto ctx_src = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    auto ctx_dst = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx_src || !ctx_dst) {
        LOG_ERR("%s: failed to create contexts\n", __func__);
        return false;
    }

    LOG("\n=== Test 6: sequence file integrity envelope ===\n");

    const size_t n_tokens = std::min<size_t>(tokens.size(), 64);
    if (n_tokens == 0) {
        LOG_ERR("%s: no tokens available\n", __func__);
        return false;
    }

    llama_batch_ptr batch((int32_t) n_tokens, 0, 1);
    for (size_t i = 0; i < n_tokens; ++i) {
        common_batch_add(batch.get(), tokens[i], (llama_pos) i, { 0 }, false);
    }
    if (llama_decode(ctx_src.get(), batch.get())) {
        LOG_ERR("%s: failed to decode source sequence\n", __func__);
        return false;
    }

    const auto get_seq_state = [&](llama_context * ctx, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size(ctx, 0);
        if (state_size == 0) {
            LOG_ERR("%s: sequence state is empty\n", __func__);
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data(ctx, state.data(), state.size(), 0);
        if (ncopy != state.size()) {
            LOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                    __func__, ncopy, state.size());
            return false;
        }
        return true;
    };

    std::vector<uint8_t> state_src;
    if (!get_seq_state(ctx_src.get(), state_src)) {
        return false;
    }

    const std::string seq_path = params.out_file + ".seq-integrity";
    const std::string bad_path = seq_path + ".bad";
    const size_t saved_size = llama_state_seq_save_file(
            ctx_src.get(), seq_path.c_str(), 0, tokens.data(), n_tokens);
    if (saved_size == 0) {
        LOG_ERR("%s: failed to save sequence state file\n", __func__);
        return false;
    }

    std::ifstream input(seq_path, std::ios::binary);
    std::vector<uint8_t> file_bytes(
            (std::istreambuf_iterator<char>(input)),
             std::istreambuf_iterator<char>());
    if (!input.is_open() || file_bytes.size() != saved_size || file_bytes.size() <= SEQ_FILE_HEADER_SIZE) {
        LOG_ERR("%s: failed to read saved sequence state file (%zu bytes, expected %zu)\n",
                __func__, file_bytes.size(), saved_size);
        return false;
    }
    const size_t token_begin = SEQ_FILE_HEADER_SIZE + sizeof(uint32_t);
    const size_t token_bytes = n_tokens*sizeof(llama_token);

    uint64_t declared_size = 0;
    memcpy(&declared_size, file_bytes.data() + 2*sizeof(uint32_t), sizeof(declared_size));
    if (declared_size != file_bytes.size()) {
        LOG_ERR("%s: declared file size %" PRIu64 " does not match actual size %zu\n",
                __func__, declared_size, file_bytes.size());
        return false;
    }

    llama_tokens tokens_out(n_tokens, LLAMA_TOKEN_NULL);
    size_t n_token_count_out = 0;
    const size_t loaded_size = llama_state_seq_load_file(
            ctx_dst.get(), seq_path.c_str(), 0,
            tokens_out.data(), tokens_out.size(), &n_token_count_out);
    if (loaded_size != saved_size ||
            n_token_count_out != n_tokens ||
            !std::equal(tokens.begin(), tokens.begin() + n_tokens, tokens_out.begin())) {
        LOG_ERR("%s: sequence state round-trip metadata mismatch\n", __func__);
        return false;
    }

    std::vector<uint8_t> state_dst;
    if (!get_seq_state(ctx_dst.get(), state_dst) || state_dst != state_src) {
        LOG_ERR("%s: sequence state differs after file round-trip\n", __func__);
        return false;
    }

    // Prepare A once, replace the pathname with a separately authenticated B,
    // and apply A without reopening the path. Then restore A at the path to
    // mutation-pin A->B->A pathname ABA independently of the retained bytes.
    llama_state_seq_file_snapshot prepared_a;
    if (!llama_state_seq_file_snapshot_prepare(seq_path.c_str(), prepared_a)) {
        LOG_ERR("%s: failed to prepare immutable sequence snapshot A\n", __func__);
        return false;
    }
    std::vector<uint8_t> file_b = file_bytes;
    file_b[token_begin] ^= 1;
    static constexpr uint64_t FNV1A64_OFFSET_BASIS = UINT64_C(14695981039346656037);
    static constexpr uint64_t FNV1A64_PRIME        = UINT64_C(1099511628211);
    const auto rewrite_checksum = [&](std::vector<uint8_t> & bytes) {
        uint64_t checksum = FNV1A64_OFFSET_BASIS;
        for (size_t i = SEQ_FILE_HEADER_SIZE; i < bytes.size(); ++i) {
            checksum ^= bytes[i];
            checksum *= FNV1A64_PRIME;
        }
        memcpy(bytes.data() + 2*sizeof(uint32_t) + sizeof(uint64_t),
               &checksum, sizeof(checksum));
    };
    rewrite_checksum(file_b);
    {
        std::ofstream output(seq_path, std::ios::binary | std::ios::trunc);
        output.write((const char *) file_b.data(), file_b.size());
        if (!output) {
            LOG_ERR("%s: failed to replace snapshot path with B\n", __func__);
            return false;
        }
    }
    llama_tokens prepared_tokens(n_tokens, LLAMA_TOKEN_NULL);
    size_t prepared_count = 0;
    ctx_dst->synchronize();
    const size_t prepared_read = ctx_dst->state_seq_apply_file_snapshot(
        0, prepared_a, prepared_tokens.data(), prepared_tokens.size(),
        &prepared_count);
    if (prepared_read != saved_size || prepared_count != n_tokens ||
            !std::equal(tokens.begin(), tokens.begin() + n_tokens,
                        prepared_tokens.begin()) ||
            !get_seq_state(ctx_dst.get(), state_dst) || state_dst != state_src) {
        LOG_ERR("%s: prepared A did not apply exactly after pathname replacement\n", __func__);
        return false;
    }
    {
        std::ofstream output(seq_path, std::ios::binary | std::ios::trunc);
        output.write((const char *) file_bytes.data(), file_bytes.size());
        if (!output) {
            LOG_ERR("%s: failed to complete A-B-A pathname mutant\n", __func__);
            return false;
        }
    }
    std::vector<uint8_t> prepared_token_copy;
    if (!prepared_a.copy_packed_token_bytes(prepared_token_copy) ||
            prepared_token_copy.size() != n_tokens*sizeof(llama_token) ||
            std::memcmp(prepared_token_copy.data(), tokens.data(),
                        prepared_token_copy.size()) != 0) {
        LOG_ERR("%s: prepared snapshot changed across A-B-A pathname mutation\n", __func__);
        return false;
    }

    // The only inspection door returns a copy of the packed token envelope.
    // Mutating that copy must leave the sealed snapshot and its exact A state
    // unchanged across the A-B-A pathname replacement above.
    llama_tokens post_prepare_tokens(n_tokens, LLAMA_TOKEN_NULL);
    size_t post_prepare_count = 0;
    std::vector<uint8_t> post_prepare_state_before;
    if (!get_seq_state(ctx_dst.get(), post_prepare_state_before)) {
        LOG_ERR("%s: failed to read destination before prepared mutation\n", __func__);
        return false;
    }
    prepared_token_copy[0] ^= 1;
    const size_t mutated_apply = ctx_dst->state_seq_apply_file_snapshot(
        0, prepared_a, post_prepare_tokens.data(), post_prepare_tokens.size(),
        &post_prepare_count);
    std::vector<uint8_t> prepared_token_copy_after;
    std::vector<uint8_t> post_prepare_state_after;
    if (mutated_apply != saved_size ||
            post_prepare_count != n_tokens ||
            !std::equal(tokens.begin(), tokens.begin() + n_tokens,
                        post_prepare_tokens.begin()) ||
            !prepared_a.copy_packed_token_bytes(prepared_token_copy_after) ||
            prepared_token_copy_after.size() != n_tokens*sizeof(llama_token) ||
            std::memcmp(prepared_token_copy_after.data(), tokens.data(),
                        prepared_token_copy_after.size()) != 0 ||
            !get_seq_state(ctx_dst.get(), post_prepare_state_after) ||
            post_prepare_state_after != post_prepare_state_before) {
        LOG_ERR("%s: packed-token copy mutation changed the sealed snapshot\n",
                __func__);
        return false;
    }

    const auto expect_rejected_unchanged = [&](const std::vector<uint8_t> & damaged, const char * fault) {
        {
            std::ofstream output(bad_path, std::ios::binary | std::ios::trunc);
            if (!damaged.empty()) {
                output.write((const char *) damaged.data(), damaged.size());
            }
            if (!output) {
                LOG_ERR("%s: failed to write %s test file\n", __func__, fault);
                return false;
            }
        }

        llama_tokens rejected_tokens(n_tokens, LLAMA_TOKEN_NULL);
        const llama_tokens rejected_tokens_before = rejected_tokens;
        size_t rejected_count = std::numeric_limits<size_t>::max();
        if (llama_state_seq_load_file(
                    ctx_dst.get(), bad_path.c_str(), 0,
                    rejected_tokens.data(), rejected_tokens.size(), &rejected_count) != 0) {
            LOG_ERR("%s: %s sequence state file was accepted\n", __func__, fault);
            return false;
        }

        std::vector<uint8_t> state_after;
        if (!get_seq_state(ctx_dst.get(), state_after) || state_after != state_dst) {
            LOG_ERR("%s: %s sequence state file changed the destination sequence\n", __func__, fault);
            return false;
        }
        if (rejected_tokens != rejected_tokens_before ||
                rejected_count != std::numeric_limits<size_t>::max()) {
            LOG_ERR("%s: %s sequence state file changed token outputs\n", __func__, fault);
            return false;
        }
        return true;
    };

    const std::vector<size_t> truncation_offsets = {
        0,
        2*sizeof(uint32_t) - 1,
        SEQ_FILE_HEADER_SIZE - 1,
        file_bytes.size()/2,
        file_bytes.size() - 1,
    };
    for (size_t offset : truncation_offsets) {
        std::vector<uint8_t> truncated(file_bytes.begin(), file_bytes.begin() + offset);
        if (!expect_rejected_unchanged(truncated, "truncated")) {
            return false;
        }
    }

    // Flip an authenticated token byte rather than assuming every
    // architecture has a nonempty sequence-memory payload. Diffusion-style
    // models can validly persist only the token envelope.
    if (token_bytes == 0 || token_begin + token_bytes > file_bytes.size()) {
        LOG_ERR("%s: saved sequence state has no token payload\n", __func__);
        return false;
    }
    std::vector<uint8_t> flipped = file_bytes;
    flipped[token_begin + token_bytes/2] ^= 0x80;
    if (!expect_rejected_unchanged(flipped, "bit-flipped")) {
        return false;
    }

    std::vector<uint8_t> old_version = file_bytes;
    const uint32_t version_v2 = 2;
    memcpy(old_version.data() + sizeof(uint32_t), &version_v2, sizeof(version_v2));
    if (!expect_rejected_unchanged(old_version, "v2")) {
        return false;
    }

    std::vector<uint8_t> foreign_magic = file_bytes;
    const uint32_t bad_magic = 0;
    memcpy(foreign_magic.data(), &bad_magic, sizeof(bad_magic));
    if (!expect_rejected_unchanged(foreign_magic, "foreign-magic")) {
        return false;
    }

    // A valid outer envelope must not make a malformed nonempty semantic
    // state publish either token output. Architectures with no sequence state
    // have no semantic payload to truncate and are covered by the envelope
    // mutants above.
    const size_t state_begin = token_begin + token_bytes;
    if (state_begin < file_bytes.size()) {
        std::vector<uint8_t> malformed_state = file_bytes;
        malformed_state.pop_back();

        const uint64_t malformed_size = malformed_state.size();
        memcpy(malformed_state.data() + 2*sizeof(uint32_t),
               &malformed_size, sizeof(malformed_size));

        uint64_t malformed_checksum = FNV1A64_OFFSET_BASIS;
        for (size_t i = SEQ_FILE_HEADER_SIZE; i < malformed_state.size(); ++i) {
            malformed_checksum ^= malformed_state[i];
            malformed_checksum *= FNV1A64_PRIME;
        }
        memcpy(malformed_state.data() + 2*sizeof(uint32_t) + sizeof(uint64_t),
               &malformed_checksum, sizeof(malformed_checksum));

        {
            std::ofstream output(bad_path, std::ios::binary | std::ios::trunc);
            output.write((const char *) malformed_state.data(), malformed_state.size());
            if (!output) {
                LOG_ERR("%s: failed to write malformed semantic-state test file\n", __func__);
                return false;
            }
        }

        llama_tokens rejected_tokens(n_tokens, LLAMA_TOKEN_NULL);
        const llama_tokens rejected_tokens_before = rejected_tokens;
        size_t rejected_count = std::numeric_limits<size_t>::max();
        if (llama_state_seq_load_file(
                    ctx_dst.get(), bad_path.c_str(), 0,
                    rejected_tokens.data(), rejected_tokens.size(), &rejected_count) != 0 ||
                rejected_tokens != rejected_tokens_before ||
                rejected_count != std::numeric_limits<size_t>::max()) {
            LOG_ERR("%s: malformed semantic state changed token outputs or was accepted\n", __func__);
            return false;
        }
    }

    std::remove(seq_path.c_str());
    std::remove(bad_path.c_str());
    LOG("PASS\n");
    return true;
}

// Test 7: non-hybrid attention-only trim equivalence
// - clone the same attention state into two contexts
// - compare the new attention-only operation with the legacy whole-memory operation
static bool test_attn_trim_nonhybrid(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    if (llama_model_is_recurrent(model) || llama_model_is_hybrid(model)) {
        LOG("\n=== Test 7: non-hybrid attention trim (skipped for recurrent/hybrid model) ===\n");
        return true;
    }

    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx = 256;
    params_ctx.n_seq_max = 1;
    params_ctx.kv_unified = true;

    auto ctx_attn = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    auto ctx_full = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx_attn || !ctx_full) {
        LOG_ERR("%s: failed to create contexts\n", __func__);
        return false;
    }

    LOG("\n=== Test 7: non-hybrid attention trim equivalence ===\n");

    const size_t n_tokens = std::min<size_t>(tokens.size(), 8);
    if (n_tokens < 4) {
        LOG_ERR("%s: need at least four tokens\n", __func__);
        return false;
    }

    llama_batch_ptr batch((int32_t) n_tokens, 0, 1);
    for (size_t i = 0; i < n_tokens; ++i) {
        common_batch_add(batch.get(), tokens[i], (llama_pos) i, { 0 }, false);
    }
    if (llama_decode(ctx_attn.get(), batch.get())) {
        LOG_ERR("%s: failed to decode source sequence\n", __func__);
        return false;
    }
    llama_synchronize(ctx_attn.get());

    llama_memory_t memory_attn = llama_get_memory(ctx_attn.get());
    if (memory_attn == nullptr) {
        LOG("\n=== Test 7: non-hybrid attention trim (skipped: model has no sequence memory) ===\n");
        return true;
    }
    if (llama_memory_seq_pos_max(memory_attn, 0) < 0) {
        LOG_ERR("%s: decoded attention memory contains no sequence positions\n", __func__);
        return false;
    }

    const llama_pos target = (llama_pos) n_tokens - 2;

    std::vector<uint8_t> initial(llama_state_seq_get_size(ctx_attn.get(), 0));
    if (initial.empty() ||
        llama_state_seq_get_data(
            ctx_attn.get(), initial.data(), initial.size(), 0) != initial.size() ||
        llama_state_seq_set_data(
            ctx_full.get(), initial.data(), initial.size(), 0) != initial.size()) {
        LOG_ERR("%s: failed to clone source sequence\n", __func__);
        return false;
    }

    if (!llama_memory_seq_rm_attn(
            memory_attn, 0, target, -1) ||
        !llama_memory_seq_rm(
            llama_get_memory(ctx_full.get()), 0, target, -1)) {
        LOG_ERR("%s: trim operation failed\n", __func__);
        return false;
    }

    std::vector<uint8_t> attn_state(
        llama_state_seq_get_size(ctx_attn.get(), 0));
    std::vector<uint8_t> full_state(
        llama_state_seq_get_size(ctx_full.get(), 0));
    if (attn_state.empty() || attn_state.size() != full_state.size() ||
        llama_state_seq_get_data(
            ctx_attn.get(), attn_state.data(), attn_state.size(), 0) !=
            attn_state.size() ||
        llama_state_seq_get_data(
            ctx_full.get(), full_state.data(), full_state.size(), 0) !=
            full_state.size() ||
        attn_state != full_state ||
        llama_memory_seq_pos_max(llama_get_memory(ctx_attn.get()), 0) != target - 1 ||
        llama_memory_seq_pos_max(llama_get_memory(ctx_full.get()), 0) != target - 1) {
        LOG_ERR("%s: attention-only trim differs from non-hybrid seq_rm\n", __func__);
        return false;
    }

    LOG("PASS\n");
    return true;
}


// Test 8/9: seq copy (scatter)
// - decode the same prefix on two sequences, interleaving seq 0 cells between the seq 1 cells
// - save the seq 1 state, free the interleaved seq 0 cells, and restore via the given io path
// - the restore destination is non-contiguous: scatter reads are batched per contiguous run
// - save again on the host and compare the two blobs byte for byte
static bool test_seq_cp_scatter(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, int test_num, bool on_device) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    LOGV(LOG_LEVEL_INFO, "\n=== Test %d: seq copy (%s, scatter) ===\n", test_num, on_device ? "device" : "host");

    const uint32_t flags = on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : LLAMA_STATE_SEQ_FLAGS_NONE;

    auto decode_one = [&](llama_token tok, int pos, llama_seq_id seq) {
        llama_batch_ptr batch(1, 0, 1);
        common_batch_add(batch.get(), tok, pos, { seq }, true);
        return llama_decode(ctx.get(), batch.get()) == 0;
    };

    // seq 0 cells 0,1,4 interleave the seq 1 cells 2,3,5
    if (!decode_one(tokens[0], 0, 0) ||
        !decode_one(tokens[1], 1, 0) ||
        !decode_one(tokens[0], 0, 1) ||
        !decode_one(tokens[1], 1, 1) ||
        !decode_one(tokens[2], 2, 0) ||
        !decode_one(tokens[2], 2, 1)) {
        LOG_ERR("%s: failed to build interleaved state\n", __func__);
        return false;
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, uint32_t fl, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size_ext(ctx.get(), seq_id, fl);
        if (state_size == 0) {
            LOG_ERR("%s: sequence state is empty\n", __func__);
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), state.data(), state.size(), seq_id, fl);
        if (ncopy != state.size()) {
            LOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                    __func__, ncopy, state.size());
            return false;
        }

        return true;
    };

    // host blob: contains the KV data, used for the byte-for-byte comparison
    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, LLAMA_STATE_SEQ_FLAGS_NONE, state_before)) {
        return false;
    }

    // save via the io path under test
    std::vector<uint8_t> state_save;
    if (!get_seq_state(1, flags, state_save)) {
        return false;
    }
    LOG_TRC("%s: seq 1 saved via %s, %zu bytes\n", __func__, on_device ? "device" : "host", state_save.size());

    // free seq 0's cells so the ring is fragmented: the restore destination (seq 1's interleaved cells) stays non-contiguous
    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return false;
    }

    // restore via the io path under test
    const size_t nset = llama_state_seq_set_data_ext(ctx.get(), state_save.data(), state_save.size(), 1, flags);
    if (nset != state_save.size()) {
        LOG_ERR("%s: seq set data length %zu does not match expected length %zu\n", __func__, nset, state_save.size());
        return false;
    }
    LOG_TRC("%s: seq 1 restored via %s, %zu bytes\n", __func__, on_device ? "device" : "host", nset);

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, LLAMA_STATE_SEQ_FLAGS_NONE, state_after)) {
        return false;
    }

    // the blob is serialized in sequence cell order, so identical bytes iff the restore wrote the same KV
    if (state_before.size() != state_after.size() || memcmp(state_before.data(), state_after.data(), state_before.size()) != 0) {
        LOG_ERR("\n%s: error: restored KV state is not byte-identical to the saved state\n", __func__);
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS\n");
    return true;
}


// Test 10: state blob round-trip
// compares blobs rather than generated text: a partially restored cell still decodes to plausible tokens
static bool test_state_roundtrip(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    LOG("\n=== Test 10: state blob round-trip ===\n");

    if (llama_decode(ctx.get(), llama_batch_get_one(const_cast<llama_token *>(tokens.data()), (int32_t) tokens.size()))) {
        LOG_ERR("\n%s: failed to decode prompt\n", __func__);
        return false;
    }

    std::vector<uint8_t> blob_a(llama_state_seq_get_size(ctx.get(), 0));
    const size_t n_a = llama_state_seq_get_data(ctx.get(), blob_a.data(), blob_a.size(), 0);
    if (n_a != blob_a.size()) {
        LOG_ERR("\n%s: saved %zu bytes, expected %zu\n", __func__, n_a, blob_a.size());
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("\n%s: failed to erase seq 0\n", __func__);
        return false;
    }

    if (llama_state_seq_set_data(ctx.get(), blob_a.data(), blob_a.size(), 0) != blob_a.size()) {
        LOG_ERR("\n%s: failed to restore seq 0\n", __func__);
        return false;
    }

    std::vector<uint8_t> blob_b(llama_state_seq_get_size(ctx.get(), 0));
    const size_t n_b = llama_state_seq_get_data(ctx.get(), blob_b.data(), blob_b.size(), 0);
    if (n_b != n_a) {
        LOG_ERR("\n%s: re-saved %zu bytes, expected %zu\n", __func__, n_b, n_a);
        return false;
    }

    size_t n_diff = 0;
    size_t i_diff = 0;
    for (size_t i = 0; i < n_a; i++) {
        if (blob_a[i] != blob_b[i]) {
            if (n_diff == 0) {
                i_diff = i;
            }
            n_diff++;
        }
    }

    if (n_diff > 0) {
        LOG_ERR("\n%s: state changed across a restore: %zu of %zu bytes differ, first at offset %zu\n",
                __func__, n_diff, n_a, i_diff);
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS\n");
    return true;
}

// Test 11: position-range blobs
// - build seq 1 with its cells interleaved between seq 0 cells, so the range writer has to order by position
// - a fresh context that appends the range in three blobs and then takes the partial state must end up
//   with the same sequence state, byte for byte, as the source
// - one blob for the whole range must install under the other stream layout (unified <-> split)
// - p_limit keeps a prefix of a blob; a blob that does not continue the sequence, names another range
//   or is cut short is refused and leaves the sequence as it was
static bool test_state_range(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    LOG("\n=== Test 11: position-range blobs ===\n");

    if (llama_model_is_recurrent(model)) {
        LOG("a recurrent state has no position-addressable part, skipped\n");
        return true;
    }

    const llama_pos n  = (llama_pos) std::min<size_t>(tokens.size(), 48);
    const llama_pos c0 = n / 3;
    const llama_pos c1 = 2 * n / 3;
    if (c0 < 2) {
        LOG_ERR("%s: the prompt is too short\n", __func__);
        return false;
    }

    // plain KV caches ignore PARTIAL_ONLY and write everything, so only these have a partial part
    const bool has_partial = llama_model_is_recurrent(model) || llama_model_is_hybrid(model) || llama_model_n_swa(model) > 0;

    const auto make_ctx = [&](bool unified) {
        auto params_ctx = common_context_params_to_llama(params);
        params_ctx.n_ctx      = 512;
        params_ctx.n_seq_max  = 2;
        params_ctx.kv_unified = unified;
        return llama_context_ptr{llama_init_from_model(model, params_ctx)};
    };

    const auto get_range = [&](llama_context * ctx, llama_pos p0, llama_pos p1, std::vector<uint8_t> & blob) {
        blob.resize(llama_state_seq_get_size_range(ctx, 1, p0, p1));
        return !blob.empty() && llama_state_seq_get_data_range(ctx, blob.data(), blob.size(), 1, p0, p1) == blob.size();
    };

    const auto get_state = [&](llama_context * ctx, llama_state_seq_flags flags, std::vector<uint8_t> & blob) {
        blob.resize(llama_state_seq_get_size_ext(ctx, 1, flags));
        return !blob.empty() && llama_state_seq_get_data_ext(ctx, blob.data(), blob.size(), 1, flags) == blob.size();
    };

    const auto append = [&](llama_context * ctx, const std::vector<uint8_t> & blob, llama_pos p0, llama_pos p1, llama_pos p_limit) {
        return llama_state_seq_append_data(ctx, blob.data(), blob.size(), 1, p0, p1, p_limit) > 0;
    };

    std::vector<uint8_t> src_full;
    std::vector<uint8_t> src_partial;
    std::vector<uint8_t> src_all;  // [0, n)
    std::vector<uint8_t> src_head; // [0, c0)
    std::vector<uint8_t> src_mid;  // [c0, c1)
    std::vector<uint8_t> src_tail; // [c1, n)
    std::vector<uint8_t> src_cut;  // [0, c1 - 1)

    // one context at a time: a dynamic-VBR cache does not share the device with a second one
    {
        auto ctx = make_ctx(true);
        if (!ctx) {
            LOG_ERR("%s: failed to create the source context\n", __func__);
            return false;
        }

        for (llama_pos i = 0; i < n; ++i) {
            for (llama_seq_id seq = 0; seq < 2; ++seq) {
                llama_batch_ptr batch(1, 0, 1);
                common_batch_add(batch.get(), tokens[i], i, { seq }, true);
                if (llama_decode(ctx.get(), batch.get()) != 0) {
                    LOG_ERR("%s: failed to decode position %d of seq %d\n", __func__, i, seq);
                    return false;
                }
            }
        }

        if (!get_state(ctx.get(), LLAMA_STATE_SEQ_FLAGS_NONE, src_full) ||
            (has_partial && !get_state(ctx.get(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, src_partial)) ||
            !get_range(ctx.get(), 0,  n,      src_all)  ||
            !get_range(ctx.get(), 0,  c0,     src_head) ||
            !get_range(ctx.get(), c0, c1,     src_mid)  ||
            !get_range(ctx.get(), c1, n,      src_tail) ||
            !get_range(ctx.get(), 0,  c1 - 1, src_cut)) {
            LOG_ERR("%s: failed to save the source blobs\n", __func__);
            return false;
        }

        // a range the sequence does not hold cannot be written
        if (llama_state_seq_get_size_range(ctx.get(), 1, 0, n + 1) != 0) {
            LOG_ERR("%s: a range past the end of the sequence was written\n", __func__);
            return false;
        }
    }

    // three blobs and the partial state rebuild the source state
    {
        auto ctx = make_ctx(true);
        if (!ctx) {
            LOG_ERR("%s: failed to create the tiling context\n", __func__);
            return false;
        }

        if (!append(ctx.get(), src_head, 0,  c0, n) ||
            !append(ctx.get(), src_mid,  c0, c1, n) ||
            !append(ctx.get(), src_tail, c1, n,  n)) {
            LOG_ERR("%s: failed to append the three blobs\n", __func__);
            return false;
        }

        if (has_partial && llama_state_seq_set_data_ext(ctx.get(), src_partial.data(), src_partial.size(), 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != src_partial.size()) {
            LOG_ERR("%s: failed to install the partial state\n", __func__);
            return false;
        }

        std::vector<uint8_t> dst_full;
        if (!get_state(ctx.get(), LLAMA_STATE_SEQ_FLAGS_NONE, dst_full) || dst_full != src_full) {
            LOG_ERR("%s: the tiled sequence state differs from the source (%zu vs %zu bytes)\n", __func__, dst_full.size(), src_full.size());
            return false;
        }
    }

    // the other stream layout takes the same blob
    {
        auto ctx = make_ctx(false);
        if (!ctx) {
            // a dynamic-VBR cache exists only unified
            LOG_WRN("%s: no split-stream context for this configuration, the cross-layout leg is skipped\n", __func__);
        } else {
            std::vector<uint8_t> dst_all;
            if (!append(ctx.get(), src_all, 0, n, n) || !get_range(ctx.get(), 0, n, dst_all) || dst_all != src_all) {
                LOG_ERR("%s: a blob written under unified KV did not install under split KV\n", __func__);
                return false;
            }
        }
    }

    // p_limit and the refusals
    {
        auto ctx = make_ctx(true);
        if (!ctx) {
            LOG_ERR("%s: failed to create the p_limit context\n", __func__);
            return false;
        }

        // the base part holds exactly [0, p). not seq_pos_max: on a hybrid or SWA memory that is the minimum
        // over both parts, and the partial part stays empty here
        const auto holds = [&](llama_pos p) {
            return (p == 0 || llama_state_seq_get_size_range(ctx.get(), 1, 0, p) > 0) &&
                llama_state_seq_get_size_range(ctx.get(), 1, 0, p + 1) == 0;
        };

        // does not continue the (empty) sequence
        if (append(ctx.get(), src_mid, c0, c1, n) || !holds(0)) {
            LOG_ERR("%s: a blob that does not start at 0 went into an empty sequence\n", __func__);
            return false;
        }

        if (!append(ctx.get(), src_head, 0, c0, n)) {
            LOG_ERR("%s: failed to append the head blob\n", __func__);
            return false;
        }

        // names another range than the one it was written for
        if (append(ctx.get(), src_mid, c0, c1 + 1, n) || !holds(c0)) {
            LOG_ERR("%s: a blob was accepted for another range\n", __func__);
            return false;
        }

        // cut short: the cells of the failed call go, the earlier ones stay
        if (llama_state_seq_append_data(ctx.get(), src_mid.data(), src_mid.size() - 8, 1, c0, c1, n) != 0 || !holds(c0)) {
            LOG_ERR("%s: a truncated blob was accepted or left cells behind\n", __func__);
            return false;
        }

        // p_limit drops the last cell of the blob
        std::vector<uint8_t> dst_cut;
        if (!append(ctx.get(), src_mid, c0, c1, c1 - 1) || !holds(c1 - 1) ||
            !get_range(ctx.get(), 0, c1 - 1, dst_cut) || dst_cut != src_cut) {
            LOG_ERR("%s: p_limit did not keep exactly the prefix of the blob\n", __func__);
            return false;
        }
    }

    LOG("\nPASS\n");
    return true;
}


// overwrite the tensor data with 0xff bytes (NaN when read as f16/f32), so that the restore fails
static bool corrupt_state(std::vector<uint8_t> & data) {
    if (data.size() < 3*4096) {
        LOG_ERR("%s: state of %zu bytes is too small to corrupt\n", __func__, data.size());
        return false;
    }

    std::fill(data.begin() + 4096, data.end() - data.size()/4, 0xff);
    return true;
}


// Test 12: state restore failure
// a failed restore must leave the sequence empty and must not change the logits of other sequences
static bool test_state_restore_failure(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 4;
    params_ctx.kv_unified = true;

    // without flash attention, corrupted data left behind by the restore shows up as NaN logits on the other sequences
    params_ctx.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\n=== Test 12: state restore failure ===\n");

    llama_memory_t mem = llama_get_memory(ctx.get());
    if (mem == nullptr) {
        LOGV(LOG_LEVEL_INFO, "PASS (model has no memory)\n");
        return true;
    }

    const auto decode = [&](const llama_tokens & inp, llama_seq_id seq_id, std::vector<float> * logits_out) {
        llama_batch_ptr batch(inp.size(), 0, 1);
        for (size_t i = 0; i < inp.size(); ++i) {
            common_batch_add(batch.get(), inp[i], i, { seq_id }, i == inp.size() - 1);
        }

        if (llama_decode(ctx.get(), batch.get())) {
            LOG_ERR("%s: failed to decode on sequence %d\n", __func__, seq_id);
            return false;
        }

        if (logits_out && !get_current_logits(ctx.get(), *logits_out)) {
            LOG_ERR("%s: failed to get logits\n", __func__);
            return false;
        }

        return true;
    };

    const llama_tokens tokens_save  (tokens.begin(), tokens.begin() + std::min<size_t>(24, tokens.size()));
    const llama_tokens tokens_verify(tokens.end() - std::min<size_t>(8, tokens.size()), tokens.end());

    // the registered tests share a working directory, so the state file is named after the model
    const std::string path = "state-restore-failure." + std::filesystem::path(params.model.path).filename().string() + ".tmp.bin";

    llama_memory_clear(mem, true);

    std::vector<float> baseline;
    if (!decode(tokens_verify, 1, &baseline)) {
        return false;
    }

    const std::vector<std::pair<const char *, std::function<bool()>>> cases = {
        { "buffer", [&]() {
            std::vector<uint8_t> state(llama_state_seq_get_size(ctx.get(), 0));
            GGML_ASSERT(llama_state_seq_get_data(ctx.get(), state.data(), state.size(), 0) == state.size());
            llama_memory_seq_rm(mem, 0, -1, -1);

            if (!corrupt_state(state)) {
                return false;
            }

            return llama_state_seq_set_data(ctx.get(), state.data(), state.size(), 0) == 0;
        }},
        { "file", [&]() {
            GGML_ASSERT(llama_state_seq_save_file(ctx.get(), path.c_str(), 0, tokens_save.data(), tokens_save.size()) > 0);
            llama_memory_seq_rm(mem, 0, -1, -1);

            std::vector<uint8_t> data;
            {
                std::ifstream f(path, std::ios::binary);
                data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            }

            if (!corrupt_state(data)) {
                std::remove(path.c_str());
                return false;
            }

            {
                std::ofstream f(path, std::ios::binary);
                f.write((const char *) data.data(), data.size());
            }

            llama_tokens tokens_out(tokens_save.size());
            size_t n_token_count = 0;
            const size_t nread = llama_state_seq_load_file(ctx.get(), path.c_str(), 0, tokens_out.data(), tokens_out.size(), &n_token_count);
            std::remove(path.c_str());

            return nread == 0;
        }},
    };

    for (const auto & [name, restore_failed] : cases) {
        llama_memory_clear(mem, true);

        if (!decode(tokens_save, 0, nullptr)) {
            return false;
        }

        // Keep a live neighbour sharing the source prefix across the failure.
        // A cold decode alone would miss erased memberships.
        llama_memory_seq_cp(mem, 0, 2, -1, -1);
        const auto neighbour = [&]() {
            std::vector<uint8_t> bytes(llama_state_seq_get_size(ctx.get(), 2));
            if (bytes.empty() || llama_state_seq_get_data(ctx.get(), bytes.data(), bytes.size(), 2) != bytes.size()) {
                return std::vector<uint8_t>{};
            }
            return bytes;
        };
        const auto before = neighbour();
        if (before.empty()) {
            return false;
        }

        if (!restore_failed()) {
            LOG_ERR("%s: %s: restoring a corrupted state did not fail\n", __func__, name);
            return false;
        }

        if (before != neighbour()) {
            LOG_ERR("%s: %s: failed restore changed a live neighbour\n", __func__, name);
            return false;
        }
        llama_memory_seq_rm(mem, 2, -1, -1);

        if (llama_memory_seq_pos_max(mem, 0) != -1) {
            LOG_ERR("%s: %s: sequence not empty after failed restore\n", __func__, name);
            return false;
        }

        std::vector<float> logits;
        if (!decode(tokens_verify, 1, &logits)) {
            return false;
        }

        float  diff_max = 0.0f;
        size_t n_nan    = 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            if (!std::isfinite(logits[i]) || !std::isfinite(baseline[i])) {
                n_nan++;
            } else {
                diff_max = std::max(diff_max, std::fabs(logits[i] - baseline[i]));
            }
        }

        if (n_nan > 0 || diff_max > 1e-6f) {
            LOG_ERR("%s: %s: logits changed after failed restore (max diff = %g, nan = %zu)\n", __func__, name, diff_max, n_nan);
            return false;
        }

        LOG_TRC("%s: %s: logits match (max diff = %g)\n", __func__, name, diff_max);
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS\n");
    return true;
}


struct test_suite {
    std::vector<test_status> results;

    bool all_passed() const {
        return !results.empty() && std::all_of(results.begin(), results.end(), [](test_status s) { return s == test_status::PASS; });
    }
};


// --range-only: run test 11 alone, it needs no generation baseline
static bool g_range_only = false;

static std::vector<const char *> test_names = {
    "baseline", "seq_rm", "state_load", "cp_h", "cp_d", "integrity",
    "trim", "cp_h_s", "cp_d_s", "rt", "range", "rf",
};

// Run the full save/load test suite (tests 1-12) for a single model.
static test_suite run_save_load_tests_for_model(const std::string & model_path, const struct common_params & base_params) {
    test_suite suite;
    struct common_params params = base_params;
    params.model.path = model_path;

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());
        suite.results.assign(test_names.size(), test_status::SKIP);
        suite.results.front() = test_status::FAIL;
        return suite;
    }

    GGML_ASSERT(llama_init->context() == nullptr);

    // Tokenize prompt or generate random tokens
    llama_tokens tokens;
    if (params.prompt.empty()) {
        const int n_prompt = params.n_batch;

        // this path is useful for model files that do not have a tokenizer
        LOG_INF("%s: no prompt provided, generating %d (n_batch) random tokens\n", __func__, n_prompt);

        const auto * vocab = llama_model_get_vocab(model);
        const auto n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < n_prompt; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        LOG_INF("%s: tokenizing prompt '%s'\n", __func__, params.prompt.c_str());

        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    LOG_INF("%s: the input prompt is %d tokens\n", __func__, (int)tokens.size());

    if (g_range_only) {
        suite.results.push_back(test_state_range(model, params, tokens) ? test_status::PASS : test_status::FAIL);
        return suite;
    }

    // Test 1: baseline (saves state to disk)
    auto result_baseline = test_baseline(model, params, tokens);
    suite.results.push_back(result_baseline.empty() ? test_status::FAIL : test_status::PASS);

    // Test 2: sequence removal isolation
    suite.results.push_back(test_seq_rm_isolated(model, params, tokens) ? test_status::PASS : test_status::FAIL);

    if (!result_baseline.empty()) {
        // Test 3: state load
        suite.results.push_back(test_state_load(model, params, tokens, result_baseline) ? test_status::PASS : test_status::FAIL);

        // Test 4: seq copy (host)
        suite.results.push_back(test_seq_cp_host(model, params, tokens, result_baseline) ? test_status::PASS : test_status::FAIL);

        // Test 5: seq copy (device)
        suite.results.push_back(test_seq_cp_device(model, params, tokens, result_baseline) ? test_status::PASS : test_status::FAIL);
    } else {
        // tests 3-5 depend on the baseline result and the state file it saves
        suite.results.push_back(test_status::SKIP);
        suite.results.push_back(test_status::SKIP);
        suite.results.push_back(test_status::SKIP);
    }

    // Test 6: checksummed sequence file envelope and staged loading
    suite.results.push_back(test_seq_file_integrity(model, params, tokens) ? test_status::PASS : test_status::FAIL);

    // Test 7: non-hybrid attention trim equivalence
    suite.results.push_back(test_attn_trim_nonhybrid(model, params, tokens) ? test_status::PASS : test_status::FAIL);

    // Test 8: seq copy (host, scatter)
    suite.results.push_back(test_seq_cp_scatter(model, params, tokens, 8, false) ? test_status::PASS : test_status::FAIL);

    // Test 9: seq copy (device, scatter)
    suite.results.push_back(test_seq_cp_scatter(model, params, tokens, 9, true) ? test_status::PASS : test_status::FAIL);

    // Test 10: state blob round-trip
    suite.results.push_back(test_state_roundtrip(model, params, tokens) ? test_status::PASS : test_status::FAIL);

    // Test 11: position-range blobs
    suite.results.push_back(test_state_range(model, params, tokens) ? test_status::PASS : test_status::FAIL);
    suite.results.push_back(test_state_restore_failure(model, params, tokens) ? test_status::PASS : test_status::FAIL);

    LOG("\n%s\n", suite.all_passed() ? "All tests passed." : "Some tests failed or were skipped.");

    return suite;
}


static void print_usage(int /* argc */, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  %s -m your_model.gguf\n", argv[0]);
    LOG("\n  %s --models tests/test-models\n", argv[0]);
    LOG("\n  %s -m your_model.gguf -lv 5\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "";
    params.n_batch = 100;
    params.out_file = "dump_state.bin";
    params.sampling.seed = 1234;

    common_init();

    // extract our own --models DIR option before handing the rest to the common arg parser
    std::string models_dir;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --models requires a directory argument\n", __func__);
                return 1;
            }
            models_dir = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--range-only") == 0) {
            g_range_only = true;
            test_names = { "range" };
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int)filtered_argv.size() - 1;

    // in --models mode there is no single model; set a placeholder so the common parser's
    // "--model is required" check passes (each model is set individually inside the loop)
    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    if (params.n_parallel == 1) {
        LOG_TRC("%s: n_parallel == 1, enabling unified kv cache\n", __func__);
        params.kv_unified = true;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    llama_backend_init();

    if (!models_dir.empty()) {
        // run the suite over every dummy model in the directory
        if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
            LOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
            return 1;
        }

        std::vector<std::string> models;
        for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                models.push_back(entry.path().string());
            }
        }
        std::sort(models.begin(), models.end());

        if (models.empty()) {
            LOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
            return 1;
        }

        auto col_width = [](const char * name) { return (int) std::max(strlen(name), (size_t) 4); };

        size_t name_width = 5; // "Model"
        for (const auto & model_path : models) {
            name_width = std::max(name_width, std::filesystem::path(model_path).filename().string().size());
        }

        // silence everything but the table itself (LOG has verbosity LOG_LEVEL_OUTPUT = 0)
        common_log_set_verbosity_thold(0);

        LOG("%-*s", (int) name_width, "Model");
        for (const auto & name : test_names) {
            LOG("  %-*s", col_width(name), name);
        }
        LOG("\n");
        common_log_flush(common_log_main());

        size_t n_pass = 0;
        size_t n_fail = 0;
        for (const auto & model_path : models) {
            const auto name = std::filesystem::path(model_path).filename().string();

            LOG("%-*s", (int) name_width, name.c_str());
            common_log_flush(common_log_main());

            const test_suite suite = run_save_load_tests_for_model(model_path, params);

            for (size_t i = 0; i < suite.results.size(); i++) {
                LOG("  %s%*s", test_status_str(suite.results[i]), col_width(test_names[i]) - 4, "");
            }
            LOG("\n");
            common_log_flush(common_log_main());

            if (suite.all_passed()) {
                n_pass++;
            } else {
                n_fail++;
            }
        }

        common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
        common_log_flush(common_log_main());

        LOG_INF("%s: summary: %zu passed, %zu failed (of %zu)\n", __func__, n_pass, n_fail, models.size());

        return n_fail == 0 ? 0 : 1;
    }

    // single-model mode
    const test_suite suite = run_save_load_tests_for_model(params.model.path, params);
    const bool all_passed = suite.all_passed();
    if (all_passed) {
        LOG("\nAll tests passed.\n");
    }
    return all_passed ? 0 : 1;
}
