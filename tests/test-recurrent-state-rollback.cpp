#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-tree.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <set>
#include <string>
#include <vector>

static bool cache_pattern_fill = false;

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override { size += n; }
    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }
    size_t n_bytes() override { return size; }
};

static llama_context_ptr init_ctx(llama_model * model, llama_context_params cparams) {
    llama_context_ptr ctx(llama_init_from_model(model, cparams));
    if (!ctx || !cache_pattern_fill) {
        return ctx;
    }

    // Discover buffers after a full ubatch, preserving the prefill allocation size.
    const uint32_t count = llama_n_ubatch(ctx.get());
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, 0, pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx.get(), batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        return nullptr;
    }
    llama_synchronize(ctx.get());
    cache_buffer_collector collector;
    llama_get_memory(ctx.get())->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    if (collector.buffers.empty()) {
        fprintf(stderr, "%s : no cache buffers found\n", __func__);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, 0x3e);
    }
    return ctx;
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

static llama_context_ptr make_ctx(const common_params & params, llama_model * model, uint32_t n_seq_max = 1) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = n_seq_max;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  n_seq_max * (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, n_seq_max * (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams);
}

static llama_memory_recurrent * get_recurrent(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (auto * recurrent = dynamic_cast<llama_memory_recurrent *>(mem)) {
        return recurrent;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return hybrid->get_mem_recr();
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        return hybrid->get_mem_recr();
    }
    return nullptr;
}

static llama_pos get_attention_pos_max(llama_context * ctx, llama_seq_id seq_id) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return hybrid->get_mem_attn()->seq_pos_max(seq_id);
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        return hybrid->get_mem_attn()->seq_pos_max(seq_id);
    }
    return -1;
}

static bool check_depth(llama_context * ctx, llama_seq_id seq_id, uint32_t expected, const char * label) {
    const auto * recurrent = get_recurrent(ctx);
    if (recurrent == nullptr || seq_id < 0 || (size_t) seq_id >= recurrent->rollback_valid_depth.size()) {
        fprintf(stderr, "%s : cannot read rollback depth for sequence %d\n", label, seq_id);
        return false;
    }
    const uint32_t actual = recurrent->rollback_valid_depth[seq_id];
    if (actual != expected) {
        fprintf(stderr, "%s : rollback depth mismatch for sequence %d (%u != %u)\n",
                label, seq_id, actual, expected);
        return false;
    }
    return true;
}

static bool decode_range(
        llama_context *                  ctx,
        const std::vector<llama_token> & tokens,
        uint32_t                         begin,
        uint32_t                         count,
        llama_seq_id                     seq_id = 0) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t pos = begin + i;
        common_batch_add(batch, tokens[pos], pos, { seq_id }, i + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_equal_split(
        llama_context *                  ctx,
        const std::vector<llama_token> & tokens,
        uint32_t                         n_seq_tokens,
        uint32_t                         n_seqs) {
    llama_batch batch = llama_batch_init(n_seq_tokens * n_seqs, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t pos = 0; pos < n_seq_tokens; ++pos) {
            const uint32_t i = s * n_seq_tokens + pos;
            common_batch_add(batch, tokens[i], pos, { (llama_seq_id) s }, pos + 1 == n_seq_tokens);
        }
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static std::vector<uint8_t> save_seq(
        llama_context *       ctx,
        llama_seq_id          seq_id,
        llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_NONE) {
    std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx, seq_id, flags));
    const size_t n = llama_state_seq_get_data_ext(ctx, state.data(), state.size(), seq_id, flags);
    if (n != state.size()) {
        state.clear();
    }
    return state;
}

static bool load_seq(llama_context * ctx, const std::vector<uint8_t> & state, llama_seq_id seq_id) {
    return !state.empty() && llama_state_seq_set_data(ctx, state.data(), state.size(), seq_id) == state.size();
}

static bool seq_state_payload_equal(
        const std::vector<uint8_t> & lhs,
        const std::vector<uint8_t> & rhs) {
    // The in-memory sequence envelope starts with magic + source seq_id. The memory
    // payload that follows must be identical when comparing two different sequence ids.
    constexpr size_t envelope_size = sizeof(uint32_t) + sizeof(llama_seq_id);
    return lhs.size() == rhs.size() && lhs.size() >= envelope_size &&
        std::equal(lhs.begin() + envelope_size, lhs.end(), rhs.begin() + envelope_size);
}

static std::vector<float> copy_logits(llama_context * ctx, int n_vocab, int index = 0) {
    const float * logits = llama_get_logits_ith(ctx, index);
    return logits == nullptr ? std::vector<float>() : std::vector<float>(logits, logits + n_vocab);
}

static bool logits_equal(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        const char *               label,
        float                      eps = 1e-5f) {
    if (lhs.size() != rhs.size() || lhs.empty()) {
        fprintf(stderr, "%s : missing or differently sized logits\n", label);
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!std::isfinite(lhs[i]) || !std::isfinite(rhs[i]) || std::fabs(lhs[i] - rhs[i]) > eps) {
            fprintf(stderr, "%s : logits mismatch at token %zu (%g != %g)\n",
                    label, i, (double) lhs[i], (double) rhs[i]);
            return false;
        }
    }
    return true;
}

// Synthetic image-shaped primary positions exercise row multiplicity/gaps;
// actual projector inputs and media identity are covered by server gates.
static bool test_share_media_prefix(const common_params & params, llama_model * model) {
    const std::vector<llama_pos> rows = {0, 1, 2, 2, 2, 2, 4, 5, 6, 7};
    llama_kv_cells coverage;
    coverage.resize(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        coverage.pos_set(i, rows[i]);
        coverage.seq_add(i, 0);
    }
    auto missing = rows;
    missing.erase(missing.begin() + 2);
    auto extra = rows;
    extra.insert(extra.begin() + 2, 2);
    if (!coverage.seq_has_prefix_rows(0, 8, rows) || coverage.seq_has_prefix(0, 8) ||
        coverage.seq_has_prefix_rows(0, 8, missing) || coverage.seq_has_prefix_rows(0, 8, extra) ||
        coverage.seq_has_prefix_rows(0, 7, rows) || coverage.seq_has_prefix_rows(1, 8, rows)) {
        return false;
    }
    // An image at the left edge requires every repeated row. Once its primary
    // position leaves the window, those rows no longer contribute to coverage.
    if (!coverage.seq_has_prefix_rows(0, 8, rows, 2) ||
        coverage.seq_has_prefix_rows(0, 8, missing, 2) ||
        coverage.seq_has_prefix_rows(0, 8, rows, -1) ||
        coverage.seq_has_prefix_rows(0, 8, rows, 8)) { return false; }
    auto unsorted = rows;
    std::swap(unsorted[1], unsorted[2]);
    if (coverage.seq_has_prefix_rows(0, 8, unsorted, 2)) { return false; }
    coverage.seq_rm(2, 0);
    if (coverage.seq_has_prefix_rows(0, 8, rows, 2) ||
        !coverage.seq_has_prefix_rows(0, 8, rows, 3)) { return false; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(256);
    for (size_t i = 0; i < tokens.size(); ++i) { tokens[i] = llama_token(i + 1); }
    for (bool remove_source : {false, true}) {
        std::vector<float> reference;
        for (int arm = 0; arm < 3; ++arm) {
            auto ctx = make_ctx(params, model, 3);
            if (!ctx) { return false; }
            auto * mem = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()));
            if (!mem) { return false; }
            auto batch = llama_batch_init(rows.size(), 0, 1);
            std::vector<llama_pos> positions(rows.size() * 4);
            for (size_t i = 0; i < rows.size(); ++i) {
                common_batch_add(batch, tokens[i], rows[i], {0}, i + 1 == rows.size());
                for (size_t d = 0; d < 4; ++d) { positions[d * rows.size() + i] = rows[i]; }
            }
            auto * allocated_pos = batch.pos;
            batch.pos = positions.data();
            const bool decoded = llama_decode(ctx.get(), batch) == 0;
            batch.pos = allocated_pos;
            llama_batch_free(batch);
            if (!decoded) { return false; }
            const auto partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
            const auto checkpoint = save_seq(ctx.get(), 0, partial);
            const auto checkpoint_full = save_seq(ctx.get(), 0);
            if (checkpoint.empty() || checkpoint_full.empty()) { return false; }
            if (arm == 0) { mem->seq_cp(0, 1, 0, 8); }
            if (!decode_range(ctx.get(), tokens, 8, 32)) { return false; }
            const auto source_before = save_seq(ctx.get(), 0);
            if (arm != 0) {
                if (mem->can_share_attn_prefix_rows(0, 1, 8, missing) ||
                    mem->try_share_attn_prefix_rows(0, 1, 8, extra) || mem->seq_pos_max(1) != -1 ||
                    !mem->try_share_attn_prefix_rows(0, 1, 8, rows) ||
                    mem->try_share_attn_prefix_rows(0, 1, 8, rows) ||
                    llama_state_seq_set_data_ext(ctx.get(), checkpoint.data(), checkpoint.size(), 1, partial)
                        != checkpoint.size()) {
                    return false;
                }
            }
            if (source_before != save_seq(ctx.get(), 0)) { return false; }
            const auto restored = save_seq(ctx.get(), 1);
            if (restored.empty()) { return false; }
            const llama_seq_id removed = remove_source ? 0 : 1;
            const llama_seq_id survivor = 1 - removed;
            if (!mem->seq_rm(removed, -1, -1) || !decode_range(ctx.get(), tokens, 0, 4, removed)) {
                return false;
            }
            const uint32_t frontier = remove_source ? 8 : 40;
            std::vector<float> logits;
            for (uint32_t pos = frontier; pos < frontier + 32; ++pos) {
                if (!decode_range(ctx.get(), tokens, pos, 1, survivor)) { return false; }
                const auto row = copy_logits(ctx.get(), n_vocab);
                logits.insert(logits.end(), row.begin(), row.end());
            }
            if (arm == 0) { reference = std::move(logits); }
            else if (!logits_equal(reference, logits, "Media row sharing exact replay", 0.0f)) { return false; }
            // Normalize all serialized sequence IDs through the reader, not
            // just the outer envelope (attention metadata embeds IDs too).
            ctx.reset();
            auto canonical = make_ctx(params, model, 3);
            if (!canonical || !load_seq(canonical.get(), checkpoint_full, 0)) { return false; }
            const auto expected = save_seq(canonical.get(), 0);
            if (!load_seq(canonical.get(), restored, 0) || expected != save_seq(canonical.get(), 0)) {
                fprintf(stderr, "Media restored image differs after sequence-ID normalization\n");
                return false;
            }
            fprintf(stderr, "Media rows %s-first arm=%d: exact image, source unchanged, exact 32-row replay PASS\n",
                remove_source ? "source" : "destination", arm);
        }
    }
    return true;
}

static bool test_shared_swa_wrap(const common_params & params, llama_model * model) {
    auto ctx = make_ctx(params, model, 3);
    if (!ctx) { return false; }
    auto * mem = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx.get()));
    const uint32_t window = llama_model_n_swa(model);
    if (!mem || window == 0 || mem->get_swa()->get_size() <= window + 64) { return false; }
    const uint32_t prefix = mem->get_swa()->get_size() - 64;
    const uint32_t end = prefix + window + 128;
    if (end + 32 > llama_n_ctx(ctx.get())) { return false; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(end + 32);
    for (size_t i = 0; i < tokens.size(); ++i) { tokens[i] = llama_token(1 + i % (n_vocab - 1)); }
    const auto fill = [&](uint32_t begin, uint32_t stop, llama_seq_id seq) {
        for (; begin < stop; begin += 64) {
            if (!decode_range(ctx.get(), tokens, begin, std::min(64u, stop - begin), seq)) {
                fprintf(stderr, "Shared SWA wrap stalled: seq=%d pos=%u\n", seq, begin);
                return false;
            }
        }
        return true;
    };
    if (!fill(0, prefix, 0) || !mem->try_share_live_prefix(0, 1, prefix)) { return false; }
    const auto paused = params.vbr_dynamic() ? std::vector<uint8_t>() :
        save_seq(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (!fill(prefix, end, 0)) { return false; }
    if (!params.vbr_dynamic() && (paused.empty() ||
        paused != save_seq(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY))) {
        fprintf(stderr, "Shared SWA wrap changed the paused owner's window bytes\n");
        return false;
    }
    // Advancing the faster owner must not recycle the slower owner's live window.
    if (!mem->can_share_live_prefix(1, 2, prefix)) {
        fprintf(stderr, "Shared SWA wrap lost lagging owner coverage\n");
        return false;
    }
    if (!fill(prefix, end, 1)) { return false; }
    for (llama_seq_id seq : {0, 1}) {
        if (mem->get_swa()->seq_pos_min(seq) < llama_pos(prefix - window)) {
            fprintf(stderr, "Shared SWA wrap did not recycle old shared rows: seq=%d min=%d\n",
                    seq, mem->get_swa()->seq_pos_min(seq));
            return false;
        }
        for (uint32_t pos = end; pos < end + 32; ++pos) {
            if (!decode_range(ctx.get(), tokens, pos, 1, seq)) { return false; }
            const auto logits = copy_logits(ctx.get(), n_vocab);
            if (!logits_equal(logits, logits, "Shared SWA wrap finite continuation", 0.0f)) { return false; }
        }
    }
    fprintf(stderr, "Shared SWA wrap: lagging owner preserved, both owners recycled, continuation PASS\n");
    return true;
}

static bool test_share_media_swa_prefix(const common_params & params, llama_model * model) {
    const uint32_t window = llama_model_n_swa(model);
    if (window < 256) { return false; }
    // Synthetic image-shaped rows share position256. Keep distinct positions
    // contiguous so this also runs on non-M-RoPE SWA models. The row set lies
    // just inside the left edge of the first continuation's window.
    const uint32_t prefix = window + 270;
    if (params.n_ctx <= prefix + 32) { return false; }
    const llama_pos next_pos = prefix - 15;
    std::vector<llama_pos> rows(params.n_ctx);
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i] = i < 256 ? i : i < 272 ? 256 : i - 15;
    }
    const std::vector<llama_pos> expected(rows.begin(), rows.begin() + prefix);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto decode = [&](llama_context * ctx, uint32_t begin, uint32_t end, llama_seq_id seq = 0) {
        if (end > rows.size()) {
            fprintf(stderr, "Media SWA decode exceeds test context: %u > %zu\n", end, rows.size());
            return false;
        }
        for (uint32_t start = begin; start < end; start += 64) {
            auto batch = llama_batch_init(std::min(64u, end - start), 0, 1);
            for (uint32_t i = start; i < std::min(start + 64, end); ++i) {
                common_batch_add(batch, llama_token(1 + i % (n_vocab - 1)), rows[i], {seq}, i + 1 == end);
            }
            const bool ok = llama_decode(ctx, batch) == 0;
            llama_batch_free(batch);
            if (!ok) { return false; }
        }
        llama_synchronize(ctx);
        return true;
    };
    const bool live = params.vbr_dynamic();
    for (bool remove_source : {false, true}) {
        std::vector<uint8_t> reference_image;
        std::vector<float> initial_reference;
        std::vector<float> reference_logits;
        for (int arm = 0; arm < 3; ++arm) {
            auto ctx = make_ctx(params, model, 3);
            if (!ctx) { return false; }
            auto * mem = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx.get()));
            if (!mem || !decode(ctx.get(), 0, prefix)) {
                fprintf(stderr, "Media SWA initial fill failed\n");
                return false;
            }
            const auto partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
            const auto companion = live ? std::vector<uint8_t>() : save_seq(ctx.get(), 0, partial);
            const uint32_t ahead = live ? prefix + 512 : mem->get_swa()->get_size() + window + 256;
            if (!decode(ctx.get(), prefix, ahead) ||
                (!live && mem->get_swa()->seq_pos_min(0) <= next_pos - llama_pos(window))) {
                fprintf(stderr, "Media SWA window advance failed: min=%d next=%d ahead=%u\n",
                        mem->get_swa()->seq_pos_min(0), next_pos, ahead);
                return false;
            }
            // Legacy serialization deliberately rejects a retiered VBR cache.
            // Use its source-lineage contract plus exact live continuation instead.
            const auto source = live ? std::vector<uint8_t>() : save_seq(ctx.get(), 0);
            const auto lineage_before = llama_memory_vbr_state(mem, 0, 0);
            const auto tier_before = mem->vbr_representation_identity();
            if (arm == 0) {
                if (live) { mem->seq_cp(0, 1, 0, next_pos); }
                else { mem->get_base()->seq_cp(0, 1, 0, next_pos); }
            } else {
                const bool ok = live ? mem->try_share_live_prefix_rows(0, 1, next_pos, expected)
                                     : mem->try_share_attn_prefix_rows(0, 1, next_pos, expected);
                if (!ok) {
                    fprintf(stderr, "Media SWA checked sharing refused\n");
                    return false;
                }
            }
            if (!live && (companion.empty() || llama_state_seq_set_data_ext(ctx.get(), companion.data(),
                    companion.size(), 1, partial) != companion.size())) {
                fprintf(stderr, "Media SWA companion restore failed\n");
                return false;
            }
            const auto restored = live ? std::vector<uint8_t>() : save_seq(ctx.get(), 1);
            const auto lineage_after = llama_memory_vbr_state(mem, 0, 0);
            const auto tier_shared = mem->vbr_representation_identity();
            if ((!live && (source != save_seq(ctx.get(), 0) || restored.empty())) ||
                lineage_before.checkpoint_epoch != lineage_after.checkpoint_epoch ||
                lineage_before.checkpoint_epoch_swa != lineage_after.checkpoint_epoch_swa ||
                tier_before.tier_epoch != tier_shared.tier_epoch ||
                tier_before.tier_epoch_swa != tier_shared.tier_epoch_swa ||
                mem->try_share_live_prefix_rows(0, 1, next_pos, expected) ||
                mem->try_share_attn_prefix_rows(0, 1, next_pos, expected)) {
                fprintf(stderr, "Media SWA source image or occupied-destination check failed\n");
                return false;
            }
            if (arm == 0) { reference_image = restored; }
            else if (restored != reference_image) {
                fprintf(stderr, "Media SWA shared image differs from established-copy control\n");
                return false;
            }
            if (!decode(ctx.get(), prefix, prefix + 1, 1)) { return false; }
            const auto initial = copy_logits(ctx.get(), n_vocab);
            if (arm == 0) { initial_reference = initial; }
            else if (!logits_equal(initial_reference, initial, "Media SWA first destination row", 0.0f)) {
                return false;
            }
            const uint32_t advanced = live ? ahead + window + 512 : ahead;
            if (!decode(ctx.get(), ahead, advanced)) { return false; }
            const auto tier_after = mem->vbr_representation_identity();
            if (live && (tier_after.tier_epoch <= tier_before.tier_epoch ||
                         tier_after.tier_epoch_swa <= tier_before.tier_epoch_swa)) {
                fprintf(stderr, "Media SWA gate requires post-sharing retiering of both children\n");
                return false;
            }
            const llama_seq_id removed = remove_source ? 0 : 1;
            const llama_seq_id survivor = 1 - removed;
            if (!mem->seq_rm(removed, -1, -1) || !decode(ctx.get(), 0, 16, removed)) { return false; }
            const uint32_t frontier = remove_source ? prefix + 1 : advanced;
            std::vector<float> logits;
            for (uint32_t i = frontier; i < frontier + 32; ++i) {
                if (!decode(ctx.get(), i, i + 1, survivor)) { return false; }
                const auto row = copy_logits(ctx.get(), n_vocab);
                logits.insert(logits.end(), row.begin(), row.end());
            }
            if (arm == 0) { reference_logits = std::move(logits); }
            else if (!logits_equal(reference_logits, logits, "Media SWA exact replay", 0.0f)) { return false; }
            fprintf(stderr, "Media SWA %s %s-first arm=%d: exact replay; tier=(%llu,%llu)->(%llu,%llu) PASS\n",
                live ? "live" : "historical", remove_source ? "source" : "destination", arm,
                (unsigned long long) tier_before.tier_epoch, (unsigned long long) tier_before.tier_epoch_swa,
                (unsigned long long) tier_after.tier_epoch, (unsigned long long) tier_after.tier_epoch_swa);
        }
    }
    // Recycled image rows must not be mistaken for retained-window coverage.
    auto ctx = make_ctx(params, model, 3);
    if (!ctx) { return false; }
    auto * mem = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx.get()));
    if (!mem || !decode(ctx.get(), 0, mem->get_swa()->get_size() + window + 256) ||
        mem->can_share_live_prefix_rows(0, 1, next_pos, expected) ||
        mem->try_share_live_prefix_rows(0, 1, next_pos, expected) ||
        mem->get_base()->seq_pos_max(1) != -1 || mem->get_swa()->seq_pos_max(1) != -1) { return false; }
    fprintf(stderr, "Media SWA recycled-window no-mutation refusal PASS\n");
    return true;
}

static bool test_share_live_prefix_swa(const common_params & params, llama_model * model) {
    const uint32_t window = llama_model_n_swa(model);
    if (window == 0) { return false; }
    const uint32_t prefix = window + 64;
    const uint32_t ahead = prefix + 512;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(params.n_ctx);
    for (size_t i = 0; i < tokens.size(); ++i) { tokens[i] = llama_token(1 + i % (n_vocab - 1)); }
    const auto fill = [&](llama_context * ctx, uint32_t begin, uint32_t end) {
        if (end > tokens.size()) { return false; }
        for (uint32_t pos = begin; pos < end; pos += 64) {
            if (!decode_range(ctx, tokens, pos, std::min(64u, end - pos))) { return false; }
        }
        llama_synchronize(ctx);
        return true;
    };
    for (bool remove_source : { false, true }) {
        std::vector<float> reference;
        std::vector<float> initial_reference;
        for (int arm = 0; arm < 3; ++arm) {
            auto ctx = make_ctx(params, model, 3);
            if (!ctx) { return false; }
            auto * mem = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx.get()));
            const uint32_t advanced = ahead + window + 512;
            if (!mem || advanced + 32 > tokens.size() || !fill(ctx.get(), 0, ahead)) { return false; }
            const auto before = llama_memory_vbr_state(mem, 0, 0);
            const auto tier = mem->vbr_representation_identity();
            if (arm == 0) {
                // Existing unchecked composite copy is correct here because this
                // control deliberately retains every historical window row.
                mem->seq_cp(0, 1, 0, prefix);
            } else if (!mem->can_share_live_prefix(0, 1, prefix) ||
                       !mem->try_share_live_prefix(0, 1, prefix) ||
                       mem->try_share_live_prefix(0, 1, prefix)) {
                return false;
            }
            const auto after = llama_memory_vbr_state(mem, 0, 0);
            const auto after_tier = mem->vbr_representation_identity();
            if (before.checkpoint_epoch != after.checkpoint_epoch ||
                before.checkpoint_epoch_swa != after.checkpoint_epoch_swa ||
                tier.tier_epoch != after_tier.tier_epoch || tier.tier_epoch_swa != after_tier.tier_epoch_swa) {
                fprintf(stderr, "Live SWA sharing changed source lineage or tiers\n");
                return false;
            }
            if (!decode_range(ctx.get(), tokens, prefix, 1, 1)) { return false; }
            const auto initial = copy_logits(ctx.get(), n_vocab);
            if (arm == 0) {
                initial_reference = initial;
            } else if (!logits_equal(initial_reference, initial, "Live SWA first destination row", 0.0f)) {
                return false;
            }
            if (!fill(ctx.get(), ahead, advanced)) { return false; }
            const auto advanced_tier = mem->vbr_representation_identity();
            if (params.vbr_dynamic() && (!mem->get_swa()->vbr_controller_active() ||
                advanced_tier.tier_epoch == 0 || advanced_tier.tier_epoch_swa == 0)) {
                fprintf(stderr, "Live SWA gate needs actual VBR retiering: (%llu,%llu)->(%llu,%llu)\n",
                    (unsigned long long) after_tier.tier_epoch, (unsigned long long) after_tier.tier_epoch_swa,
                    (unsigned long long) advanced_tier.tier_epoch, (unsigned long long) advanced_tier.tier_epoch_swa);
                return false;
            }
            const llama_seq_id removed = remove_source ? 0 : 1;
            const llama_seq_id survivor = 1 - removed;
            if (!mem->seq_rm(removed, -1, -1) || !decode_range(ctx.get(), tokens, 0, 16, removed)) {
                return false;
            }
            const uint32_t frontier = remove_source ? prefix + 1 : advanced;
            std::vector<float> logits;
            for (uint32_t pos = frontier; pos < frontier + 32; ++pos) {
                if (!decode_range(ctx.get(), tokens, pos, 1, survivor)) { return false; }
                const auto row = copy_logits(ctx.get(), n_vocab);
                logits.insert(logits.end(), row.begin(), row.end());
            }
            if (arm == 0) {
                reference = std::move(logits);
            } else if (!logits_equal(reference, logits, "Live SWA sharing exact replay", 0.0f)) {
                return false;
            }
            fprintf(stderr, "Live SWA prefix %s-first arm=%d exact replay; tier=(%llu,%llu)->(%llu,%llu)\n",
                remove_source ? "source" : "destination", arm,
                (unsigned long long) after_tier.tier_epoch, (unsigned long long) after_tier.tier_epoch_swa,
                (unsigned long long) advanced_tier.tier_epoch, (unsigned long long) advanced_tier.tier_epoch_swa);
        }
    }
    auto ctx = make_ctx(params, model, 3);
    if (!ctx) { return false; }
    auto * mem = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx.get()));
    if (!mem || !fill(ctx.get(), 0, mem->get_swa()->get_size() + window + 256) ||
        mem->get_swa()->seq_pos_min(0) < llama_pos(prefix) ||
        mem->can_share_live_prefix(0, 1, prefix) || mem->try_share_live_prefix(0, 1, prefix) ||
        mem->get_base()->seq_pos_max(1) != -1 || mem->get_swa()->seq_pos_max(1) != -1) {
        fprintf(stderr, "Live SWA expired-window refusal failed\n");
        return false;
    }
    fprintf(stderr, "Live SWA expired-window no-mutation refusal passed\n");
    return true;
}

static bool test_share_attn_prefix_swa(const common_params & params, llama_model * model, bool remove_source) {
    auto ctx = make_ctx(params, model, 3);
    if (!ctx) { return false; }
    auto * mem = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx.get()));
    if (!mem || params.vbr_dynamic() || llama_model_n_swa(model) == 0) {
        fprintf(stderr, "SWA prefix gate requires a fixed-KV interleaved SWA model\n");
        return false;
    }
    const uint32_t window = llama_model_n_swa(model);
    const uint32_t checkpoint_pos = window + 64;
    const uint32_t source_pos = mem->get_swa()->get_size() + window + 256;
    if (source_pos + 32 > llama_n_ctx(ctx.get())) {
        fprintf(stderr, "SWA prefix gate needs context beyond its physical window pool\n");
        return false;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(source_pos + 32);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = llama_token(1 + i % (n_vocab - 1));
    }
    const auto decode_to = [&](uint32_t begin, uint32_t end) {
        for (; begin < end; begin += 64) {
            if (!decode_range(ctx.get(), tokens, begin, std::min(64u, end - begin))) {
                return false;
            }
        }
        llama_synchronize(ctx.get());
        return true;
    };
    if (!decode_to(0, checkpoint_pos)) { return false; }
    const auto historical = save_seq(ctx.get(), 0);
    const auto companion = save_seq(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (historical.empty() || companion.empty() || !decode_to(checkpoint_pos, source_pos) ||
        mem->get_swa()->seq_pos_min(0) < llama_pos(checkpoint_pos)) {
        fprintf(stderr, "SWA gate did not advance the retained window\n");
        return false;
    }
    const auto source = save_seq(ctx.get(), 0);
    if (llama_memory_can_share_attn_prefix(mem, -1, 1, checkpoint_pos) ||
        llama_memory_can_share_attn_prefix(mem, 0, 3, checkpoint_pos) ||
        llama_memory_can_share_attn_prefix(mem, 0, 0, checkpoint_pos) ||
        llama_memory_can_share_attn_prefix(mem, 0, 1, source_pos + 1) ||
        !llama_memory_try_share_attn_prefix(mem, 0, 1, checkpoint_pos)) {
        fprintf(stderr, "SWA historical prefix sharing refused after window advance\n");
        return false;
    }
    const size_t loaded = llama_state_seq_set_data_ext(ctx.get(), companion.data(), companion.size(), 1,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const bool source_equal = source == save_seq(ctx.get(), 0);
    const auto restored = save_seq(ctx.get(), 1);
    if (loaded != companion.size() || !source_equal || restored.empty()) {
        fprintf(stderr, "SWA image check: loaded=%zu/%zu source_equal=%d size=%zu/%zu\n",
            loaded, companion.size(), source_equal, historical.size(), restored.size());
        return false;
    }
    if (llama_memory_try_share_attn_prefix(mem, 0, 1, checkpoint_pos) ||
        source != save_seq(ctx.get(), 0) || restored != save_seq(ctx.get(), 1)) {
        fprintf(stderr, "SWA occupied destination was not a no-mutation refusal\n");
        return false;
    }
    const llama_seq_id removed = remove_source ? 0 : 1;
    const llama_seq_id survivor = remove_source ? 1 : 0;
    const auto & survivor_state = remove_source ? restored : source;
    if (!llama_memory_seq_rm(mem, removed, -1, -1) ||
        !decode_range(ctx.get(), tokens, 0, 32, removed) ||
        survivor_state != save_seq(ctx.get(), survivor)) {
        fprintf(stderr, "SWA removal/reuse changed the surviving sequence\n");
        return false;
    }
    const uint32_t frontier = remove_source ? checkpoint_pos : source_pos;
    for (uint32_t pos = frontier; pos < frontier + 32; ++pos) {
        if (!decode_range(ctx.get(), tokens, pos, 1, survivor)) { return false; }
        const auto row = copy_logits(ctx.get(), n_vocab);
        if (!logits_equal(row, row, "SWA survivor finite continuation", 0.0f)) { return false; }
    }
    // Compare in canonical placement: physical attention reduction order need
    // not be identical between a shared prefix and an independently loaded one.
    ctx.reset();
    std::vector<float> reference;
    std::vector<uint8_t> canonical;
    for (const auto * state : { &historical, &restored, &restored }) {
        auto replay = make_ctx(params, model, 3);
        if (!replay || !load_seq(replay.get(), *state, 0)) { return false; }
        // Normalize the sequence IDs inside every serialized KV cell as well
        // as the outer envelope before comparing complete images.
        const auto image = save_seq(replay.get(), 0);
        if (canonical.empty()) {
            canonical = image;
        } else if (canonical != image) {
            fprintf(stderr, "SWA canonical historical image mismatch\n");
            return false;
        }
        std::vector<float> logits;
        for (uint32_t pos = checkpoint_pos; pos < checkpoint_pos + 32; ++pos) {
            if (!decode_range(replay.get(), tokens, pos, 1)) { return false; }
            const auto row = copy_logits(replay.get(), n_vocab);
            logits.insert(logits.end(), row.begin(), row.end());
        }
        if (reference.empty()) {
            reference = std::move(logits);
        } else if (!logits_equal(reference, logits, "SWA canonical historical replay", 0.0f)) {
            return false;
        }
    }
    fprintf(stderr, "SWA historical prefix: %s-first removal/reuse, exact image, unchanged source, exact 32-row replay and repeat\n",
            remove_source ? "source" : "destination");
    return true;
}

static bool test_share_attn_prefix(
        llama_context * ctx, llama_context * ref, const std::vector<llama_token> & tokens, int n_vocab) {
    // Out-of-order physical cells are fine; duplicate logical positions are not.
    llama_kv_cells coverage;
    coverage.resize(4);
    for (uint32_t i = 0; i < 4; ++i) {
        coverage.pos_set(i, 3 - i);
        coverage.seq_add(i, 0);
    }
    if (!coverage.seq_has_range(0, 1, 3) || coverage.seq_has_range(0, -1, 2) ||
        coverage.seq_has_range(0, 3, 5) || !coverage.seq_has_prefix(0, 4) || coverage.seq_has_prefix(0, 5) ||
        coverage.seq_has_prefix(1, 1) || coverage.seq_has_prefix(0, 0)) {
        return false;
    }
    coverage.rm(0);
    coverage.pos_set(0, 2);
    coverage.seq_add(0, 0);
    if (coverage.seq_has_prefix(0, 4) || coverage.seq_has_prefix(0, 3) ||
        !coverage.seq_has_prefix(0, 2)) {
        return false;
    }
    auto mem = llama_get_memory(ctx);
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem);
    if (!hybrid || dynamic_cast<llama_memory_hybrid_idx *>(mem)) {
        return !llama_memory_try_share_attn_prefix(mem, 0, 1, 4);
    }
    auto * attn = hybrid->get_mem_attn();
    auto * recurrent = hybrid->get_mem_recr();
    constexpr auto partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const auto reset = [&]() {
        llama_synchronize(ctx);
        llama_synchronize(ref);
        llama_memory_clear(mem, true);
        llama_memory_clear(llama_get_memory(ref), true);
    };
    const auto share = [&](llama_seq_id src, llama_seq_id dst, llama_pos count) {
        llama_synchronize(ctx);
        const bool ready = llama_memory_can_share_attn_prefix(mem, src, dst, count);
        const bool copied = llama_memory_try_share_attn_prefix(mem, src, dst, count);
        GGML_ASSERT(ready == copied);
        return copied;
    };
    const auto restore_partial = [&](const std::vector<uint8_t> & state) {
        return !state.empty() && llama_state_seq_set_data_ext(
                ctx, state.data(), state.size(), 1, partial) == state.size();
    };

    // Source is ahead of the historical checkpoint. Destination has its own
    // recurrent row but no attention, proving the primitive does not copy RS.
    reset();
    if (!decode_range(ctx, tokens, 0, 4)) {
        return false;
    }
    if (mem->can_share_live_prefix(0, 1, 4) || mem->try_share_live_prefix(0, 1, 4)) {
        fprintf(stderr, "Companion-free sharing must refuse recurrent hybrids\n");
        return false;
    }
    const auto checkpoint = save_seq(ctx, 0, partial);
    // Independent control: the existing composite copy is valid *at* the
    // checkpoint, before the source advances. It creates the same physical row
    // placement without using the new primitive or historical restore path.
    if (!decode_range(ref, tokens, 0, 4)) {
        return false;
    }
    llama_synchronize(ref);
    if (!llama_memory_try_seq_cp(llama_get_memory(ref), 0, 1, -1, -1) ||
        !decode_range(ref, tokens, 4, 4) ||
        !decode_range(ctx, tokens, 4, 4) || !decode_range(ctx, tokens, 0, 2, 1)) {
        return false;
    }
    const auto source = save_seq(ctx, 0);
    const auto occupied = save_seq(ctx, 1);
    if (share(0, 1, 4) || occupied != save_seq(ctx, 1) ||
        !mem->seq_rm_attn(1, -1, -1)) {
        return false;
    }
    const auto independent_rs = save_seq(ctx, 1, partial);
    const auto source_depth = recurrent->rollback_valid_depth[0];
    const auto destination_depth = recurrent->rollback_valid_depth[1];
    for (const auto count : { -1, 0, 9 }) {
        if (share(0, 1, count)) {
            return false;
        }
    }
    if (share(0, 0, 4) || share(-1, 1, 4) || share(0, LLAMA_MAX_SEQ, 4) ||
        source != save_seq(ctx, 0) || independent_rs != save_seq(ctx, 1, partial) ||
        attn->seq_pos_max(1) != -1 || !share(0, 1, 4) ||
        source != save_seq(ctx, 0) || independent_rs != save_seq(ctx, 1, partial) ||
        recurrent->rollback_valid_depth[0] != source_depth ||
        recurrent->rollback_valid_depth[1] != destination_depth || attn->seq_pos_max(1) != 3) {
        return false;
    }
    const auto & cells = attn->get_cells(0);
    uint32_t shared = 0;
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_has(i, 1)) {
            if (!cells.seq_has(i, 0) || cells.pos_get(i) >= 4) {
                return false;
            }
            ++shared;
        }
    }
    if (shared != 4 || !restore_partial(checkpoint) ||
        recurrent->cells[0].tail == recurrent->cells[1].tail ||
        recurrent->seq_pos_max(0) != 7 || recurrent->seq_pos_max(1) != 3 ||
        source != save_seq(ctx, 0)) {
        return false;
    }

    // Same batch geometry and physical KV layout for control and destination.
    // A verify/rollback on the destination must leave the active source intact.
    if (!decode_range(ctx, tokens, 4, 3, 1) || !decode_range(ref, tokens, 4, 3, 1) ||
        !logits_equal(copy_logits(ctx, n_vocab, -1), copy_logits(ref, n_vocab, -1), "shared prefix verify", 0.0f) ||
        !mem->seq_rm(1, 6, -1) || !llama_memory_seq_rm(llama_get_memory(ref), 1, 6, -1) ||
        !decode_range(ctx, tokens, 6, 1, 1) || !decode_range(ref, tokens, 6, 1, 1) ||
        !logits_equal(copy_logits(ctx, n_vocab), copy_logits(ref, n_vocab), "shared prefix rollback", 0.0f) ||
        source != save_seq(ctx, 0)) {
        return false;
    }
    // Removing the source must not free the shared prefix out from under A2.
    if (!mem->seq_rm(0, -1, -1) || !llama_memory_seq_rm(llama_get_memory(ref), 0, -1, -1) ||
        !decode_range(ctx, tokens, 7, 1, 1) || !decode_range(ref, tokens, 7, 1, 1) ||
        !logits_equal(copy_logits(ctx, n_vocab), copy_logits(ref, n_vocab), "source cleared first", 0.0f)) {
        return false;
    }

    // Failure after sharing: a truncated historical image must fail, and the
    // caller can discard only A2 before retrying. Also exercise A2-first removal.
    reset();
    if (!load_seq(ctx, source, 0) || !load_seq(ref, source, 0) || !share(0, 1, 4)) {
        return false;
    }
    auto truncated = checkpoint;
    truncated.resize(truncated.size() / 2);
    if (restore_partial(truncated) || !mem->seq_rm(1, -1, -1) ||
        source != save_seq(ctx, 0) || !share(0, 1, 4) || !restore_partial(checkpoint) ||
        !mem->seq_rm(1, -1, -1) || source != save_seq(ctx, 0) ||
        !decode_range(ctx, tokens, 8, 1) || !decode_range(ref, tokens, 8, 1) ||
        !logits_equal(copy_logits(ctx, n_vocab), copy_logits(ref, n_vocab), "destination cleared first", 0.0f)) {
        return false;
    }
    // A source with a hole still has plausible min/max bounds, but is not a hit.
    if (!mem->seq_rm_attn(0, 2, 3)) {
        return false;
    }
    const auto gappy = save_seq(ctx, 0);
    if (share(0, 1, 4) || gappy != save_seq(ctx, 0) || attn->seq_pos_max(1) != -1) {
        return false;
    }
    reset();
    fprintf(stderr, "%s : attention sharing, independent RS, rollback and cleanup passed\n", __func__);
    return true;
}

// Compare late historical sharing with the existing full sequence copy taken
// at the checkpoint. Run contexts sequentially: independent VBR trees must not
// compete for one process's co-tenancy offer or change the control's budget.
static bool test_share_attn_prefix_vbr(const common_params & params, llama_model * model) {
    const uint32_t checkpoint_pos = params.n_ctx / 8;
    const uint32_t source_pos = params.n_ctx * 3 / 8;
    constexpr uint32_t replay = 32;
    constexpr auto partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(source_pos + replay);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = (llama_token) ((i * 17 + 1) % n_vocab);
    }
    std::vector<float> reference;
    for (bool source_first : { false, true }) {
        for (int arm = 0; arm < 3; ++arm) {
            // arm 0: existing composite copy before source advances;
            // arms 1/2: late attention share + historical RS, including a repeat.
            auto ctx = make_ctx(params, model, 2);
            if (!ctx) { return false; }
            auto mem = llama_get_memory(ctx.get());
            auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem);
            if (!hybrid || !hybrid->get_mem_attn()->vbr_controller_active()) {
                fprintf(stderr, "VBR sharing gate requires an armed ordinary hybrid\n");
                return false;
            }
            auto * attn = hybrid->get_mem_attn();
            const auto prefill = [&](uint32_t begin, uint32_t end) {
                for (uint32_t pos = begin; pos < end; pos += 64) {
                    if (!decode_range(ctx.get(), tokens, pos, std::min(64u, end - pos))) { return false; }
                }
                llama_synchronize(ctx.get());
                return true;
            };
            if (!prefill(0, checkpoint_pos)) { return false; }
            const auto checkpoint = save_seq(ctx.get(), 0, partial);
            const auto epoch = attn->vbr_checkpoint_epoch(0);
            const auto checkpoint_tier = attn->vbr_tier_epoch();
            if (checkpoint.empty() || (arm == 0 && !llama_memory_try_seq_cp(mem, 0, 1, -1, -1)) ||
                !prefill(checkpoint_pos, source_pos)) { return false; }
            const auto source = save_seq(ctx.get(), 0, partial);
            const auto tier = attn->vbr_tier_epoch();
            if (epoch != attn->vbr_checkpoint_epoch(0) || tier <= checkpoint_tier) {
                fprintf(stderr, "VBR gate needs a retiered, unchanged source lineage: epoch=%llu -> %llu tier=%llu\n",
                        (unsigned long long) epoch, (unsigned long long) attn->vbr_checkpoint_epoch(0),
                        (unsigned long long) tier);
                return false;
            }
            if (arm != 0) {
                if (llama_memory_can_share_attn_prefix(mem, 0, 1, source_pos + 1) ||
                    !llama_memory_try_share_attn_prefix(mem, 0, 1, checkpoint_pos) ||
                    llama_memory_try_share_attn_prefix(mem, 0, 1, checkpoint_pos) ||
                    llama_state_seq_set_data_ext(ctx.get(), checkpoint.data(), checkpoint.size(),
                        1, partial) != checkpoint.size()) { return false; }
            }
            if (source != save_seq(ctx.get(), 0, partial) ||
                epoch != attn->vbr_checkpoint_epoch(0) || tier != attn->vbr_tier_epoch()) {
                fprintf(stderr, "sharing changed source recurrent state, lineage or KV tiers\n");
                return false;
            }
            uint32_t shared = 0;
            const auto & cells = attn->get_cells(0);
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (cells.seq_has(i, 1)) {
                    if (!cells.seq_has(i, 0) || cells.pos_get(i) >= (llama_pos) checkpoint_pos) { return false; }
                    ++shared;
                }
            }
            if (shared != checkpoint_pos || (source_first && !llama_memory_seq_rm(mem, 0, -1, -1))) {
                return false;
            }
            std::vector<float> logits;
            for (uint32_t pos = checkpoint_pos; pos < checkpoint_pos + replay; ++pos) {
                if (!decode_range(ctx.get(), tokens, pos, 1, 1)) { return false; }
                const auto row = copy_logits(ctx.get(), n_vocab);
                logits.insert(logits.end(), row.begin(), row.end());
            }
            if (!llama_memory_seq_rm(mem, 1, -1, -1)) { return false; }
            if (!source_first) {
                if (source != save_seq(ctx.get(), 0, partial) ||
                    !decode_range(ctx.get(), tokens, source_pos, 1)) { return false; }
                const auto row = copy_logits(ctx.get(), n_vocab);
                logits.insert(logits.end(), row.begin(), row.end());
            }
            if (arm == 0) { reference = logits; }
            if (!logits_equal(reference, logits, "VBR historical share", 0.0f)) { return false; }
            llama_memory_clear(mem, true);
            if (attn->seq_pos_max(0) != -1 || attn->seq_pos_max(1) != -1) { return false; }
            fprintf(stderr, "VBR share: source_first=%d arm=%d exact logits=%zu tier_epoch=%llu->%llu PASS\n",
                    source_first, arm, logits.size(), (unsigned long long) checkpoint_tier,
                    (unsigned long long) tier);
        }
    }
    return true;
}

static bool test_nonfinite_reset(llama_context * ctx, const std::vector<llama_token> & tokens, int n_vocab) {
    auto * recurrent = get_recurrent(ctx);
    if (recurrent == nullptr) {
        return false;
    }
    auto mem = llama_get_memory(ctx);
    for (uint32_t count : { 1u, 4u }) {
        llama_memory_clear(mem, true);
        if (!decode_range(ctx, tokens, 0, count)) {
            return false;
        }
        const auto reference = copy_logits(ctx, n_vocab, -1);
        llama_synchronize(ctx);
        // A reset must overwrite old state, not multiply it by zero: NaN and infinity
        // otherwise survive, poisoning every later request until the context is recreated.
        for (const auto * planes : { &recurrent->r_l, &recurrent->s_l, &recurrent->p_l }) {
            for (auto * tensor : *planes) {
                if (tensor == nullptr) {
                    continue;
                }
                GGML_ASSERT(tensor->type == GGML_TYPE_F32);
                std::vector<float> poison(tensor->ne[0]);
                for (size_t i = 0; i < poison.size(); ++i) {
                    poison[i] = i % 3 == 0 ? std::numeric_limits<float>::quiet_NaN() :
                                i % 3 == 1 ? std::numeric_limits<float>::infinity() :
                                             -std::numeric_limits<float>::infinity();
                }
                ggml_backend_tensor_set(tensor, poison.data(), 0, poison.size() * sizeof(float));
            }
        }
        if (!llama_memory_seq_rm(mem, 0, 0, -1) || !decode_range(ctx, tokens, 0, count) ||
            !logits_equal(reference, copy_logits(ctx, n_vocab, -1), "nonfinite recurrent reset")) {
            return false;
        }
    }
    llama_memory_clear(mem, true);
    return true;
}

static bool abort_decode(void *) {
    return true;
}

static void set_resize_test_fault(bool enabled) {
#ifdef _WIN32
    _putenv_s("LLAMA_RECURRENT_RESIZE_TEST_FAIL", enabled ? "before_publish" : "");
#else
    if (enabled) {
        setenv("LLAMA_RECURRENT_RESIZE_TEST_FAIL", "before_publish", 1);
    } else {
        unsetenv("LLAMA_RECURRENT_RESIZE_TEST_FAIL");
    }
#endif
}

static bool get_recurrent_epoch(llama_memory_recurrent * recurrent, uint64_t & epoch) {
    auto memory_context = recurrent->init_full();
    auto * recurrent_context = dynamic_cast<llama_memory_recurrent_context *>(memory_context.get());
    if (recurrent_context == nullptr) {
        return false;
    }
    epoch = recurrent_context->get_tensor_binding_epoch();
    return true;
}

static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab) {
    constexpr uint32_t  n_seqs     = 2;
    // Recurrent split keeps the rollback tail together; the ubatch must be
    // larger than n_rs_seq while remaining smaller than n_replay so this still
    // exercises multiple ubatches per sequence.
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return init_ctx(model, cparams);
    };

    auto ctx_roll = make_ctx_multi();
    auto ctx_ref  = make_ctx_multi();
    if (!ctx_roll || !ctx_ref) {
        fprintf(stderr, "%s : failed to init multi-seq contexts\n", __func__);
        return false;
    }

    if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
        fprintf(stderr, "%s : n_rs_seq is too small for split replay\n", __func__);
        return false;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        // The rollback plane retains one prior state per transition within a
        // decode batch. Decode the rollback context's boundary token together
        // with its tail so a three-token rollback has three authenticated
        // snapshots; the reference consumes that boundary token separately.
        for (llama_pos pos = 0; pos < p0 - 1; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll.get(), batch) == 0;
        ok = ok && llama_decode(ctx_ref.get(),  batch) == 0;

        common_batch_clear(batch);
        common_batch_add(batch, tok(s, p0 - 1), p0 - 1, { (llama_seq_id) s }, false);
        ok = ok && llama_decode(ctx_ref.get(), batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p0 - 1; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll.get(), batch) == 0;
        llama_batch_free(batch);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        fprintf(stderr, "%s : multi-seq prefill/rollback failed\n", __func__);
        return false;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll.get(), batch) == 0;
    ok = ok && llama_decode(ctx_ref.get(), batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        fprintf(stderr, "%s : multi-seq replay decode failed\n", __func__);
        return false;
    }

    // identical ubatch shapes from bit-exact states: a correct implementation
    // matches bitwise, so eps only allows backend scheduling noise
    constexpr float eps = 1e-7f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing multi-seq logits at index %u\n", __func__, i);
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float diff = logit_diff(l_roll[t], l_ref[t]);
            if (diff > eps && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
        }
    }

    if (diff_max > eps) {
        fprintf(stderr, "%s : multi-seq split replay logits mismatch (max diff %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, seq_first, pos_first);
        return false;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (max diff %g)\n", __func__, (double) diff_max);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref.get(), batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll.get(), batch_one) == 0;
        ok = ok && llama_decode(ctx_ref.get(), batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            diff_tail = std::max(diff_tail, logit_diff(l_roll[t], l_ref[t]));
        }
    }

    if (!ok || diff_tail > eps) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (ok=%d, max diff %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail);
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g)\n", __func__, (double) diff_tail);
    return true;
}

static bool test_indexed_hybrid_tree_collection(const llama_model & model) {
    const llama_memory_i::layer_filter_cb reject_all = [](int32_t) { return false; };

    // No Qwen4 fixture is needed to pin the topology decision: empty layer
    // filters construct only the composite type, without allocating model
    // payload tensors. The derived-type branch must run before the generic
    // llama_memory_hybrid branch or the QSA owner would be omitted.
    llama_memory_hybrid_idx indexed(
        model,
        GGML_TYPE_F16, GGML_TYPE_F16, false, 8, 1, 0, LLAMA_SWA_TYPE_NONE,
        GGML_TYPE_F32, GGML_TYPE_F32, 1,
        1, 1, false, false,
        reject_all, reject_all, reject_all);

    if (llama_memory_try_share_attn_prefix(&indexed, 0, 1, 4)) {
        fprintf(stderr, "%s : indexed hybrid accepted attention-only sharing\n", __func__);
        return false;
    }

    std::vector<llama_memory_tree_child> tree;
    if (!llama_memory_tree_collect(&indexed, tree) || tree.size() != 2 ||
        tree[0].child_id != 0 || tree[0].attention != indexed.get_mem_attn() ||
        tree[0].recurrent != nullptr || tree[0].qsa_index_owner != &indexed ||
        tree[0].dependency_mode != checkpoint_child_dependency_mode::live_guarded ||
        tree[1].child_id != 1 || tree[1].attention != nullptr ||
        tree[1].recurrent != indexed.get_mem_recr() || tree[1].qsa_index_owner != nullptr ||
        tree[1].dependency_mode != checkpoint_child_dependency_mode::absent) {
        fprintf(stderr, "%s : indexed hybrid topology was incomplete or misclassified\n", __func__);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    set_resize_test_fault(false);

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    std::vector<std::string> parser_args(argv, argv + argc);
    const auto fill_arg = std::find(parser_args.begin(), parser_args.end(), "--cache-pattern-fill");
    cache_pattern_fill = fill_arg != parser_args.end();
    if (cache_pattern_fill) {
        parser_args.erase(fill_arg);
    }
    const auto shared_swa_arg = std::find(parser_args.begin(), parser_args.end(), "--shared-swa-wrap-only");
    const bool shared_swa_only = shared_swa_arg != parser_args.end();
    if (shared_swa_only) {
        parser_args.erase(shared_swa_arg);
        parser_args.push_back("--attn-prefix-only");
    }
    const auto media_swa_arg = std::find(parser_args.begin(), parser_args.end(), "--media-swa-prefix-only");
    const bool media_swa_only = media_swa_arg != parser_args.end();
    if (media_swa_only) {
        parser_args.erase(media_swa_arg);
        parser_args.push_back("--attn-prefix-only");
    }
    const auto media_arg = std::find(parser_args.begin(), parser_args.end(), "--media-prefix-only");
    const bool media_only = media_arg != parser_args.end();
    if (media_only) {
        parser_args.erase(media_arg);
        parser_args.push_back("--attn-prefix-only");
    }
    const auto live_arg = std::find(parser_args.begin(), parser_args.end(), "--live-swa-prefix-only");
    const bool live_swa_only = live_arg != parser_args.end();
    if (live_swa_only) {
        parser_args.erase(live_arg);
        parser_args.push_back("--attn-prefix-only");
    }
    const auto swa_arg = std::find(parser_args.begin(), parser_args.end(), "--swa-prefix-only");
    const bool swa_prefix_only = swa_arg != parser_args.end();
    if (swa_prefix_only) {
        parser_args.erase(swa_arg);
        parser_args.push_back("--attn-prefix-only");
    }
    const auto prefix_arg = std::find(parser_args.begin(), parser_args.end(), "--attn-prefix-only");
    const bool prefix_only = prefix_arg != parser_args.end();
    if (prefix_only) {
        parser_args.erase(prefix_arg);
    } else {
        parser_args.push_back("-ct");
        parser_args.push_back("f16");
    }
    std::vector<char *> parser_argv;
    parser_argv.reserve(parser_args.size());
    for (auto & arg : parser_args) {
        parser_argv.push_back(&arg[0]);
    }
    if (!common_params_parse(
            int(parser_argv.size()), parser_argv.data(), params,
            LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // This rollback test is cache-representation agnostic and intentionally
    // pins the static CPU-compatible cache rather than inheriting CLI defaults.
    GGML_ASSERT(prefix_only || params.cache_type_k == GGML_TYPE_F16);
    GGML_ASSERT(prefix_only || params.cache_type_v == GGML_TYPE_F16);
    GGML_ASSERT(prefix_only || !params.vbr_enabled());

    // The production MTP/VBR server uses unified KV. Its single physical stream
    // must remain distinct from the logical multi-sequence graph capacity.
    params.kv_unified = true;

    ggml_backend_load_all();

    if (shared_swa_only) {
        auto mparams = common_model_params_to_llama(params);
        llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
        return model && test_shared_swa_wrap(params, model.get()) ? 0 : 1;
    }

    if (media_swa_only) {
        auto mparams = common_model_params_to_llama(params);
        llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
        return model && test_share_media_swa_prefix(params, model.get()) ? 0 : 1;
    }

    if (media_only) {
        auto mparams = common_model_params_to_llama(params);
        llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
        return model && test_share_media_prefix(params, model.get()) ? 0 : 1;
    }

    if (live_swa_only) {
        auto mparams = common_model_params_to_llama(params);
        llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
        return model && test_share_live_prefix_swa(params, model.get()) ? 0 : 1;
    }

    if (swa_prefix_only) {
        auto mparams = common_model_params_to_llama(params);
        llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
        return model && test_share_attn_prefix_swa(params, model.get(), true) &&
            test_share_attn_prefix_swa(params, model.get(), false) ? 0 : 1;
    }

    if (params.vbr_dynamic()) {
        if (params.n_ctx < 2048 || params.vbr_vram_budget_bytes == 0) {
            fprintf(stderr, "VBR sharing gate needs -c >= 2048 and an explicit --vbr-vram budget\n");
            return 1;
        }
        auto mparams = common_model_params_to_llama(params);
        llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
        return model && test_share_attn_prefix_vbr(params, model.get()) ? 0 : 1;
    }

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!test_indexed_hybrid_tree_collection(*model)) {
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(32);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = (llama_token) ((i + 1) % std::max(n_vocab, 1));
    }

    auto ctx_src      = make_ctx(params, model);
    auto ctx_test     = make_ctx(params, model);
    auto ctx_ref      = make_ctx(params, model);
    auto ctx_parallel = make_ctx(params, model, 3);
    if (!ctx_src || !ctx_test || !ctx_ref || !ctx_parallel) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }
    {
        auto prefix_ref = make_ctx(params, model, 3);
        if (!prefix_ref || !test_share_attn_prefix(ctx_parallel.get(), prefix_ref.get(), tokens, n_vocab)) {
            fprintf(stderr, "%s : attention prefix sharing failed\n", __func__);
            return 1;
        }
    }
    if (prefix_only) {
        return 0;
    }
    if (!test_nonfinite_reset(ctx_test.get(), tokens, n_vocab)) {
        fprintf(stderr, "%s : nonfinite recurrent reset failed\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s : nonfinite recurrent reset checks passed\n", __func__);

    // A unified attention cache owns one physical stream while remaining logically
    // capable of the configured sequence count. Composite lowering may narrow that
    // logical cap without changing the physical stream layout.
    llama_memory_t parallel_mem = llama_get_memory(ctx_parallel.get());
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(parallel_mem)) {
        llama_kv_cache_context unrestricted(hybrid->get_mem_attn());
        llama_kv_cache_context limited(hybrid->get_mem_attn(), 2);
        if (unrestricted.get_max_graph_seqs() != std::numeric_limits<uint32_t>::max() ||
            limited.get_max_graph_seqs() != 2) {
            fprintf(stderr, "%s : unified attention logical capacity was confused with its physical stream\n", __func__);
            return 1;
        }
    } else if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(parallel_mem)) {
        llama_kv_cache_iswa_context unrestricted(hybrid->get_mem_attn());
        llama_kv_cache_iswa_context limited(hybrid->get_mem_attn(), 2);
        if (unrestricted.get_max_graph_seqs() != std::numeric_limits<uint32_t>::max() ||
            limited.get_max_graph_seqs() != 2) {
            fprintf(stderr, "%s : unified iSWA attention logical capacity was confused with its physical stream\n", __func__);
            return 1;
        }
    }

    auto * recurrent = get_recurrent(ctx_test.get());
    if (recurrent == nullptr || recurrent->n_rs_seq < 3) {
        fprintf(stderr, "%s : skipping because recurrent rollback depth is less than 3\n", __func__);
        return 0;
    }
    const uint32_t n_rs_seq = recurrent->n_rs_seq;

    // Starting a graph invalidates the planes that graph is about to overwrite, but a
    // rollback selected by the preceding seq_rm() is an input to that graph. Keep its
    // selector alive until s_copy() consumes it while constructing the graph.
    {
        llama_seq_id seq_id = 0;
        llama_seq_id * seq_ids[] = { &seq_id };
        int32_t n_seq_id[] = { 1 };
        llama_ubatch ubatch = {};
        ubatch.b_equal_seqs = 1;
        ubatch.n_tokens = 1;
        ubatch.n_seq_tokens = 1;
        ubatch.n_seqs = 1;
        ubatch.n_seq_id = n_seq_id;
        ubatch.seq_id = seq_ids;

        recurrent->set_rs_idx(0, 2);
        recurrent->rollback_valid_depth[0] = 2;
        recurrent->invalidate_rollback(ubatch);
        if (recurrent->rs_idx[0] != 2 || recurrent->rollback_valid_depth[0] != 0) {
            fprintf(stderr, "%s : graph invalidation discarded a pending rollback selector\n", __func__);
            return 1;
        }
        recurrent->reset_rollback_state(0);
    }

    // Preserve the original regression: a selected rollback plane must serialize as the
    // logical active row and replay identically after restore.
    if (!decode_range(ctx_src.get(), tokens, 0, 4) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, 3, -1)) {
        fprintf(stderr, "%s : rolled-back checkpoint setup failed\n", __func__);
        return 1;
    }
    const auto rolled_back_state = save_seq(ctx_src.get(), 0);
    if (!load_seq(ctx_test.get(), rolled_back_state, 0) ||
        !decode_range(ctx_src.get(),  tokens, 3, 1) ||
        !decode_range(ctx_test.get(), tokens, 3, 1) ||
        !logits_equal(copy_logits(ctx_src.get(), n_vocab), copy_logits(ctx_test.get(), n_vocab),
                "rolled-back checkpoint restore")) {
        fprintf(stderr, "%s : rolled-back checkpoint did not replay identically\n", __func__);
        return 1;
    }
    llama_memory_clear(llama_get_memory(ctx_src.get()),  true);
    llama_memory_clear(llama_get_memory(ctx_test.get()), true);

    // A restore installs only the active row. A following one-token decode writes group 0,
    // so rollback one must fail without changing serialized state or the current logits.
    if (!decode_range(ctx_src.get(), tokens, 0, 3)) {
        fprintf(stderr, "%s : source prefix decode failed\n", __func__);
        return 1;
    }
    const auto restored_state = save_seq(ctx_src.get(), 0);
    if (!load_seq(ctx_test.get(), restored_state, 0) ||
        !check_depth(ctx_test.get(), 0, 0, "restore") ||
        !decode_range(ctx_test.get(), tokens, 3, 1) ||
        !check_depth(ctx_test.get(), 0, 0, "restore then one-token decode")) {
        fprintf(stderr, "%s : restore setup failed\n", __func__);
        return 1;
    }

    const auto state_before_rm  = save_seq(ctx_test.get(), 0);
    const auto logits_before_rm = copy_logits(ctx_test.get(), n_vocab);
    const int32_t tail_before_rm = get_recurrent(ctx_test.get())->cells[0].tail;
    const uint32_t rs_before_rm  = get_recurrent(ctx_test.get())->rs_idx[0];
    if (!load_seq(ctx_ref.get(), state_before_rm, 0)) {
        fprintf(stderr, "%s : failed to retain pre-rm reference\n", __func__);
        return 1;
    }
    // Exercise the composite memory operation. For hybrid models, a rejected
    // recurrent rollback must not remove the corresponding attention entries.
    if (llama_memory_seq_rm(llama_get_memory(ctx_test.get()), 0, 3, -1)) {
        fprintf(stderr, "%s : stale rollback unexpectedly succeeded after restore + decode\n", __func__);
        return 1;
    }
    const auto state_after_rm  = save_seq(ctx_test.get(), 0);
    const auto logits_after_rm = copy_logits(ctx_test.get(), n_vocab);
    const bool state_unchanged  = state_before_rm == state_after_rm;
    const bool logits_unchanged = logits_equal(logits_before_rm, logits_after_rm, "failed rollback");
    const llama_pos pos_after_rm = llama_memory_seq_pos_max(llama_get_memory(ctx_test.get()), 0);
    const auto * recurrent_after_rm = get_recurrent(ctx_test.get());
    const bool metadata_unchanged = recurrent_after_rm->cells[0].tail == tail_before_rm &&
        recurrent_after_rm->rs_idx[0] == rs_before_rm;
    if (!state_unchanged || !logits_unchanged || !metadata_unchanged || pos_after_rm != 3) {
        fprintf(stderr, "%s : failed-rm details: state=%s, logits=%s, metadata=%s, pos=%d\n",
                __func__, state_unchanged ? "same" : "changed",
                logits_unchanged ? "same" : "changed",
                metadata_unchanged ? "same" : "changed", pos_after_rm);
        fprintf(stderr, "%s : failed rollback mutated state, logits, or position\n", __func__);
        return 1;
    }
    if (!decode_range(ctx_test.get(), tokens, 4, 1) ||
        !decode_range(ctx_ref.get(),  tokens, 4, 1) ||
        !logits_equal(copy_logits(ctx_test.get(), n_vocab), copy_logits(ctx_ref.get(), n_vocab),
                "post-failed-rm continuation")) {
        fprintf(stderr, "%s : failed rollback changed continuation\n", __func__);
        return 1;
    }

    // A four-token verify writes the active plane plus rollback planes 1..3.
    // Each rollback is checked independently against a retained prefix reference.
    for (uint32_t rollback = 1; rollback <= 3; ++rollback) {
        llama_memory_clear(llama_get_memory(ctx_test.get()), true);
        llama_memory_clear(llama_get_memory(ctx_ref.get()),  true);

        if (!decode_range(ctx_test.get(), tokens, 0, 2) ||
            !decode_range(ctx_test.get(), tokens, 2, 4) ||
            !check_depth(ctx_test.get(), 0, 3, "four-token verify")) {
            fprintf(stderr, "%s : four-token verify setup failed for rollback %u\n", __func__, rollback);
            return 1;
        }

        const uint32_t rollback_pos = 6 - rollback;
        if (!decode_range(ctx_ref.get(), tokens, 0, rollback_pos) ||
            !llama_memory_seq_rm(llama_get_memory(ctx_test.get()), 0, rollback_pos, -1) ||
            !decode_range(ctx_test.get(), tokens, rollback_pos, 1) ||
            !decode_range(ctx_ref.get(),  tokens, rollback_pos, 1) ||
            !logits_equal(copy_logits(ctx_test.get(), n_vocab), copy_logits(ctx_ref.get(), n_vocab),
                    "bounded rollback reference")) {
            fprintf(stderr, "%s : rollback %u did not match retained reference\n", __func__, rollback);
            return 1;
        }
    }

    // A narrow decode replaces, rather than extends, the valid-depth assignment.
    llama_memory_clear(llama_get_memory(ctx_test.get()), true);
    if (!decode_range(ctx_test.get(), tokens, 0, 4) ||
        !check_depth(ctx_test.get(), 0, 3, "verify before narrow decode") ||
        !decode_range(ctx_test.get(), tokens, 4, 1) ||
        !check_depth(ctx_test.get(), 0, 0, "verify then narrow decode")) {
        fprintf(stderr, "%s : verify then narrow decode setup failed\n", __func__);
        return 1;
    }
    const auto narrow_state = save_seq(ctx_test.get(), 0);
    if (get_recurrent(ctx_test.get())->seq_rm(0, 4, -1) ||
        narrow_state != save_seq(ctx_test.get(), 0)) {
        fprintf(stderr, "%s : stale rollback succeeded or mutated after narrow decode\n", __func__);
        return 1;
    }

    // The assignment clamps at n_rs_seq, and full removal is never rejected by the guard.
    llama_memory_clear(llama_get_memory(ctx_test.get()), true);
    if (!decode_range(ctx_test.get(), tokens, 0, n_rs_seq + 1) ||
        !check_depth(ctx_test.get(), 0, n_rs_seq, "clamped verify") ||
        !llama_memory_seq_rm(llama_get_memory(ctx_test.get()), 0, -1, -1) ||
        !check_depth(ctx_test.get(), 0, 0, "full removal")) {
        fprintf(stderr, "%s : clamp or full-removal check failed\n", __func__);
        return 1;
    }

    // A successful detectable copy into an empty destination copies the composite state.
    // Give the empty destination synthetic rollback metadata to prove that the copied active
    // row resets it rather than inheriting the source's valid planes.
    if (!decode_range(ctx_parallel.get(), tokens, 0, 4, 1) ||
        !check_depth(ctx_parallel.get(), 1, 3, "seq_cp source")) {
        fprintf(stderr, "%s : successful seq_cp setup failed\n", __func__);
        return 1;
    }
    const auto source_state = save_seq(ctx_parallel.get(), 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    auto * recurrent_parallel = get_recurrent(ctx_parallel.get());
    recurrent_parallel->set_rs_idx(0, 3);
    recurrent_parallel->rollback_valid_depth[0] = 3;
    const bool copy_succeeded = llama_get_memory(ctx_parallel.get())->try_seq_cp(1, 0, -1, -1);
    const auto destination_state = save_seq(ctx_parallel.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (!copy_succeeded ||
        !seq_state_payload_equal(source_state, destination_state) ||
        !check_depth(ctx_parallel.get(), 0, 0, "seq_cp destination") ||
        !check_depth(ctx_parallel.get(), 1, 3, "seq_cp source")) {
        size_t first_diff = 0;
        while (first_diff < source_state.size() &&
               first_diff < destination_state.size() &&
               source_state[first_diff] == destination_state[first_diff]) {
            ++first_diff;
        }
        fprintf(stderr, "%s : successful seq_cp details: result=%s, source=%zu B, destination=%zu B, first_diff=%zu\n",
                __func__, copy_succeeded ? "true" : "false",
                source_state.size(), destination_state.size(), first_diff);
        fprintf(stderr, "%s : successful seq_cp did not copy state or reset destination depth\n", __func__);
        return 1;
    }

    // Fill every recurrent cell, then exercise the recurrent implementation directly.
    // Reserve-before-clear must report exhaustion while preserving the occupied destination
    // byte-for-byte, including its rollback metadata.
    llama_memory_clear(llama_get_memory(ctx_parallel.get()), true);
    if (!decode_equal_split(ctx_parallel.get(), tokens, 4, 3)) {
        fprintf(stderr, "%s : seq_cp exhaustion setup failed\n", __func__);
        return 1;
    }
    recurrent_parallel = get_recurrent(ctx_parallel.get());
    const auto exhausted_dst_state = save_seq(ctx_parallel.get(), 0);
    const int32_t exhausted_dst_tail = recurrent_parallel->cells[0].tail;
    const llama_pos exhausted_dst_pos = recurrent_parallel->seq_pos_max(0);
    const uint32_t exhausted_dst_rs_idx = recurrent_parallel->rs_idx[0];
    const uint32_t exhausted_dst_depth = recurrent_parallel->rollback_valid_depth[0];
    const uint32_t exhausted_used = recurrent_parallel->used;
    if (recurrent_parallel->try_seq_cp(1, 0, -1, -1) ||
        exhausted_dst_state != save_seq(ctx_parallel.get(), 0) ||
        recurrent_parallel->cells[0].tail != exhausted_dst_tail ||
        recurrent_parallel->seq_pos_max(0) != exhausted_dst_pos ||
        recurrent_parallel->rs_idx[0] != exhausted_dst_rs_idx ||
        recurrent_parallel->rollback_valid_depth[0] != exhausted_dst_depth ||
        recurrent_parallel->used != exhausted_used) {
        fprintf(stderr, "%s : exhausted recurrent seq_cp succeeded or mutated destination\n", __func__);
        return 1;
    }
    if (!recurrent_parallel->seq_rm(0, -1, -1) ||
        !recurrent_parallel->try_seq_cp(1, 0, -1, -1) ||
        recurrent_parallel->cells[0].tail < 0 ||
        recurrent_parallel->used != exhausted_used) {
        fprintf(stderr, "%s : full-pool backup restore did not reuse the freed live cell\n", __func__);
        return 1;
    }

    // The hybrid fallback cannot roll back the attention copy, so detected failure
    // invalidates both children. The destination is empty and therefore never selectable
    // as a half-copied composite checkpoint.
    llama_memory_t mem_parallel = llama_get_memory(ctx_parallel.get());
    if (dynamic_cast<llama_memory_hybrid *>(mem_parallel) != nullptr ||
        dynamic_cast<llama_memory_hybrid_iswa *>(mem_parallel) != nullptr) {
        if (mem_parallel->try_seq_cp(1, 0, -1, -1) ||
            recurrent_parallel->get_cell_count(0) != 0 ||
            recurrent_parallel->seq_pos_max(0) != -1 ||
            get_attention_pos_max(ctx_parallel.get(), 0) != -1 ||
            llama_memory_seq_pos_max(mem_parallel, 0) != -1 ||
            !check_depth(ctx_parallel.get(), 0, 0, "failed hybrid seq_cp destination")) {
            fprintf(stderr, "%s : failed hybrid seq_cp left an incoherent destination\n", __func__);
            return 1;
        }
    }

    // Equal-split batching still publishes each participating sequence independently.
    llama_memory_clear(llama_get_memory(ctx_parallel.get()), true);
    if (!decode_equal_split(ctx_parallel.get(), tokens, 4, 2) ||
        !check_depth(ctx_parallel.get(), 0, 3, "equal split seq 0") ||
        !check_depth(ctx_parallel.get(), 1, 3, "equal split seq 1") ||
        !decode_range(ctx_parallel.get(), tokens, 4, 1, 0) ||
        !check_depth(ctx_parallel.get(), 0, 0, "narrow seq 0") ||
        !check_depth(ctx_parallel.get(), 1, 3, "untouched seq 1")) {
        fprintf(stderr, "%s : per-sequence equal-split assignment failed\n", __func__);
        return 1;
    }

    // A live logical sequence may occupy a high physical cell. Shrinking must
    // refuse atomically instead of truncating that recurrent state while leaving
    // the attention half reusable.
    llama_memory_clear(mem_parallel, true);
    llama_memory_clear(llama_get_memory(ctx_ref.get()), true);
    if (!llama_memory_recurrent_expand(mem_parallel, 3) ||
        !decode_range(ctx_parallel.get(), tokens, 0, 1, 1) ||
        !decode_range(ctx_parallel.get(), tokens, 0, 1, 2) ||
        !decode_range(ctx_parallel.get(), tokens, 0, 1, 0) ||
        !decode_range(ctx_ref.get(), tokens, 0, 1, 0) ||
        !llama_memory_seq_rm(mem_parallel, 1, -1, -1) ||
        !llama_memory_seq_rm(mem_parallel, 2, -1, -1)) {
        fprintf(stderr, "%s : high-cell shrink setup failed\n", __func__);
        return 1;
    }
    recurrent_parallel = get_recurrent(ctx_parallel.get());
    const int32_t high_tail = recurrent_parallel->cells[0].tail;
    const auto high_state_before = save_seq(ctx_parallel.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const auto high_r_before = recurrent_parallel->r_l;
    const auto high_s_before = recurrent_parallel->s_l;
    const auto high_p_before = recurrent_parallel->p_l;
    uint64_t high_epoch_before = 0;
    uint64_t high_epoch_after = 0;
    if (high_tail < 1 || high_state_before.empty() ||
        !get_recurrent_epoch(recurrent_parallel, high_epoch_before) ||
        llama_memory_recurrent_shrink(mem_parallel, 1) ||
        recurrent_parallel->size != 3 ||
        recurrent_parallel->cells[0].tail != high_tail ||
        recurrent_parallel->r_l != high_r_before ||
        recurrent_parallel->s_l != high_s_before ||
        recurrent_parallel->p_l != high_p_before ||
        save_seq(ctx_parallel.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != high_state_before ||
        !get_recurrent_epoch(recurrent_parallel, high_epoch_after) ||
        high_epoch_after != high_epoch_before ||
        !decode_range(ctx_parallel.get(), tokens, 1, 1, 0) ||
        !decode_range(ctx_ref.get(), tokens, 1, 1, 0) ||
        !logits_equal(copy_logits(ctx_parallel.get(), n_vocab), copy_logits(ctx_ref.get(), n_vocab),
                "post-refused-high-cell-shrink continuation")) {
        fprintf(stderr, "%s : high-cell shrink was not failure-atomic\n", __func__);
        return 1;
    }

    // A resize failure after all replacement tensors and metadata have been staged must
    // leave the live cache completely untouched and usable. A following successful resize
    // must publish one new binding epoch and preserve the active recurrent row.
    llama_memory_clear(mem_parallel, true);
    llama_memory_clear(llama_get_memory(ctx_ref.get()), true);
    const uint32_t pre_zero_shrink_size = recurrent_parallel->size;
    if (llama_memory_recurrent_shrink(mem_parallel, 0) ||
        recurrent_parallel->size != pre_zero_shrink_size ||
        !llama_memory_recurrent_shrink(mem_parallel, 1) ||
        !decode_range(ctx_parallel.get(), tokens, 0, 4) ||
        !decode_range(ctx_ref.get(), tokens, 0, 4)) {
        fprintf(stderr, "%s : resize failure-atomic setup failed\n", __func__);
        return 1;
    }

    recurrent_parallel = get_recurrent(ctx_parallel.get());
    const auto contracted_mctx = mem_parallel->init_full();
    if (!contracted_mctx || contracted_mctx->get_max_graph_seqs() != 1 ||
        llama_graph_reserve(ctx_parallel.get(), 2, 2, 2) != nullptr ||
        llama_graph_reserve(ctx_parallel.get(), 1, 1, 1) == nullptr) {
        fprintf(stderr, "%s : contracted recurrent graph capacity was not enforced\n", __func__);
        return 1;
    }

    // Force the production sched_reserve() owner through a real topology setter while
    // recurrent storage is contracted. It must reserve the one-sequence graph rather
    // than rebuilding the old configured two-sequence shape.
    llama_set_causal_attn(ctx_parallel.get(), false);
    llama_set_causal_attn(ctx_parallel.get(), true);
    llama_set_causal_attn(ctx_ref.get(), false);
    llama_set_causal_attn(ctx_ref.get(), true);
    if (!decode_range(ctx_parallel.get(), tokens, 4, 1) ||
        !decode_range(ctx_ref.get(), tokens, 4, 1) ||
        !logits_equal(copy_logits(ctx_parallel.get(), n_vocab), copy_logits(ctx_ref.get(), n_vocab),
                "contracted scheduler reserve continuation")) {
        fprintf(stderr, "%s : contracted production scheduler reserve failed\n", __func__);
        return 1;
    }

    const auto resize_state_before = save_seq(ctx_parallel.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const auto resize_r_before = recurrent_parallel->r_l;
    const auto resize_s_before = recurrent_parallel->s_l;
    const auto resize_p_before = recurrent_parallel->p_l;
    const auto resize_rs_idx_before = recurrent_parallel->rs_idx;
    const auto resize_depth_before = recurrent_parallel->rollback_valid_depth;
    const uint32_t resize_size_before = recurrent_parallel->size;
    const uint32_t resize_head_before = recurrent_parallel->head;
    const uint32_t resize_used_before = recurrent_parallel->used;
    const uint32_t resize_n_before = recurrent_parallel->n;
    const int32_t resize_rs_z_before = recurrent_parallel->rs_z;
    uint64_t resize_epoch_before = 0;
    if (resize_state_before.empty() || !get_recurrent_epoch(recurrent_parallel, resize_epoch_before)) {
        fprintf(stderr, "%s : failed to snapshot recurrent resize state\n", __func__);
        return 1;
    }

    set_resize_test_fault(true);
    const bool injected_resize_succeeded = llama_memory_recurrent_expand(mem_parallel, 2);
    set_resize_test_fault(false);

    uint64_t resize_epoch_after_failure = 0;
    if (injected_resize_succeeded ||
        recurrent_parallel->size != resize_size_before ||
        recurrent_parallel->head != resize_head_before ||
        recurrent_parallel->used != resize_used_before ||
        recurrent_parallel->n != resize_n_before ||
        recurrent_parallel->rs_z != resize_rs_z_before ||
        recurrent_parallel->r_l != resize_r_before ||
        recurrent_parallel->s_l != resize_s_before ||
        recurrent_parallel->p_l != resize_p_before ||
        recurrent_parallel->rs_idx != resize_rs_idx_before ||
        recurrent_parallel->rollback_valid_depth != resize_depth_before ||
        save_seq(ctx_parallel.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != resize_state_before ||
        !get_recurrent_epoch(recurrent_parallel, resize_epoch_after_failure) ||
        resize_epoch_after_failure != resize_epoch_before) {
        fprintf(stderr, "%s : failed recurrent resize mutated live state or bindings\n", __func__);
        return 1;
    }

    if (!decode_range(ctx_parallel.get(), tokens, 5, 1) ||
        !decode_range(ctx_ref.get(), tokens, 5, 1) ||
        !logits_equal(copy_logits(ctx_parallel.get(), n_vocab), copy_logits(ctx_ref.get(), n_vocab),
                "post-failed-resize continuation")) {
        fprintf(stderr, "%s : failed recurrent resize broke continuation\n", __func__);
        return 1;
    }

    if (!llama_memory_recurrent_expand(mem_parallel, 2)) {
        fprintf(stderr, "%s : recurrent resize did not recover after injected failure\n", __func__);
        return 1;
    }
    const auto expanded_mctx = mem_parallel->init_full();
    if (!expanded_mctx || expanded_mctx->get_max_graph_seqs() != 2 ||
        llama_graph_reserve(ctx_parallel.get(), 2, 2, 2) == nullptr) {
        fprintf(stderr, "%s : unified hybrid graph capacity did not follow recurrent expansion\n", __func__);
        return 1;
    }
    uint64_t resize_epoch_after_success = 0;
    const uint64_t expected_resize_epoch = resize_epoch_before == UINT64_MAX ? 1 : resize_epoch_before + 1;
    bool tensors_rebound = false;
    for (size_t i = 0; i < resize_r_before.size(); ++i) {
        tensors_rebound = tensors_rebound ||
            (resize_r_before[i] != nullptr && resize_r_before[i] != recurrent_parallel->r_l[i]) ||
            (resize_s_before[i] != nullptr && resize_s_before[i] != recurrent_parallel->s_l[i]) ||
            (resize_p_before[i] != nullptr && resize_p_before[i] != recurrent_parallel->p_l[i]);
    }
    if (recurrent_parallel->size != 2 || !tensors_rebound ||
        !get_recurrent_epoch(recurrent_parallel, resize_epoch_after_success) ||
        resize_epoch_after_success != expected_resize_epoch ||
        !decode_range(ctx_parallel.get(), tokens, 6, 1) ||
        !decode_range(ctx_ref.get(), tokens, 6, 1) ||
        !logits_equal(copy_logits(ctx_parallel.get(), n_vocab), copy_logits(ctx_ref.get(), n_vocab),
                "post-successful-resize continuation")) {
        fprintf(stderr, "%s : successful recurrent resize did not publish one usable binding epoch\n", __func__);
        return 1;
    }

    // Invalidation happens before graph submission, so an aborted decode cannot leave a
    // positive depth that is not backed by a successfully committed write.
    llama_memory_clear(llama_get_memory(ctx_test.get()), true);
    if (!decode_range(ctx_test.get(), tokens, 0, 4) ||
        !check_depth(ctx_test.get(), 0, 3, "pre-abort verify")) {
        fprintf(stderr, "%s : abort setup failed\n", __func__);
        return 1;
    }
    llama_set_abort_callback(ctx_test.get(), abort_decode, nullptr);
    llama_batch abort_batch = llama_batch_init(1, 0, 1);
    common_batch_add(abort_batch, tokens[4], 4, { 0 }, true);
    const int32_t abort_result = llama_decode(ctx_test.get(), abort_batch);
    llama_batch_free(abort_batch);
    llama_set_abort_callback(ctx_test.get(), nullptr, nullptr);
    if (abort_result != 2 || !check_depth(ctx_test.get(), 0, 0, "aborted decode")) {
        fprintf(stderr, "%s : aborted decode returned %d or retained positive depth\n",
                __func__, abort_result);
        return 1;
    }

    if (!test_multi_seq_split_replay(params, model, n_vocab)) {
        return 1;
    }

    fprintf(stderr, "%s : recurrent rollback-plane validity checks passed\n", __func__);
    return 0;
}
