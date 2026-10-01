// What a restored sliding-window sequence owes, and what it does not.
//
// Owed, asserted here: every K/V row saved by each route equals the live cache's row
// at the same position, bit for bit (including older held rows on the ranged route);
// and two live runs of the same batches give the same finite logits.
// Not owed, reported here: the logits of the same next tokens. Cells land at other indices than
// the live ring had them, and the attention kernels reduce over cells in index order. The `+M`
// arms move an exact restore by M cells and nothing else, which is the control for that effect.
//
// Run by hand with a sliding-window model:
//   test-state-restore-swa-exact -m gemma.gguf -ngl 99 -fa on [-ctk ... -ctv ...]
// Exits 77 where the model has no sliding window.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "llama-io.h"
#include "llama-kv-cache-iswa.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static constexpr int SKIP     = 77;
static constexpr int N_PROBE  = 8;
static constexpr int N_OFFSET = 200;

static constexpr llama_state_seq_flags TAIL_FLAGS =
    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_SWA_HELD_CELLS;

struct row_collector : llama_io_write_i {
    std::vector<uint8_t> buf;

    void write(const void * src, size_t size) override {
        buf.insert(buf.end(), (const uint8_t *) src, (const uint8_t *) src + size);
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t at = buf.size();
        buf.resize(at + size);
        ggml_backend_tensor_get(tensor, buf.data() + at, offset, size);
    }
    size_t n_bytes() override { return buf.size(); }
};

struct arm {
    std::string          name;
    llama_context_ptr    ctx;
    std::vector<float>   logits; // N_PROBE * n_vocab
    bool                held = true;
};

// Canonicalize physical placement without discarding older held cells. Use the production
// row serializers; the ordinary range API intentionally masks those cells and cannot test them.
struct llama_kv_cache_state_test {
    static std::vector<uint8_t> swa_rows(const llama_kv_cache * cache, bool held) {
        cache->state_write_prepare();
        const uint32_t stream = cache->seq_to_stream[0];
        const auto & cells = cache->v_cells[stream];
        std::vector<std::pair<llama_pos, uint32_t>> order;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cache->state_write_includes_cell(cells, i, 0, held)) {
                order.emplace_back(cells.pos_get(i), i);
            }
        }
        std::sort(order.begin(), order.end());
        GGML_ASSERT(!order.empty());
        llama_kv_cache::cell_ranges_t cr { stream, {} };
        for (const auto & cell : order) {
            if (!cr.data.empty() && cr.data.back().second == cell.second) {
                cr.data.back().second++;
            } else {
                cr.data.emplace_back(cell.second, cell.second + 1);
            }
        }
        row_collector out;
        const uint32_t count = order.size();
        out.write(&count, sizeof(count));
        cache->state_write_meta(out, cr, 0);
        cache->state_write_data(out, cr);
        return out.buf;
    }
};

static bool decode(llama_context * ctx, const std::vector<llama_token> & tokens, int p0, int p1, llama_seq_id seq, int n_batch) {
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    bool ok = true;
    for (int i = p0; ok && i < p1; i += n_batch) {
        common_batch_clear(batch);
        for (int j = i; j < std::min(i + n_batch, p1); ++j) {
            common_batch_add(batch, tokens[j], j, {seq}, j + 1 == p1);
        }
        ok = llama_decode(ctx, batch) == 0;
    }
    llama_batch_free(batch);
    return ok;
}

static llama_kv_cache_iswa * iswa_of(llama_context * ctx) {
    return dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx));
}

static std::vector<uint8_t> rows(const llama_kv_cache * cache, llama_pos p0, llama_pos p1) {
    row_collector out;
    cache->state_write_range(out, 0, p0, p1);
    return out.buf;
}

int main(int argc, char ** argv) {
    common_params params;

    params.kv_unified = true;
    params.n_parallel = 2;
    params.n_ctx      = 8192;
    params.n_batch    = 512;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model * model = llama_init->model();
    if (model == nullptr || llama_init->context() == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }
    const int n_swa = llama_model_n_swa(model);
    if (n_swa <= 0 || iswa_of(llama_init->context()) == nullptr) {
        fprintf(stderr, "%s : no sliding-window cache, skipping\n", __func__);
        return SKIP;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_batch = params.n_batch;

    std::string text;
    while (text.size() < 64u*params.n_ctx) {
        text += "A river crosses the valley. The notes describe its history, and the mountain above it. ";
    }
    const std::vector<llama_token> tokens = common_tokenize(llama_init->context(), text, true);

    const llama_context_params cparams = common_context_params_to_llama(params);
    const int wrapped_n = std::min<int>(4*n_swa + 137, params.n_ctx - N_OFFSET - N_PROBE - n_batch);
    if (wrapped_n <= n_swa) {
        fprintf(stderr, "FAILED - increase context size to exercise a wrapped window\n");
        return 1;
    }

    bool ok = true;

    // under the window the restored layout is the live one; over it the live ring has wrapped
    for (const int n : { n_swa/2, wrapped_n }) {
        GGML_ASSERT((int) tokens.size() >= n + N_PROBE);
        fprintf(stderr, "\n== %d tokens, window %d ==\n", n, n_swa);

        std::vector<arm> arms;
        auto add_live = [&](const char * name) {
            arm a = { name, llama_context_ptr(llama_init_from_model(model, cparams)), {} };
            GGML_ASSERT(a.ctx && decode(a.ctx.get(), tokens, 0, n, 0, n_batch));
            arms.push_back(std::move(a));
        };
        add_live("live");
        add_live("live again");

        llama_context * live = arms[0].ctx.get();

        std::vector<uint8_t> whole(llama_state_seq_get_size(live, 0));
        std::vector<uint8_t> base (llama_state_seq_get_size_range(live, 0, 0, n));
        std::vector<uint8_t> tail (llama_state_seq_get_size_ext(live, 0, TAIL_FLAGS));
        GGML_ASSERT(llama_state_seq_get_data      (live, whole.data(), whole.size(), 0)             == whole.size());
        GGML_ASSERT(llama_state_seq_get_data_range(live, base.data(),  base.size(),  0, 0, n)       == base.size());
        GGML_ASSERT(llama_state_seq_get_data_ext  (live, tail.data(),  tail.size(),  0, TAIL_FLAGS) == tail.size());

        // offset > 0: another sequence holds the first cells while this one is placed, then leaves.
        // Over the window the held-cells tail fills the window cache, so the ranged +M arm has no room.
        auto add_restored = [&](const std::string & name, bool ranged, int offset) {
            arm a = { name, llama_context_ptr(llama_init_from_model(model, cparams)), {} };
            a.held = ranged;
            llama_context * ctx = a.ctx.get();
            GGML_ASSERT(ctx);
            if (offset > 0) {
                GGML_ASSERT(decode(ctx, tokens, 0, offset, 1, n_batch));
            }
            const bool placed = ranged
                ? llama_state_seq_append_data(ctx, base.data(), base.size(), 0, 0, n, n) == base.size() &&
                  llama_state_seq_set_data_ext(ctx, tail.data(), tail.size(), 0, TAIL_FLAGS) == tail.size()
                : llama_state_seq_set_data(ctx, whole.data(), whole.size(), 0) == whole.size();
            if (!placed) {
                const bool optional = ranged && offset > 0 && n > n_swa;
                fprintf(stderr, "%-22s could not be placed: %s\n", name.c_str(),
                        optional ? "optional wrapped offset control skipped" : "FAILED required restore");
                ok = ok && optional;
                return;
            }
            llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1);
            if (ranged) {
                const auto * restored_swa = iswa_of(ctx)->get_swa();
                const auto * live_swa = iswa_of(live)->get_swa();
                ok = ok && restored_swa->seq_pos_min(0) == live_swa->seq_pos_min(0) &&
                           restored_swa->seq_pos_max(0) == live_swa->seq_pos_max(0);
            }
            arms.push_back(std::move(a));
        };
        add_restored("restored whole",     false, 0);
        add_restored("restored ranged",    true,  0);
        add_restored("restored whole +M",  false, N_OFFSET);
        add_restored("restored ranged +M", true,  N_OFFSET);

        // Whole-state restore owes the normal window; range+held-tail owes all held cells.
        const std::vector<uint8_t> base_rows = rows(iswa_of(live)->get_base(), 0, n);
        const auto window_rows = llama_kv_cache_state_test::swa_rows(iswa_of(live)->get_swa(), false);
        const auto held_rows = llama_kv_cache_state_test::swa_rows(iswa_of(live)->get_swa(), true);
        if (n > n_swa && held_rows.size() <= window_rows.size()) {
            fprintf(stderr, "FAILED - wrapped control has no older held rows to test\n");
            ok = false;
        }
        fprintf(stderr, "rows compared: base [0, %d) %zu bytes, window %zu bytes, all held %zu bytes\n",
                n, base_rows.size(), window_rows.size(), held_rows.size());

        for (arm & a : arms) {
            llama_kv_cache_iswa * mem = iswa_of(a.ctx.get());
            const bool same_base = rows(mem->get_base(), 0, n)       == base_rows;
            const bool same_swa = llama_kv_cache_state_test::swa_rows(mem->get_swa(), a.held) ==
                                  (a.held ? held_rows : window_rows);
            const llama_pos held = mem->get_swa()->seq_pos_min(0);

            a.logits.resize((size_t) N_PROBE*n_vocab);
            for (int i = 0; i < N_PROBE; ++i) {
                GGML_ASSERT(decode(a.ctx.get(), tokens, n + i, n + i + 1, 0, n_batch));
                std::memcpy(a.logits.data() + (size_t) i*n_vocab, llama_get_logits_ith(a.ctx.get(), -1), n_vocab*sizeof(float));
            }

            // against the live arm: same-token logits at every probe
            const std::vector<float> & ref = arms[0].logits;
            const bool finite = std::all_of(a.logits.begin(), a.logits.end(),
                                            [](float x) { return std::isfinite(x); });
            if (!finite) {
                fprintf(stderr, "%s: FAILED nonfinite logits\n", a.name.c_str());
                ok = false;
            }
            float  max_abs   = 0.0f;
            double max_kld   = 0.0;
            int    n_top1    = 0;
            for (int i = 0; i < N_PROBE; ++i) {
                const float * p = ref.data()      + (size_t) i*n_vocab;
                const float * q = a.logits.data() + (size_t) i*n_vocab;
                double zp = 0.0, zq = 0.0;
                const float * top_p = std::max_element(p, p + n_vocab);
                const float * top_q = std::max_element(q, q + n_vocab);
                const float mp = *top_p;
                const float mq = *top_q;
                for (int v = 0; v < n_vocab; ++v) {
                    max_abs = std::max(max_abs, std::fabs(p[v] - q[v]));
                    zp += std::exp((double) p[v] - mp);
                    zq += std::exp((double) q[v] - mq);
                }
                double kld = 0.0;
                for (int v = 0; v < n_vocab; ++v) {
                    const double lp = p[v] - mp - std::log(zp);
                    kld += std::exp(lp)*(lp - (q[v] - mq - std::log(zq)));
                }
                max_kld = std::max(max_kld, kld);
                n_top1 += top_p - p == top_q - q;
            }
            const bool same_logits = std::memcmp(ref.data(), a.logits.data(), ref.size()*sizeof(float)) == 0;

            fprintf(stderr, "%-22s window cache from %5d | rows base %s window %s | logits %s max|d| %.4g max KLD %.3g top-1 %d/%d\n",
                    a.name.c_str(), held, same_base ? "EXACT" : "DIFFER", same_swa ? "EXACT" : "DIFFER",
                    same_logits ? "EXACT" : "differ", max_abs, max_kld, n_top1, N_PROBE);

            ok = ok && same_base && same_swa && (a.name != "live again" || same_logits);
        }
    }

    fprintf(stderr, "\n%s\n", ok ? "SUCCESS - restored rows are the live rows and live runs repeat"
                                 : "FAILED - a restored row differs or live runs do not repeat");
    return ok ? 0 : 1;
}
