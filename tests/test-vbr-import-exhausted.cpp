// A sequence state import into a dynamic-VBR cache grows the cache's physical backing itself.
// With the device exhausted it has to fail like any other rejected import: nothing installed,
// what the cache held before still there byte for byte (another sequence, and the prefix a
// range is appended to), the context still usable, and the same import accepted once memory
// is back.
//
// Run with a model whose KV dynamic VBR can manage:
//   test-vbr-import-exhausted -m model.gguf -ctk vbr -fa on -ngl 99
// Exits 77 where there is no GPU or the controller did not arm.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <vector>

static constexpr int    SKIP        = 77;
static constexpr size_t MIB         = 1024*1024;
static constexpr size_t LEAVE       = 4*MIB;
static constexpr int    N_PROMPT    = 4096;
static constexpr int    N_PREFIX    = 1024; // of the prompt, live before the rest is appended
static constexpr int    N_BYSTANDER = 1024; // sequence 1, which no import here touches

static bool decode_range(llama_context * ctx, llama_batch & batch, llama_seq_id seq, int p0, int p1) {
    const int n_batch = llama_n_batch(ctx);
    for (int i = p0; i < p1; i += n_batch) {
        common_batch_clear(batch);
        for (int j = i; j < std::min(i + n_batch, p1); ++j) {
            common_batch_add(batch, 1 + (j + 7*seq) % 1000, j, {seq}, j + 1 == p1);
        }
        if (llama_decode(ctx, batch)) {
            return false;
        }
    }
    return true;
}

static size_t free_bytes(ggml_backend_dev_t dev) {
    size_t free, total;
    ggml_backend_dev_memory(dev, &free, &total);
    return free;
}

// take the device's free memory down to LEAVE: whole pieces while they fit, then halves
static std::vector<ggml_backend_buffer_t> exhaust(ggml_backend_dev_t dev) {
    std::vector<ggml_backend_buffer_t> held;
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    for (size_t piece = 256*MIB; piece >= 2*MIB;) {
        ggml_backend_buffer_t buf = free_bytes(dev) < LEAVE + piece ? nullptr : ggml_backend_buft_alloc_buffer(buft, piece);
        if (buf == nullptr) {
            piece /= 2;
            continue;
        }
        held.push_back(buf);
    }
    return held;
}

// a short sequence in the prompt's place lets the mapped watermark fall back
static bool drop_prompt(llama_context * ctx, llama_batch & batch) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, -1, -1);
    const bool ok = decode_range(ctx, batch, 0, 0, 8);
    llama_memory_seq_rm(mem, 0, -1, -1);
    return ok;
}

// run `import` with the device exhausted: it has to be refused
static bool starved(const char * name, ggml_backend_dev_t dev, const std::function<size_t()> & import) {
    const size_t free_before = free_bytes(dev);
    std::vector<ggml_backend_buffer_t> held = exhaust(dev);
    fprintf(stderr, "%s : free device memory %zu -> %zu MiB\n", name, free_before/MIB, free_bytes(dev)/MIB);

    const size_t n_read = import();

    for (ggml_backend_buffer_t buf : held) {
        ggml_backend_buffer_free(buf);
    }
    if (n_read != 0) {
        fprintf(stderr, "%s : FAILED - the import was not starved (returned %zu), nothing was tested\n", name, n_read);
    }
    return n_read == 0;
}

static std::vector<uint8_t> rows(llama_context * ctx, llama_seq_id seq, int p0, int p1) {
    std::vector<uint8_t> out(llama_state_seq_get_size_range(ctx, seq, p0, p1));
    if (llama_state_seq_get_data_range(ctx, out.data(), out.size(), seq, p0, p1) != out.size()) {
        out.clear();
    }
    return out;
}

// `import` installs the rest of the saved prompt into sequence 0, on top of `prefix` (its first
// N_PREFIX tokens) when there is one, and returns the bytes it read. `continues` is whether what
// it installs is all that the model needs to decode on: a range is not on a recurrent or
// sliding-window model.
static bool check_import(const char * name, llama_context * ctx, llama_batch & batch, ggml_backend_dev_t dev,
        const std::vector<uint8_t> & range, const std::vector<uint8_t> * prefix, size_t n_expected,
        bool continues, const std::function<size_t()> & import) {
    llama_memory_t mem = llama_get_memory(ctx);

    const int n_prefix = prefix ? N_PREFIX : 0;

    if (!drop_prompt(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode the short sequence\n", name);
        return false;
    }
    if (prefix &&
        llama_state_seq_append_data(ctx, prefix->data(), prefix->size(), 0, 0, n_prefix, n_prefix) != prefix->size()) {
        fprintf(stderr, "%s : failed to place the prefix\n", name);
        return false;
    }
    // where a range does not continue, the memory reports no position for one either
    const llama_pos            pos_before       = llama_memory_seq_pos_max(mem, 0);
    const llama_pos            pos_bystander    = llama_memory_seq_pos_max(mem, 1);
    const std::vector<uint8_t> bystander_before = rows(ctx, 1, 0, N_BYSTANDER);
    if (bystander_before.empty() || (prefix && rows(ctx, 0, 0, n_prefix) != *prefix) ||
        (continues && pos_before != n_prefix - 1)) {
        fprintf(stderr, "%s : failed to read back what the cache holds\n", name);
        return false;
    }

    fprintf(stderr, "%s : import is %zu MiB on %d tokens\n", name, n_expected/MIB, n_prefix);
    if (!starved(name, dev, import)) {
        return false;
    }
    if (llama_memory_seq_pos_max(mem, 0) != pos_before) {
        fprintf(stderr, "%s : FAILED - the refused import left sequence 0 at %d, not %d\n", name,
                (int) llama_memory_seq_pos_max(mem, 0), (int) pos_before);
        return false;
    }
    if (prefix && rows(ctx, 0, 0, n_prefix) != *prefix) {
        fprintf(stderr, "%s : FAILED - the refused import changed the prefix it was appended to\n", name);
        return false;
    }
    if (llama_memory_seq_pos_max(mem, 1) != pos_bystander || rows(ctx, 1, 0, N_BYSTANDER) != bystander_before) {
        fprintf(stderr, "%s : FAILED - the refused import changed the other sequence\n", name);
        return false;
    }
    // the other sequence goes on, a token longer after every check: its decode is the boundary
    // where the controller settles the refusal, and it comes first so that the token takes a
    // cell below the import's
    if (!decode_range(ctx, batch, 1, pos_bystander + 1, pos_bystander + 2)) {
        fprintf(stderr, "%s : FAILED - the other sequence does not continue\n", name);
        return false;
    }
    if (import() != n_expected) {
        fprintf(stderr, "%s : FAILED - the import was refused with memory available again\n", name);
        return false;
    }
    if (continues && (llama_memory_seq_pos_max(mem, 0) != N_PROMPT - 1 || !decode_range(ctx, batch, 0, N_PROMPT, N_PROMPT + 1))) {
        fprintf(stderr, "%s : FAILED - the restored sequence does not continue\n", name);
        return false;
    }
    if (rows(ctx, 0, 0, N_PROMPT) != range || rows(ctx, 1, 0, N_BYSTANDER) != bystander_before) {
        fprintf(stderr, "%s : FAILED - the cache does not hold what was imported beside what it held\n", name);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    params.kv_unified = true;
    params.n_parallel = 2;
    params.n_ctx      = 2*N_PROMPT;
    params.n_batch    = 512;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        fprintf(stderr, "%s : no GPU device, skipping\n", __func__);
        return SKIP;
    }

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_context * ctx = llama_init->context();
    if (ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, 1);

    // the other sequence first: it takes the low cells, so the watermark can fall back to it
    if (!decode_range(ctx, batch, 1, 0, N_BYSTANDER) || !decode_range(ctx, batch, 0, 0, N_PROMPT)) {
        fprintf(stderr, "%s : failed to decode the prompts\n", __func__);
        return 1;
    }
    if (llama_memory_vbr_state(llama_get_memory(ctx), 0, 0).checkpoint_epoch == 0) {
        fprintf(stderr, "%s : no dynamic-VBR controller on this cache, skipping\n", __func__);
        return SKIP;
    }

    std::vector<uint8_t> whole(llama_state_seq_get_size(ctx, 0));
    const std::vector<uint8_t> range  = rows(ctx, 0, 0, N_PROMPT);
    const std::vector<uint8_t> prefix = rows(ctx, 0, 0, N_PREFIX);
    const std::vector<uint8_t> rest   = rows(ctx, 0, N_PREFIX, N_PROMPT);
    if (llama_state_seq_get_data(ctx, whole.data(), whole.size(), 0) != whole.size() ||
        range.empty() || prefix.empty() || rest.empty()) {
        fprintf(stderr, "%s : failed to save the sequence\n", __func__);
        return 1;
    }

    const llama_model * model = llama_init->model();
    const bool range_continues =
        !llama_model_is_recurrent(model) && !llama_model_is_hybrid(model) && llama_model_n_swa(model) == 0;

    const auto import_whole = [&]() {
        return llama_state_seq_set_data(ctx, whole.data(), whole.size(), 0);
    };

    const bool ok =
        check_import("whole", ctx, batch, dev, range, nullptr, whole.size(), true, import_whole) &&
        check_import("range", ctx, batch, dev, range, nullptr, range.size(), range_continues, [&]() {
            return llama_state_seq_append_data(ctx, range.data(), range.size(), 0, 0, N_PROMPT, N_PROMPT);
        }) &&
        check_import("range on a prefix", ctx, batch, dev, range, &prefix, rest.size(), range_continues, [&]() {
            return llama_state_seq_append_data(ctx, rest.data(), rest.size(), 0, N_PREFIX, N_PROMPT, N_PROMPT);
        }) &&
        // the last refusal meets no further decode: the context is freed with it, as a server
        // that shuts down after a refused restore frees it
        drop_prompt(ctx, batch) &&
        starved("freed after a refusal", dev, import_whole);

    llama_batch_free(batch);

    if (!ok) {
        return 1;
    }
    llama_init.reset();

    fprintf(stderr, "%s : SUCCESS - exhausted imports are refused, what the cache held stays, the context keeps working\n", __func__);
    return 0;
}
