#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-adapter.h"
#include "llama-cpp.h"
#include "llama-context.h"
#include "llama-ext.h"
#include "llama-graph.h"
#include "llama-model-loader.h"
#include "llama-memory.h"
#include "llama-model.h"
#include "llama-safetensors-qwen3.h"
#include "models/dflash-selector-family.h"
#include "speculative.h"

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using json = llama_safetensors_json;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

struct temporary_directory {
    std::filesystem::path path;

    temporary_directory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned i = 0; i < 100; ++i) {
            auto candidate = std::filesystem::temp_directory_path() /
                ("llama-native-dflash-" + std::to_string(stamp) + "-" + std::to_string(i));
            if (std::filesystem::create_directory(candidate)) {
                path = std::move(candidate);
                return;
            }
        }
        throw std::runtime_error("cannot create native DFlash test directory");
    }

    ~temporary_directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

std::vector<uint8_t> half_bytes(size_t count, unsigned tag, bool input) {
    std::vector<uint8_t> bytes(count * sizeof(ggml_fp16_t));
    for (size_t i = 0; i < count; ++i) {
        // Distinct by role, module, and channel, including negative input signs.
        const float value = input ? ((i + tag) % 2 ? -1.0f : 1.0f) :
            float(16 + tag + i % 7) / 32.0f;
        const ggml_fp16_t half = ggml_fp32_to_fp16(value);
        std::memcpy(bytes.data() + i * sizeof(half), &half, sizeof(half));
    }
    return bytes;
}

struct projection {
    const char * source;
    const char * canonical;
    uint64_t inputs;
    uint64_t outputs;
};

const std::array<projection, 8> projections = {{
    { "fc", "fc", 256, 128 },
    { "layers.0.self_attn.q_proj", "blk.0.attn_q", 128, 256 },
    { "layers.0.self_attn.k_proj", "blk.0.attn_k", 128, 128 },
    { "layers.0.self_attn.v_proj", "blk.0.attn_v", 128, 128 },
    { "layers.0.self_attn.o_proj", "blk.0.attn_output", 256, 128 },
    { "layers.0.mlp.gate_proj", "blk.0.ffn_gate", 128, 256 },
    { "layers.0.mlp.up_proj", "blk.0.ffn_up", 128, 256 },
    { "layers.0.mlp.down_proj", "blk.0.ffn_down", 256, 128 },
}};

json fixture_config(bool sliding) {
    return {
        {"architectures", {"DFlash2DraftModel"}}, {"model_type", "qwen3"},
        {"hidden_size", 128}, {"intermediate_size", 256}, {"num_hidden_layers", 1},
        {"num_attention_heads", 2}, {"num_key_value_heads", 1}, {"head_dim", 128},
        {"num_target_layers", 4}, {"vocab_size", 256}, {"max_position_embeddings", 512},
        {"rope_parameters", {{"rope_theta", 10000000}}}, {"rms_norm_eps", 1e-6},
        {"use_sliding_window", sliding}, {"sliding_window", 256},
        {"layer_types", {sliding ? "sliding_attention" : "full_attention"}},
        {"quantization_config", {{"quant_method", "exl3"}}},
        {"dflash_config", {{"block_size", 4}, {"mask_token_id", 255},
            {"target_layer_ids", {0, 2}}, {"conv_kernel_size", 2}, {"conv_group_size", 16},
            {"selector_rank", 16}, {"selector_top_k", 4}}},
    };
}

void write_fixture(const std::filesystem::path & path, const std::string & prefix,
                   bool sliding, bool standalone) {
    std::filesystem::create_directory(path);
    json header = json::object();
    std::vector<uint8_t> payload;
    auto add = [&](const std::string & name, const char * dtype,
                   const std::vector<uint64_t> & shape, const std::vector<uint8_t> & bytes) {
        const size_t begin = payload.size();
        payload.insert(payload.end(), bytes.begin(), bytes.end());
        header[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {begin, payload.size()}}};
    };
    for (size_t i = 0; i < projections.size(); ++i) {
        const auto & p = projections[i];
        const std::string module = prefix + p.source;
        add(module + ".trellis", "I16", {p.inputs / 16, p.outputs / 16, 64},
            std::vector<uint8_t>(p.inputs * p.outputs / 2, uint8_t(i + 1)));
        add(module + ".suh", "F16", {p.inputs}, half_bytes(p.inputs, i, true));
        add(module + ".svh", "F16", {p.outputs}, half_bytes(p.outputs, i, false));
        add(module + ".mul1", "I32", {}, std::vector<uint8_t>(4));
    }
    for (const char * name : {"hidden_norm.weight", "norm.weight", "layers.0.input_layernorm.weight",
                             "layers.0.post_attention_layernorm.weight", "layers.0.self_attn.q_norm.weight",
                             "layers.0.self_attn.k_norm.weight"}) {
        add(prefix + name, "F16", {128}, half_bytes(128, 0, true));
    }
    for (const char * conv : {"attention_conv", "mlp_conv"}) {
        const std::string module = prefix + "layers.0." + conv;
        add(module + ".base_kernel", "F16", {2, 2, 128}, std::vector<uint8_t>(1024));
        add(module + ".kernel_projection.weight", "F16", {32, 128}, std::vector<uint8_t>(8192));
    }
    add(prefix + "candidate_selector.hidden_projection.weight", "F16", {16, 128},
        half_bytes(16 * 128, 9, false));
    for (const char * name : {"predecessor_codebook", "successor_codebook"}) {
        add(prefix + "candidate_selector." + name + (prefix.empty() ? ".weight" : ""),
            "F16", {256, 16}, half_bytes(256 * 16, 10, false));
    }
    if (standalone) {
        // Optional own embedding/head permits decoder graph reservation without
        // constructing a second target model or borrowing synthetic pointers.
        add(prefix + "embed_tokens.weight", "F16", {256, 128}, half_bytes(256 * 128, 11, false));
        add("lm_head.weight", "F16", {256, 128}, half_bytes(256 * 128, 12, false));
    }
    std::string serialized = header.dump();
    serialized.append((8 - serialized.size() % 8) % 8, ' ');
    std::array<uint8_t, 8> length{};
    for (size_t i = 0; i < length.size(); ++i) length[i] = uint8_t(uint64_t(serialized.size()) >> (8 * i));
    std::ofstream shard(path / "model.safetensors", std::ios::binary);
    shard.write(reinterpret_cast<const char *>(length.data()), length.size());
    shard.write(serialized.data(), serialized.size());
    shard.write(reinterpret_cast<const char *>(payload.data()), payload.size());
    require(bool(shard), "cannot write DFlash safetensors fixture");
    std::ofstream config(path / "config.json");
    config << fixture_config(sliding).dump();
    require(bool(config), "cannot write DFlash config fixture");
}

// Only metadata probing uses this adapter. The actual model below is loaded
// through the production native safetensors entry point, not this test source.
class probe_source final : public llama_model_tensor_source {
public:
    explicit probe_source(const llama_safetensors_importer & importer) : importer(importer) {}
    bool describe(const std::string & name, ggml_type & type,
                  std::array<int64_t, GGML_MAX_DIMS> & ne) const override {
        return importer.describe(name, type, ne);
    }
    size_t tensor_capacity_hint() const override { return importer.tensor_capacity_hint(); }
    void bind(const std::string &) const override { throw std::runtime_error("unexpected probe bind"); }
    void load(ggml_tensor *, bool) const override { throw std::runtime_error("unexpected probe load"); }
    void validate_complete() const override { throw std::runtime_error("unexpected probe validation"); }
private:
    const llama_safetensors_importer & importer;
};

void check_family(const std::filesystem::path & path, bool sliding) {
    gguf_context_ptr preflight(llama_model_load_metadata(path.string().c_str()));
    require(preflight && std::string(gguf_get_val_str(preflight.get(),
        gguf_find_key(preflight.get(), "general.architecture"))) == "dflash",
        "native metadata preflight missed DFlash");
    require(common_speculative_types_from_model(path.string()) ==
        std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
        "plain -md native sidecar type was not detected before target creation");
    llama_safetensors_qwen3_importer importer(path, fixture_config(sliding));
    gguf_context_ptr metadata(importer.build_metadata());
    probe_source source(importer);
    std::vector<std::string> splits;
    llama_model_loader loader(metadata.get(), nullptr, nullptr, &source, "", splits,
        nullptr, LLAMA_LOAD_MODE_NONE, false, true, false, nullptr, nullptr);
    require(loader.get_tensor_meta_exact("selector.hidden_proj.weight") == nullptr,
        "native schema test accidentally used a prepopulated GGUF tensor map");
    for (const char * name : {"selector.hidden_proj.weight", "selector.pred_codebook",
                             "selector.succ_codebook", "blk.0.attn_conv.base", "blk.0.ffn_conv.proj.weight"}) {
        require(loader.has_tensor_exact(name), std::string("native exact lookup missed ") + name);
    }
    for (const char * name : {"selector_hidden.weight", "selector_predecessor.weight",
                             "selector_successor.weight", "blk.0.attn_conv_base", "blk.0.ffn_conv_proj.weight"}) {
        require(!loader.has_tensor_exact(name), std::string("native exact lookup invented compatibility alias ") + name);
    }
    require(llm_dflash_selector_family_from_loader(true, 1, loader) == llm_dflash_selector_family::fork_dflash2,
        "native loader did not select the fork DFlash2 schema");
}

void check_auxiliary(const ggml_tensor * tensor, const std::vector<uint8_t> & expected) {
    require(tensor && tensor->type == GGML_TYPE_F16 && ggml_nbytes(tensor) == expected.size(),
        "missing or misshaped EXL3 auxiliary");
    std::vector<uint8_t> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    require(actual == expected, std::string("EXL3 auxiliary changed: ") + tensor->name);
}

void check_model(const llama_model & model, bool standalone, bool sliding) {
    require(model.arch == LLM_ARCH_DFLASH, "native fixture selected the wrong graph architecture");
    require(model.vocab.get_type() == LLAMA_VOCAB_TYPE_NONE && model.vocab.n_tokens() == 256,
        "tokenizer-free native fixture has the wrong vocabulary");
    require(bool(model.tok_embd) == standalone && bool(model.output) == standalone,
        "optional shared target weights were fabricated or lost");
    require(model.hparams.dflash_target_layer_ids[0] == 1 && model.hparams.dflash_target_layer_ids[1] == 3 &&
            model.hparams.n_embd_inp_enc() == 256, "target feature indexing or FC input width changed");
    require(model.hparams.dflash_mask_token_id == 255 && model.hparams.dflash_block_size == 4,
        "DFlash block or mask metadata changed");
    require(model.hparams.is_swa(0) == sliding && model.hparams.rope_type == LLAMA_ROPE_TYPE_NEOX,
        "DFlash attention metadata changed");
    for (size_t i = 0; i < projections.size(); ++i) {
        const auto & p = projections[i];
        const std::string name = p.canonical;
        const ggml_tensor * weight = model.get_tensor((name + ".weight").c_str());
        require(weight && weight->type == GGML_TYPE_EXL3_4 && weight->ne[0] == int64_t(p.inputs) &&
            weight->ne[1] == int64_t(p.outputs), "native EXL3 projection geometry changed: " + name);
        check_auxiliary(model.get_tensor((name + ".input_scale").c_str()), half_bytes(p.inputs, i, true));
        check_auxiliary(model.get_tensor((name + ".scale").c_str()), half_bytes(p.outputs, i, false));
    }
    require(model.output_norm_enc && model.output_norm && model.dflash2_selector_hidden &&
        model.dflash2_selector_pred && model.dflash2_selector_succ && model.layers[0].dflash2_attn_conv_base &&
        model.layers[0].dflash2_attn_conv_proj && model.layers[0].dflash2_ffn_conv_base &&
        model.layers[0].dflash2_ffn_conv_proj, "native DFlash learned tensors missing");
}

void check_matmul(ggml_cgraph * graph, const llama_model & model, const projection & p) {
    const std::string name = p.canonical;
    const auto * weight = model.get_tensor((name + ".weight").c_str());
    size_t matches = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto * node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_MUL_MAT || node->src[0] != weight) continue;
        require(node->src[2] == model.get_tensor((name + ".scale").c_str()) &&
                node->src[3] == model.get_tensor((name + ".input_scale").c_str()),
            "DFlash graph dropped or swapped EXL3 auxiliaries: " + name);
        ++matches;
    }
    require(matches > 0, "DFlash graph lacks expected EXL3 projection: " + name);
}

void check_encoder(const llama_model & model) {
    llm_graph_result result(1024);
    llama_adapter_loras_ordered loras;
    llm_graph_params params{};
    params.arch = model.arch;
    params.hparams = model.hparams;
    params.ubatch.n_tokens = 2;
    params.gtype = LLM_GRAPH_TYPE_ENCODER;
    params.loras = &loras;
    params.res = &result;
    auto graph = model.build_arch_graph(params);
    check_matmul(result.get_gf(), model, projections[0]);
}

void check_decoder(llama_model & model) {
    auto params = llama_context_default_params();
    params.n_ctx = 256;
    params.n_batch = params.n_ubatch = 4;
    params.n_threads = params.n_threads_batch = 1;
    params.offload_kqv = false;
    params.op_offload = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context_ptr ctx(llama_init_from_model(&model, params));
    require(bool(ctx), "cannot create CPU native DFlash context");
    ggml_cgraph * graph = llama_graph_reserve(ctx.get(), 4, 1, 4);
    require(graph != nullptr, "cannot reserve native DFlash decoder graph");
    for (size_t i = 1; i < projections.size(); ++i) check_matmul(graph, model, projections[i]);

    // Build (but do not execute) the other decoder mode against the context's
    // ordinary cache: raw target features -> FC -> injected K/V. No scheduler
    // is needed because injection has no attention/output computation.
    auto memory = ctx->get_memory()->init_full();
    require(memory && memory->get_status() == LLAMA_MEMORY_STATUS_SUCCESS,
        "cannot initialize native DFlash injection cache view");
    llama_batch_allocr batches(model.hparams.n_pos_per_embd());
    llm_graph_result result(2048);
    llama_adapter_loras_ordered loras;
    std::vector<float> features(2 * model.hparams.n_embd_inp_enc());
    llm_graph_params inject{};
    inject.arch = model.arch;
    inject.hparams = model.hparams;
    inject.cparams = ctx->get_cparams();
    inject.cparams.dflash_fused_inject = true;
    inject.ubatch = batches.ubatch_reserve(2, 1);
    inject.ubatch.token = nullptr;
    inject.ubatch.embd = features.data();
    inject.gtype = LLM_GRAPH_TYPE_DECODER;
    inject.loras = &loras;
    inject.mctx = memory.get();
    inject.res = &result;
    auto injection_graph = model.build_arch_graph(inject);
    for (const size_t i : {size_t(0), size_t(2), size_t(3)}) {
        check_matmul(result.get_gf(), model, projections[i]);
    }
}

} // namespace

int main() {
    try {
        // Register only CPU; do not load dynamic accelerator backends.
        if (!ggml_backend_reg_by_name("CPU")) ggml_backend_register(ggml_backend_cpu_reg());
        llama_backend_init();
        temporary_directory temporary;
        ggml_backend_dev_t devices[] = {nullptr};
        auto params = llama_model_default_params();
        params.devices = devices;
        params.n_gpu_layers = 0;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.progress_callback = [](float, void *) { return true; };
        for (const bool sliding : {false, true}) {
            const std::string prefix = sliding ? "model." : "";
            for (const bool standalone : {false, true}) {
                const auto path = temporary.path / (std::string(sliding ? "swa" : "full") +
                    (standalone ? "-standalone" : "-sidecar"));
                write_fixture(path, prefix, sliding, standalone);
                check_family(path, sliding);
                llama_model_ptr model(llama_model_load_from_safetensors_dir(path, params));
                require(bool(model), "production native DFlash loader rejected fixture");
                check_model(*model, standalone, sliding);
                check_encoder(*model);
                if (standalone) check_decoder(*model);
            }
        }
        llama_backend_free();
        std::cout << "native DFlash2 loader and EXL3 graph contracts passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "native DFlash2 integration test: " << error.what() << '\n';
        llama_backend_free();
        return 1;
    }
}
