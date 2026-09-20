#include "llama-safetensors-qwen3.h"

#include "llama-safetensors-metadata.h"
#include "llama-safetensors-names.h"
#include "llama-safetensors-tensor.h"
#include "llama.h"

#include <algorithm>
#include <cstring>
#include <regex>
#include <stdexcept>
#include <string_view>

namespace {

llama_safetensors_source_name plain_source(std::string name) {
    return { std::move(name), {}, std::nullopt };
}

llama_safetensors_source_name projection_source(std::string module, llama_safetensors_quant_role role) {
    std::string source = module;
    switch (role) {
        case llama_safetensors_quant_role::WEIGHT:
            source += ".weight";
            break;
        case llama_safetensors_quant_role::WEIGHT_SCALE:
            source += ".weight_scale";
            break;
        case llama_safetensors_quant_role::INPUT_SCALE:
            source += ".input_global_scale";
            break;
    }
    return { std::move(source), std::move(module), role };
}

std::optional<llama_safetensors_source_name> map_target_source(
        uint32_t n_layer, const std::string & source_prefix, const std::string & target_name) {
    if (target_name == "token_embd.weight") {
        return projection_source(source_prefix + "embed_tokens", llama_safetensors_quant_role::WEIGHT);
    }
    if (target_name == "output_norm.weight") {
        return plain_source(source_prefix + "norm.weight");
    }
    if (target_name == "output.weight") {
        return projection_source("lm_head", llama_safetensors_quant_role::WEIGHT);
    }
    if (target_name == "output.bias") {
        return plain_source("lm_head.bias");
    }
    if (target_name == "output.scale") {
        return projection_source("lm_head", llama_safetensors_quant_role::WEIGHT_SCALE);
    }
    if (target_name == "output.input_scale") {
        return projection_source("lm_head", llama_safetensors_quant_role::INPUT_SCALE);
    }

    static const std::regex layer_pattern(R"(^blk\.([0-9]+)\.(.+)$)");
    std::smatch             match;
    if (!std::regex_match(target_name, match, layer_pattern)) {
        return std::nullopt;
    }
    const uint32_t layer = std::stoul(match[1].str());
    if (layer >= n_layer) {
        throw std::runtime_error("native Qwen3 importer does not support layer " + std::to_string(layer));
    }
    const std::string prefix = source_prefix + "layers." + std::to_string(layer) + ".";
    return llama_safetensors_map_decoder_tensor(prefix, match[2].str());
}

bool is_dflash2(const llama_safetensors_json & config) {
    return config.value("architectures", std::vector<std::string>{}) ==
        std::vector<std::string>{"DFlash2DraftModel"};
}

void validate_model_contract(const llama_safetensors_json & config) {
    const uint32_t n_layer  = config.value("num_hidden_layers", 0U);
    const uint32_t n_embd   = config.value("hidden_size", 0U);
    const uint32_t n_head   = config.value("num_attention_heads", 0U);
    const uint32_t n_kv     = config.value("num_key_value_heads", 0U);
    const uint32_t head_dim = config.value("head_dim", n_head == 0 ? 0U : n_embd / n_head);
    if (n_layer == 0 || n_embd == 0 || n_head == 0 || n_kv == 0 || head_dim == 0 || n_head % n_kv != 0) {
        throw std::runtime_error("native Qwen3 importer does not support this tensor geometry");
    }
}

bool llama_rope_permuted_target(const std::string & target_name) {
    const auto has_suffix = [&](std::string_view suffix) {
        return target_name.size() >= suffix.size() &&
               std::string_view(target_name).substr(target_name.size() - suffix.size()) == suffix;
    };
    return has_suffix(".attn_q.weight") || has_suffix(".attn_k.weight") ||
           has_suffix(".attn_q.scale") || has_suffix(".attn_k.scale");
}

std::vector<uint8_t> permute_llama_rows(
        std::vector<uint8_t> source, uint32_t heads, uint32_t head_dim) {
    const size_t rows = size_t(heads) * head_dim;
    if (head_dim % 2 != 0 || rows == 0 || source.size() % rows != 0) {
        throw std::runtime_error("Llama RoPE tensor has an unsupported row layout");
    }
    const size_t row_size = source.size() / rows;
    std::vector<uint8_t> result(source.size());
    for (uint32_t head = 0; head < heads; ++head) {
        for (uint32_t pair = 0; pair < head_dim / 2; ++pair) {
            for (uint32_t parity = 0; parity < 2; ++parity) {
                const size_t dst_row = size_t(head) * head_dim + 2 * pair + parity;
                const size_t src_row = size_t(head) * head_dim + parity * (head_dim / 2) + pair;
                std::memcpy(result.data() + dst_row * row_size,
                            source.data() + src_row * row_size, row_size);
            }
        }
    }
    return result;
}

std::vector<uint8_t> permute_w8a16_scale_rows(
        std::vector<uint8_t> source, uint32_t heads, uint32_t head_dim) {
    ggml_w8a16_scale_header header;
    if (source.size() < sizeof(header)) {
        throw std::runtime_error("W8A16 scale bundle is truncated");
    }
    std::memcpy(&header, source.data(), sizeof(header));
    const size_t rows = size_t(heads) * head_dim;
    if (header.magic != GGML_W8A16_SCALE_MAGIC || header.version != 1 ||
        header.n_channels != rows || header.values_offset != sizeof(header) ||
        header.total_size != source.size() ||
        source.size() - header.values_offset != rows * sizeof(ggml_bf16_t)) {
        throw std::runtime_error("W8A16 scale bundle has an unsupported row layout");
    }
    std::vector<uint8_t> values(source.begin() + header.values_offset, source.end());
    values = permute_llama_rows(std::move(values), heads, head_dim);
    std::memcpy(source.data() + header.values_offset, values.data(), values.size());
    return source;
}

}  // namespace

llama_safetensors_qwen3_importer::llama_safetensors_qwen3_importer(const std::filesystem::path & model_dir,
                                                                   llama_safetensors_json        config,
                                                                   llama_safetensors_io_mode     io_mode) :
    model_dir_(model_dir),
    config_(std::move(config)) {
    const std::string model_type = config_.value("model_type", std::string());
    const bool is_vl = model_type == "qwen3_vl";
    const bool is_draft = is_dflash2(config_);
    // Classic Hugging Face Mistral checkpoints use the Llama execution
    // architecture and the same Q/K RoPE row permutation. Newer Mistral3/4
    // architectures have different tensor and graph contracts and are not
    // selected here.
    const bool is_llama = model_type == "llama" || model_type == "mistral";
    text_config_ = is_vl ? config_.at("text_config") : config_;
    source_prefix_ = is_vl ? "model.language_model." : "model.";
    architecture_ = is_draft ? "dflash" : is_vl ? "qwen3vl" : is_llama ? "llama" : model_type;
    validate_model_contract(text_config_);
    n_layer_                                      = text_config_.at("num_hidden_layers").get<uint32_t>();
    n_head_                                       = text_config_.at("num_attention_heads").get<uint32_t>();
    n_head_kv_                                    = text_config_.at("num_key_value_heads").get<uint32_t>();
    head_dim_                                     = text_config_.value(
        "head_dim", text_config_.at("hidden_size").get<uint32_t>() / n_head_);
    registry_ = llama_safetensors_registry::load(model_dir_, io_mode);
    quant_    = std::make_unique<llama_safetensors_quant_adapters>(config_, registry_);
    if (is_draft) {
        // Published sidecars omit the tokenizer and shared embedding/head. They
        // consume target token IDs; ordinary DFlash setup attaches shared tensors.
        source_prefix_ = registry_.find("model.hidden_norm.weight") ? "model." : "";
        if (model_type != "qwen3" || !config_.value("rope_is_neox_style", true) ||
            config_.contains("final_logit_softcapping") || config_.at("dflash_config").contains("final_logit_softcapping")) {
            throw std::runtime_error("native DFlash2 currently requires the Qwen NeoX-RoPE backbone");
        }
        return;
    }
    generation_                                   = llama_safetensors_read_json(model_dir_ / "generation_config.json");
    tokenizer_                                    = llama_safetensors_read_tokenizer_json(model_dir_ / "tokenizer.json");
    const llama_safetensors_json tokenizer_config = llama_safetensors_read_json(model_dir_ / "tokenizer_config.json");
    add_bos_token_ = tokenizer_config.value("add_bos_token", is_llama);
    if (tokenizer_config.contains("pad_token") && tokenizer_config.at("pad_token").is_string()) {
        padding_token_ = tokenizer_config.at("pad_token").get<std::string>();
    }
    if (tokenizer_config.contains("chat_template") && tokenizer_config.at("chat_template").is_string()) {
        chat_template_ = tokenizer_config.at("chat_template").get<std::string>();
    }
}

std::optional<llama_safetensors_source_name> llama_safetensors_qwen3_importer::map_source(
        const std::string & name) const {
    if (architecture_ == "dflash") {
        if (name == "enc.output_norm.weight") return plain_source(source_prefix_ + "hidden_norm.weight");
        for (const auto & item : std::array<std::pair<std::string, std::string>, 4>{{
                {"fc", "fc"}, {"selector.hidden_proj", "candidate_selector.hidden_projection"},
                {"selector.pred_codebook", "candidate_selector.predecessor_codebook"},
                {"selector.succ_codebook", "candidate_selector.successor_codebook"}}}) {
            const auto & canonical = item.first;
            const std::string module = source_prefix_ + item.second;
            if (name == canonical) {
                return plain_source(registry_.find(module + ".weight") ? module + ".weight" : module);
            }
            if (name == canonical + ".weight") return projection_source(module, llama_safetensors_quant_role::WEIGHT);
            if (name == canonical + ".scale") return projection_source(module, llama_safetensors_quant_role::WEIGHT_SCALE);
            if (name == canonical + ".input_scale") return projection_source(module, llama_safetensors_quant_role::INPUT_SCALE);
        }
        static const std::regex conv(R"(^blk\.([0-9]+)\.(attn|ffn)_conv\.(base|proj\.weight)$)");
        std::smatch match;
        if (std::regex_match(name, match, conv)) {
            if (std::stoul(match[1].str()) >= n_layer_) return std::nullopt;
            const std::string module = source_prefix_ + "layers." + match[1].str() +
                (match[2] == "attn" ? ".attention_conv." : ".mlp_conv.");
            return plain_source(module + (match[3] == "base" ? "base_kernel" : "kernel_projection.weight"));
        }
    }
    return map_target_source(n_layer_, source_prefix_, name);
}

llama_safetensors_tensor_binding llama_safetensors_qwen3_importer::map_tensor(const std::string & name) const {
    auto source = map_source(name);
    if (!source) throw std::runtime_error("unsupported " + architecture_ + " target tensor '" + name + "'");
    return llama_safetensors_bind_tensor(*quant_, std::move(*source));
}

bool llama_safetensors_qwen3_importer::probe(const llama_safetensors_json & config) {
    const std::string model_type = config.value("model_type", std::string());
    return is_dflash2(config) || model_type == "qwen2" || model_type == "qwen3" ||
           model_type == "qwen3_vl" || model_type == "llama" ||
           model_type == "mistral";
}

gguf_context * llama_safetensors_qwen3_importer::build_metadata() const {
    llama_safetensors_metadata_sink sink;
    sink.set_string("general.architecture", architecture_);
    sink.set_string("general.type", "model");
    sink.set_string("general.name",
                    model_dir_.filename().empty() ? "Qwen3 Safetensors" : model_dir_.filename().string());
    sink.set_u32("general.file_type", quant_->file_type());
    sink.set_u32("general.quantization_version", 2);
    if (architecture_ != "dflash") llama_safetensors_emit_sampling_defaults(sink, generation_);

    const std::string prefix = architecture_ + ".";
    sink.set_u32(prefix + "block_count", text_config_.at("num_hidden_layers").get<uint32_t>());
    sink.set_u32(prefix + "context_length", text_config_.at("max_position_embeddings").get<uint32_t>());
    sink.set_u32(prefix + "embedding_length", text_config_.at("hidden_size").get<uint32_t>());
    sink.set_u32(prefix + "feed_forward_length", text_config_.at("intermediate_size").get<uint32_t>());
    sink.set_u32(prefix + "attention.head_count", text_config_.at("num_attention_heads").get<uint32_t>());
    sink.set_u32(prefix + "attention.head_count_kv", text_config_.at("num_key_value_heads").get<uint32_t>());
    const uint32_t head_dim = text_config_.value(
        "head_dim", text_config_.at("hidden_size").get<uint32_t>() /
                    text_config_.at("num_attention_heads").get<uint32_t>());
    sink.set_u32(prefix + "attention.key_length", head_dim);
    sink.set_u32(prefix + "attention.value_length", head_dim);
    const auto & rope = text_config_.contains("rope_parameters") ?
        text_config_.at("rope_parameters") : text_config_;
    sink.set_f32(prefix + "rope.freq_base", rope.at("rope_theta").get<float>());
    sink.set_f32(prefix + "attention.layer_norm_rms_epsilon", text_config_.at("rms_norm_eps").get<float>());
    if (architecture_ == "qwen3vl") {
        std::array<int32_t, 4> sections = { 0, 0, 0, 0 };
        const auto & source_sections = text_config_.at("rope_scaling").at("mrope_section");
        if (!source_sections.is_array() || source_sections.empty() || source_sections.size() > sections.size()) {
            throw std::runtime_error("native Qwen3-VL importer received invalid mrope sections");
        }
        for (size_t i = 0; i < source_sections.size(); ++i) {
            sections[i] = source_sections.at(i).get<int32_t>();
        }
        sink.set_i32_array(prefix + "rope.dimension_sections", sections.data(), sections.size());
        sink.set_u32(prefix + "n_deepstack_layers", static_cast<uint32_t>(
            config_.at("vision_config").at("deepstack_visual_indexes").size()));
    }

    if (architecture_ == "dflash") {
        const auto & cfg = config_.at("dflash_config");
        const uint32_t vocab = config_.at("vocab_size").get<uint32_t>();
        const uint32_t mask = cfg.at("mask_token_id").get<uint32_t>();
        const uint32_t block = cfg.at("block_size").get<uint32_t>();
        auto layers = cfg.at("target_layer_ids").get<std::vector<uint32_t>>();
        if (vocab == 0 || mask >= vocab || block < 3 || block > 64 || layers.empty() || layers.size() > 8) {
            throw std::runtime_error("invalid DFlash2 vocabulary, block size, or target layers");
        }
        for (auto & layer : layers) {
            if (layer >= config_.at("num_target_layers").get<uint32_t>()) {
                throw std::runtime_error("DFlash2 target layer is out of range");
            }
            ++layer; // Runtime feature extraction numbers the embedding output as zero.
        }
        sink.set_string("tokenizer.ggml.model", "none");
        sink.set_u32(prefix + "vocab_size", vocab);
        sink.set_u32("tokenizer.ggml.mask_token_id", mask);
        sink.set_u32(prefix + "block_size", block);
        sink.set_u32_array(prefix + "target_layers", layers.data(), layers.size());
        const auto causal = config_.value("is_causal", llama_safetensors_json());
        sink.set_bool(prefix + "attention.causal", causal.is_null() ? cfg.value("causal", false) : causal.get<bool>());
        for (const auto & item : {std::pair{"output_multiplier", "logit_scale"},
                                  std::pair{"input_embedding_scale", "embedding_scale"}}) {
            const auto value = cfg.value(item.first, config_.value(item.first, llama_safetensors_json()));
            if (!value.is_null()) sink.set_f32(prefix + item.second, value.get<float>());
        }
        for (const char * key : {"conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"}) {
            sink.set_u32(prefix + key, cfg.at(key).get<uint32_t>());
        }
        const auto draft_window = cfg.value("swa_window_size", llama_safetensors_json());
        const auto model_window = config_.value("sliding_window", llama_safetensors_json());
        const uint32_t window = !draft_window.is_null() && draft_window.get<uint32_t>() > 0 ?
            draft_window.get<uint32_t>() : model_window.is_null() ? 0U : model_window.get<uint32_t>();
        if ((config_.value("use_sliding_window", false) || cfg.value("use_swa", false)) && window > 0) {
            const auto types = config_.at("layer_types").get<std::vector<std::string>>();
            if (types.size() != n_layer_) throw std::runtime_error("invalid DFlash2 sliding-window pattern");
            auto pattern = std::make_unique<bool[]>(n_layer_);
            for (uint32_t i = 0; i < n_layer_; ++i) {
                if (types[i] != "sliding_attention" && types[i] != "full_attention") {
                    throw std::runtime_error("unsupported DFlash2 layer type");
                }
                pattern[i] = types[i] == "sliding_attention";
            }
            sink.set_u32(prefix + "attention.sliding_window", window);
            gguf_context * metadata = sink.release();
            gguf_set_arr_data(metadata, (prefix + "attention.sliding_window_pattern").c_str(),
                    GGUF_TYPE_BOOL, pattern.get(), n_layer_);
            return metadata;
        }
        return sink.release();
    }

    const uint32_t bos = llama_safetensors_first_token_id(generation_.at("bos_token_id"), "bos_token_id");
    const uint32_t eos = llama_safetensors_first_token_id(generation_.at("eos_token_id"), "eos_token_id");
    if (architecture_ == "llama" && std::filesystem::is_regular_file(model_dir_ / "tokenizer.model")) {
        llama_safetensors_emit_spm_tokenizer(
            sink, model_dir_ / "tokenizer.model", text_config_.at("vocab_size").get<uint32_t>(),
            bos, eos, eos, add_bos_token_, chat_template_);
    } else {
        llama_safetensors_bpe_policy tokenizer_policy{
            architecture_ == "llama" ? "llama-bpe" : "qwen2",
            text_config_.at("vocab_size").get<uint32_t>(),
            bos,
            eos,
            architecture_ == "llama" ? padding_token_ : std::optional<std::string>("<|endoftext|>"),
            true,
            {},
            add_bos_token_,
        };
        llama_safetensors_emit_bpe_tokenizer(sink, tokenizer_, tokenizer_policy, chat_template_);
    }
    return sink.release();
}

bool llama_safetensors_qwen3_importer::describe(const std::string &                  target_name,
                                                ggml_type &                          type,
                                                std::array<int64_t, GGML_MAX_DIMS> & ne) const {
    auto source = map_source(target_name);
    if (!source) {
        return false;
    }
    llama_safetensors_tensor_binding binding = llama_safetensors_bind_tensor(*quant_, std::move(*source));
    return llama_safetensors_describe_tensor(registry_, binding, type, ne);
}

size_t llama_safetensors_qwen3_importer::tensor_capacity_hint() const {
    return std::max<size_t>(256, registry_.tensors().size() * 2);
}

void llama_safetensors_qwen3_importer::bind(const std::string & target_name) const {
    llama_safetensors_consume_tensor(*quant_, map_tensor(target_name));
}

std::optional<llama_model_tensor_file_region> llama_safetensors_qwen3_importer::file_region(
        const ggml_tensor * destination) const {
    if (architecture_ == "llama" && llama_rope_permuted_target(destination->name) && ggml_nelements(destination) > 1) {
        return std::nullopt;
    }
    return llama_safetensors_tensor_file_region(registry_,
        map_tensor(destination->name), destination);
}

bool llama_safetensors_qwen3_importer::can_stream(const std::string & target_name) const {
    if (architecture_ == "llama" && llama_rope_permuted_target(target_name)) return false;
    const auto binding = map_tensor(target_name);
    return binding.quant && quant_->can_stream(*binding.quant);
}

void llama_safetensors_qwen3_importer::stream(
        const std::string & target_name, const std::function<void(const void *, size_t)> & write) const {
    if (!can_stream(target_name)) throw std::runtime_error("unsupported Qwen3 streaming transform: " + target_name);
    const auto binding = map_tensor(target_name);
    quant_->stream(*binding.quant, write);
}

bool llama_safetensors_qwen3_importer::load(
        const std::string & target_name, ggml_tensor * destination, bool check_tensor) const {
    if (architecture_ == "llama" && llama_rope_permuted_target(target_name) &&
        ggml_nelements(destination) > 1) {
        return false;
    }
    return llama_safetensors_load_tensor_direct(
        registry_, map_tensor(target_name), destination, check_tensor);
}

std::vector<uint8_t> llama_safetensors_qwen3_importer::materialize(const std::string & target_name,
                                                                   ggml_type           target_type,
                                                                   size_t              target_size) const {
    try {
        const llama_safetensors_tensor_binding binding =
            map_tensor(target_name);
        std::vector<uint8_t> result = llama_safetensors_materialize_tensor(
            registry_, *quant_, binding,
            target_type, target_size);
        if (architecture_ == "llama" && llama_rope_permuted_target(target_name) &&
            target_size > ggml_type_size(target_type)) {
            const bool key = target_name.find(".attn_k.") != std::string::npos;
            const uint32_t heads = key ? n_head_kv_ : n_head_;
            result = binding.quant &&
                    binding.quant->materialization ==
                        llama_safetensors_quant_materialization::QUANTO_W8A16_SCALE ?
                permute_w8a16_scale_rows(std::move(result), heads, head_dim_) :
                permute_llama_rows(std::move(result), heads, head_dim_);
        }
        return result;
    } catch (const std::exception & error) {
        throw std::runtime_error("failed to materialize '" + target_name + "': " + error.what());
    }
}

void llama_safetensors_qwen3_importer::validate_complete() const {
    quant_->validate_complete();
}
