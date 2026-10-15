#include "clip-safetensors.h"

#include "llama-safetensors.h"
#include "llama-safetensors-metadata.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// Both the Qwen4 and GLM5-Next multimodal checkpoints namespace their vision
// weights under this prefix; the converter strips it before mapping.
constexpr const char * VISUAL_PREFIX = "model.visual.";

ggml_type source_type(llama_safetensors_dtype dtype) {
    switch (dtype) {
        case llama_safetensors_dtype::BF16: return GGML_TYPE_BF16;
        case llama_safetensors_dtype::F16:  return GGML_TYPE_F16;
        case llama_safetensors_dtype::F32:  return GGML_TYPE_F32;
        default: throw std::runtime_error("vision source tensor has an unsupported dtype");
    }
}

uint32_t checked_u32(const llama_safetensors_json & object, const char * key) {
    const uint64_t value = object.at(key).get<uint64_t>();
    if (value == 0 || value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("invalid vision config value '") + key + "'");
    }
    return static_cast<uint32_t>(value);
}

bool is_supported_model_type(const std::string & model_type) {
    return model_type == "qwen4_exp" || model_type == "glm5_next";
}

// Mirror the GGUF converter's preprocessor loading (conversion/base.py): the
// image statistics may live in preprocessor_config.json (nested under
// media_proc_cfg) or in processor_config.json (nested under image_processor).
// GLM5-Next ships only the latter. Nested keys take precedence, as upstream.
const struct { const char * file; const char * nested; } preprocessor_files[] = {
    { "preprocessor_config.json", "media_proc_cfg" },
    { "processor_config.json",    "image_processor" },
};

bool has_preprocessor_config(const std::filesystem::path & dir) {
    for (const auto & entry : preprocessor_files) {
        if (std::filesystem::is_regular_file(dir / entry.file)) {
            return true;
        }
    }
    return false;
}

llama_safetensors_json load_preprocessor_config(const std::filesystem::path & dir) {
    llama_safetensors_json merged = llama_safetensors_json::object();
    for (const auto & entry : preprocessor_files) {
        const std::filesystem::path path = dir / entry.file;
        if (!std::filesystem::is_regular_file(path)) {
            continue;
        }
        const llama_safetensors_json cfg = llama_safetensors_read_json(path);
        merged.update(cfg);
        if (cfg.contains(entry.nested)) {
            merged.update(cfg.at(entry.nested));
        }
    }
    return merged;
}

} // namespace

struct clip_safetensors_source::impl {
    enum class family {
        QWEN4_EXP,
        GLM5_NEXT,
    };

    enum class transform {
        NONE,
        TO_F32,
        PATCH_TEMPORAL_SLICE,
    };

    struct binding {
        const llama_safetensors_tensor * source = nullptr;
        std::vector<int64_t>             ne;
        ggml_type                        type = GGML_TYPE_COUNT;
        transform                        kind = transform::NONE;
        uint32_t                         temporal_slice = 0;
    };

    family                                        model_family = family::QWEN4_EXP;
    std::filesystem::path                         model_dir;
    llama_safetensors_json                        config;
    llama_safetensors_json                        preprocessor;
    llama_safetensors_registry                    registry;
    std::unordered_map<std::string, binding>      bindings;
    std::unordered_set<std::string>               consumed_sources;

    explicit impl(const std::filesystem::path & path) :
        model_dir(path),
        config(llama_safetensors_read_model_config(path)),
        preprocessor(load_preprocessor_config(path)),
        registry(llama_safetensors_registry::load(path)) {
        const std::string model_type = config.value("model_type", std::string());
        if (!is_supported_model_type(model_type) || !config.contains("vision_config")) {
            throw std::runtime_error("native mmproj directory is not a supported multimodal source");
        }
        model_family = model_type == "glm5_next" ? family::GLM5_NEXT : family::QWEN4_EXP;

        for (const llama_safetensors_tensor & tensor : registry.tensors()) {
            if (tensor.name.rfind(VISUAL_PREFIX, 0) == 0) {
                add_source(tensor);
            }
        }
        if (consumed_sources.empty()) {
            throw std::runtime_error("vision source contains no vision tensors");
        }

        size_t visual_count = 0;
        for (const llama_safetensors_tensor & tensor : registry.tensors()) {
            visual_count += tensor.name.rfind(VISUAL_PREFIX, 0) == 0;
        }
        if (consumed_sources.size() != visual_count) {
            throw std::runtime_error("vision source contains unbound tensors");
        }
    }

    void bind(const std::string & canonical, const llama_safetensors_tensor & tensor,
              transform kind = transform::NONE, uint32_t temporal_slice = 0) {
        binding value;
        value.source = &tensor;
        value.type = kind == transform::TO_F32 ? GGML_TYPE_F32 : source_type(tensor.dtype);
        value.kind = kind;
        value.temporal_slice = temporal_slice;

        if (kind == transform::PATCH_TEMPORAL_SLICE) {
            if (tensor.shape.size() != 5 || tensor.shape[2] != 2 || temporal_slice >= tensor.shape[2]) {
                throw std::runtime_error("vision patch embedding has an unsupported temporal shape");
            }
            value.ne = {
                static_cast<int64_t>(tensor.shape[4]),
                static_cast<int64_t>(tensor.shape[3]),
                static_cast<int64_t>(tensor.shape[1]),
                static_cast<int64_t>(tensor.shape[0]),
            };
        } else {
            if (tensor.shape.empty() || tensor.shape.size() > GGML_MAX_DIMS) {
                throw std::runtime_error("vision tensor has an unsupported rank");
            }
            for (auto it = tensor.shape.rbegin(); it != tensor.shape.rend(); ++it) {
                if (*it == 0 || *it > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                    throw std::runtime_error("vision tensor has an invalid shape");
                }
                value.ne.push_back(static_cast<int64_t>(*it));
            }
        }

        auto inserted = bindings.emplace(canonical, std::move(value));
        if (!inserted.second) {
            throw std::runtime_error("duplicate canonical vision tensor '" + canonical + "'");
        }
        consumed_sources.insert(tensor.name);
    }

    void add_source(const llama_safetensors_tensor & tensor) {
        const std::string relative = tensor.name.substr(std::strlen(VISUAL_PREFIX));
        if (relative == "patch_embed.proj.bias") {
            bind("v.patch_embd.bias", tensor);
            return;
        }
        if (relative == "patch_embed.proj.weight") {
            bind("v.patch_embd.weight", tensor, transform::PATCH_TEMPORAL_SLICE, 0);
            bind("v.patch_embd.weight.1", tensor, transform::PATCH_TEMPORAL_SLICE, 1);
            return;
        }
        if (model_family == family::GLM5_NEXT) {
            add_source_glm5(tensor, relative);
        } else {
            add_source_qwen4(tensor, relative);
        }
    }

    void add_source_qwen4(const llama_safetensors_tensor & tensor, const std::string & relative) {
        if (relative == "pos_embed.weight") {
            // Qwen3-VL interpolates this table for dynamic image sizes, and the
            // ggml interpolation operation requires F32 input.
            bind("v.position_embd.weight", tensor, transform::TO_F32);
            return;
        }
        if (relative.rfind("merger.", 0) == 0) {
            const std::string suffix = relative.substr(std::strlen("merger."));
            if (suffix.rfind("norm.", 0) == 0) {
                bind("v.post_ln." + suffix.substr(std::strlen("norm.")), tensor);
            } else if (suffix.rfind("linear_fc1.", 0) == 0) {
                bind("mm.0." + suffix.substr(std::strlen("linear_fc1.")), tensor);
            } else if (suffix.rfind("linear_fc2.", 0) == 0) {
                bind("mm.2." + suffix.substr(std::strlen("linear_fc2.")), tensor);
            } else {
                throw std::runtime_error("unsupported Qwen4 vision merger tensor '" + tensor.name + "'");
            }
            return;
        }
        static const std::unordered_map<std::string, std::string> tails = {
            { "attn.proj.weight",       "attn_out.weight" },
            { "attn.proj.bias",         "attn_out.bias" },
            { "attn.qkv.weight",        "attn_qkv.weight" },
            { "attn.qkv.bias",          "attn_qkv.bias" },
            { "mlp.linear_fc1.weight",  "ffn_up.weight" },
            { "mlp.linear_fc1.bias",    "ffn_up.bias" },
            { "mlp.linear_fc2.weight",  "ffn_down.weight" },
            { "mlp.linear_fc2.bias",    "ffn_down.bias" },
            { "norm1.weight",           "ln1.weight" },
            { "norm1.bias",             "ln1.bias" },
            { "norm2.weight",           "ln2.weight" },
            { "norm2.bias",             "ln2.bias" },
        };
        if (bind_block(tensor, relative, tails, "Qwen4")) {
            return;
        }
        throw std::runtime_error("unsupported Qwen4 vision tensor '" + tensor.name + "'");
    }

    // "blocks.<n>.<tail>" -> "v.blk.<n>.<tails[tail]>"; false when relative is not a block tensor
    bool bind_block(const llama_safetensors_tensor & tensor, const std::string & relative,
                    const std::unordered_map<std::string, std::string> & tails, const char * family_name) {
        if (relative.rfind("blocks.", 0) != 0) {
            return false;
        }
        const size_t index_begin = std::strlen("blocks.");
        const size_t index_end = relative.find('.', index_begin);
        if (index_end == std::string::npos || index_end == index_begin) {
            throw std::runtime_error(std::string("invalid ") + family_name + " vision block tensor '" + tensor.name + "'");
        }
        const std::string index_text = relative.substr(index_begin, index_end - index_begin);
        if (!std::all_of(index_text.begin(), index_text.end(), [](char c) { return c >= '0' && c <= '9'; })) {
            throw std::runtime_error(std::string("invalid ") + family_name + " vision block index in '" + tensor.name + "'");
        }
        const auto mapped = tails.find(relative.substr(index_end + 1));
        if (mapped == tails.end()) {
            throw std::runtime_error(std::string("unsupported ") + family_name + " vision block tensor '" + tensor.name + "'");
        }
        bind("v.blk." + index_text + "." + mapped->second, tensor);
        return true;
    }

    void add_source_glm5(const llama_safetensors_tensor & tensor, const std::string & relative) {
        if (relative.rfind("post_layernorm.", 0) == 0) {
            bind("v.post_ln." + relative.substr(std::strlen("post_layernorm.")), tensor);
            return;
        }
        if (relative.rfind("downsample.", 0) == 0) {
            // Conv2D patch merger: passed through unchanged (ggml_conv_2d handles BF16 kernels).
            bind("mm.patch_merger." + relative.substr(std::strlen("downsample.")), tensor);
            return;
        }
        if (relative.rfind("merger.", 0) == 0) {
            const std::string suffix = relative.substr(std::strlen("merger."));
            if (suffix == "proj.weight") {
                bind("mm.model.fc.weight", tensor);
            } else if (suffix.rfind("post_projection_norm.", 0) == 0) {
                bind("mm.post_norm." + suffix.substr(std::strlen("post_projection_norm.")), tensor);
            } else if (suffix.rfind("gate_proj.", 0) == 0) {
                bind("mm.gate." + suffix.substr(std::strlen("gate_proj.")), tensor);
            } else if (suffix.rfind("up_proj.", 0) == 0) {
                bind("mm.up." + suffix.substr(std::strlen("up_proj.")), tensor);
            } else if (suffix.rfind("down_proj.", 0) == 0) {
                bind("mm.down." + suffix.substr(std::strlen("down_proj.")), tensor);
            } else {
                throw std::runtime_error("unsupported GLM5-Next vision merger tensor '" + tensor.name + "'");
            }
            return;
        }
        static const std::unordered_map<std::string, std::string> tails = {
            { "attn.qkv.weight",       "attn_qkv.weight" },
            { "attn.qkv.bias",         "attn_qkv.bias" },
            { "attn.q_norm.weight",    "attn_q_norm.weight" },
            { "attn.k_norm.weight",    "attn_k_norm.weight" },
            { "attn.proj.weight",      "attn_out.weight" },
            { "attn.proj.bias",        "attn_out.bias" },
            { "mlp.gate_proj.weight",  "ffn_gate.weight" },
            { "mlp.gate_proj.bias",    "ffn_gate.bias" },
            { "mlp.up_proj.weight",    "ffn_up.weight" },
            { "mlp.up_proj.bias",      "ffn_up.bias" },
            { "mlp.down_proj.weight",  "ffn_down.weight" },
            { "mlp.down_proj.bias",    "ffn_down.bias" },
            { "norm1.weight",          "ln1.weight" },
            { "norm1.bias",            "ln1.bias" },
            { "norm2.weight",          "ln2.weight" },
            { "norm2.bias",            "ln2.bias" },
        };
        if (bind_block(tensor, relative, tails, "GLM5-Next")) {
            return;
        }
        throw std::runtime_error("unsupported GLM5-Next vision tensor '" + tensor.name + "'");
    }
};

bool clip_safetensors_source::supports(const std::string & path) {
    const std::filesystem::path dir(path);
    if (!std::filesystem::is_directory(dir) || !std::filesystem::is_regular_file(dir / "config.json") ||
        !has_preprocessor_config(dir)) {
        return false;
    }
    try {
        const llama_safetensors_json config = llama_safetensors_read_model_config(dir);
        return is_supported_model_type(config.value("model_type", std::string())) && config.contains("vision_config");
    } catch (...) {
        return false;
    }
}

clip_safetensors_source::clip_safetensors_source(const std::string & path) : impl_(new impl(path)) {}
clip_safetensors_source::~clip_safetensors_source() = default;

gguf_context * clip_safetensors_source::build_metadata() const {
    const llama_safetensors_json & vision = impl_->config.at("vision_config");
    const llama_safetensors_json & text = impl_->config.at("text_config");
    const uint32_t patch_size = checked_u32(vision, "patch_size");
    const uint32_t merge = vision.value("spatial_merge_size", 2u);

    llama_safetensors_metadata_sink sink;
    sink.set_string("general.name", impl_->model_dir.filename().string() + " vision");
    sink.set_u32("general.file_type", 32); // LLAMA_FTYPE_MOSTLY_BF16
    sink.set_bool("clip.has_vision_encoder", true);
    sink.set_u32("clip.vision.embedding_length", checked_u32(vision, "hidden_size"));
    sink.set_u32("clip.vision.feed_forward_length", checked_u32(vision, "intermediate_size"));
    sink.set_u32("clip.vision.block_count", checked_u32(vision, "depth"));
    sink.set_u32("clip.vision.projection_dim", checked_u32(text, "hidden_size"));
    sink.set_u32("clip.vision.attention.head_count", checked_u32(vision, "num_heads"));
    sink.set_u32("clip.vision.patch_size", patch_size);
    sink.set_u32("clip.vision.spatial_merge_size", merge);

    if (impl_->model_family == impl::family::GLM5_NEXT) {
        sink.set_string("general.description", "Native GLM5-Next safetensors vision tower");
        sink.set_string("clip.projector_type", "glm5v");
        // GLM5-Next uses a SwiGLU FFN; the runtime hard-requires use_silu, the
        // clamp, and the image-token pixel budgets (converted from token counts).
        sink.set_bool("clip.use_silu", true);
        sink.set_f32("clip.vision.attention.layer_norm_epsilon", vision.value("rms_norm_eps", 1.0e-5f));
        sink.set_u32("clip.vision.image_size", checked_u32(vision, "image_size"));
        sink.set_f32("clip.vision.swiglu_clamp", vision.at("swiglu_limit").get<float>());

        const uint64_t pixels_per_token = static_cast<uint64_t>(patch_size) * merge * patch_size * merge;
        const uint64_t min_tokens = impl_->preprocessor.at("min_image_tokens").get<uint64_t>();
        const uint64_t max_tokens = impl_->preprocessor.at("max_image_tokens").get<uint64_t>();
        const uint64_t min_pixels = min_tokens * pixels_per_token;
        const uint64_t max_pixels = max_tokens * pixels_per_token;
        if (max_pixels == 0 || max_pixels > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("GLM5-Next vision image pixel budget is out of range");
        }
        sink.set_u32("clip.vision.image_min_pixels", static_cast<uint32_t>(min_pixels));
        sink.set_u32("clip.vision.image_max_pixels", static_cast<uint32_t>(max_pixels));
    } else {
        const uint32_t n_positions = checked_u32(vision, "num_position_embeddings");
        const uint32_t grid = static_cast<uint32_t>(std::llround(std::sqrt(static_cast<double>(n_positions))));
        if (static_cast<uint64_t>(grid) * grid != n_positions) {
            throw std::runtime_error("Qwen4 vision position count is not a square grid");
        }
        sink.set_string("general.description", "Native Qwen4 safetensors vision tower");
        sink.set_string("clip.projector_type", "qwen3vl_merger");
        sink.set_bool("clip.use_gelu", true);
        sink.set_f32("clip.vision.attention.layer_norm_epsilon", text.value("rms_norm_eps", 1.0e-6f));
        sink.set_u32("clip.vision.image_size", grid * patch_size);
    }

    const std::vector<float> image_mean = impl_->preprocessor.at("image_mean").get<std::vector<float>>();
    const std::vector<float> image_std = impl_->preprocessor.at("image_std").get<std::vector<float>>();
    if (image_mean.size() < 3 || image_std.size() < 3) {
        throw std::runtime_error("vision preprocessor image statistics are incomplete");
    }
    sink.set_f32_array("clip.vision.image_mean", image_mean.data(), image_mean.size());
    sink.set_f32_array("clip.vision.image_std", image_std.data(), image_std.size());
    return sink.release();
}

ggml_context * clip_safetensors_source::build_tensor_metadata() const {
    ggml_init_params params = {
        /*.mem_size   =*/ (impl_->bindings.size() + 1) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * context = ggml_init(params);
    if (!context) {
        throw std::runtime_error("failed to allocate vision tensor metadata");
    }
    try {
        for (const auto & entry : impl_->bindings) {
            const impl::binding & binding = entry.second;
            ggml_tensor * tensor = ggml_new_tensor(
                context, binding.type, static_cast<int>(binding.ne.size()), binding.ne.data());
            ggml_set_name(tensor, entry.first.c_str());
        }
    } catch (...) {
        ggml_free(context);
        throw;
    }
    return context;
}

size_t clip_safetensors_source::tensor_count() const {
    return impl_->bindings.size();
}

size_t clip_safetensors_source::model_size() const {
    size_t total = 0;
    for (const auto & entry : impl_->bindings) {
        const impl::binding & binding = entry.second;
        size_t elements = 1;
        for (int64_t extent : binding.ne) {
            if (elements > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent)) {
                throw std::runtime_error("vision tensor size overflows size_t");
            }
            elements *= static_cast<size_t>(extent);
        }
        total += ggml_row_size(binding.type, elements);
    }
    return total;
}

bool clip_safetensors_source::load(const std::string & canonical_name, ggml_tensor * destination) const {
    const auto found = impl_->bindings.find(canonical_name);
    if (found == impl_->bindings.end()) {
        return false;
    }
    const impl::binding & binding = found->second;
    if (destination->type != binding.type || ggml_nbytes(destination) != ggml_row_size(binding.type, ggml_nelements(destination))) {
        throw std::runtime_error("vision destination contract changed for '" + canonical_name + "'");
    }

    std::vector<uint8_t> bytes;
    if (binding.kind == impl::transform::NONE) {
        bytes = impl_->registry.read(*binding.source);
    } else if (binding.kind == impl::transform::TO_F32) {
        const std::vector<uint8_t> source = impl_->registry.read(*binding.source);
        const size_t n = ggml_nelements(destination);
        bytes.resize(n * sizeof(float));
        if (binding.source->dtype == llama_safetensors_dtype::BF16) {
            ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(source.data()),
                                  reinterpret_cast<float *>(bytes.data()), n);
        } else if (binding.source->dtype == llama_safetensors_dtype::F16) {
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(source.data()),
                                  reinterpret_cast<float *>(bytes.data()), n);
        } else if (binding.source->dtype == llama_safetensors_dtype::F32) {
            bytes = source;
        } else {
            throw std::runtime_error("Qwen4 position embedding has an unsupported source dtype");
        }
    } else {
        const std::vector<uint8_t> source = impl_->registry.read(*binding.source);
        const size_t element_size = llama_safetensors_dtype_size(binding.source->dtype);
        const size_t plane_elements = static_cast<size_t>(binding.source->shape[3] * binding.source->shape[4]);
        const size_t plane_bytes = plane_elements * element_size;
        const size_t outer = static_cast<size_t>(binding.source->shape[0] * binding.source->shape[1]);
        bytes.resize(outer * plane_bytes);
        for (size_t i = 0; i < outer; ++i) {
            const size_t source_offset = (i * binding.source->shape[2] + binding.temporal_slice) * plane_bytes;
            std::memcpy(bytes.data() + i * plane_bytes, source.data() + source_offset, plane_bytes);
        }
    }
    if (bytes.size() != ggml_nbytes(destination)) {
        throw std::runtime_error("vision materialization has the wrong size for '" + canonical_name + "'");
    }
    ggml_backend_tensor_set(destination, bytes.data(), 0, bytes.size());
    return true;
}
