#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class common_mtp_vocab_trim_status {
    not_applicable,
    cached,
    created,
    failed,
};

struct common_mtp_vocab_trim_result {
    std::string                  path;
    common_mtp_vocab_trim_status status = common_mtp_vocab_trim_status::not_applicable;
    std::string                  detail;
};

// Prepare a cached FR-Spec-style derivative of a supported standalone Qwen-27B
// MTP GGUF. The source is never modified. Any unsupported shape or failure
// returns the source path so speculative decoding remains available.
common_mtp_vocab_trim_result common_mtp_vocab_trim_prepare(const std::string & source_path, uint32_t draft_vocab_size);

// The balanced draft vocabulary (ascending target token ids) for any GGUF whose
// tokenizer is the one the map was built for. Used for MTP sidecars that borrow
// the target's LM head, which the model trims in memory (llama_model_set_draft_vocab).
bool common_mtp_vocab_trim_ids(const std::string & gguf_path, uint32_t draft_vocab_size,
                               std::vector<int32_t> & ids, std::string & reason);

// Narrow model-free seam used by the GGUF codec test. Production callers must
// use common_mtp_vocab_trim_prepare(), which owns model admission and the map.
bool common_mtp_vocab_trim_repack_for_test(const std::string &          source_path,
                                           const std::string &          destination_path,
                                           const std::vector<int64_t> & draft_to_target,
                                           std::string &                error);

// Canonical tokenizer-token digest seam. Production admission reads the same
// serialization directly from GGUF metadata without materializing this vector.
std::string common_mtp_vocab_trim_tokenizer_digest_for_test(const std::vector<std::string> & tokens);
