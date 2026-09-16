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

// Narrow model-free seam used by the GGUF codec test. Production callers must
// use common_mtp_vocab_trim_prepare(), which owns model admission and the map.
bool common_mtp_vocab_trim_repack_for_test(const std::string &          source_path,
                                           const std::string &          destination_path,
                                           const std::vector<int64_t> & draft_to_target,
                                           std::string &                error);

// Canonical tokenizer-token digest seam. Production admission reads the same
// serialization directly from GGUF metadata without materializing this vector.
std::string common_mtp_vocab_trim_tokenizer_digest_for_test(const std::vector<std::string> & tokens);
