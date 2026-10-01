#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

inline void llama_store_le_u32(uint8_t (&data)[4], uint32_t value) {
    for (size_t i = 0; i < sizeof(data); ++i) {
        data[i] = uint8_t(value >> (8*i));
    }
}

inline void llama_store_le_u64(uint8_t (&data)[8], uint64_t value) {
    for (size_t i = 0; i < sizeof(data); ++i) {
        data[i] = uint8_t(value >> (8*i));
    }
}

inline uint32_t llama_load_le_u32(const uint8_t (&data)[4]) {
    uint32_t value = 0;
    for (size_t i = 0; i < sizeof(data); ++i) {
        value |= uint32_t(data[i]) << (8*i);
    }
    return value;
}

inline uint64_t llama_load_le_u64(const uint8_t (&data)[8]) {
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(data); ++i) {
        value |= uint64_t(data[i]) << (8*i);
    }
    return value;
}

// Shared streaming SHA-256 implementation. update(nullptr, 0) is a no-op;
// nonempty input must be readable. finish() finalizes this instance once.
// Runtime acceleration does not change the digest or require CPU build flags.
class llama_sha256 {
public:
    llama_sha256();
    void update(const void * data, size_t size);
    std::array<uint8_t, 32> finish();

private:
    std::array<uint32_t, 8> state;
    std::array<uint8_t, 64> block = {};
    uint64_t total_len = 0;
    size_t block_len = 0;
};

// Raw-byte SHA-256 convenience entry point; canonical identity serialization
// continues to use llama_sha256_writer below.
std::array<uint8_t, 32> llama_sha256_digest(const void * data, size_t size);

// Canonical length-delimited digest serialization shared by every SHA-256 identity domain
// (little-endian fixed-width integers, u64-length-prefixed byte strings). Keeping one byte
// format here prevents silent drift between digest domains.
class llama_sha256_writer {
public:
    void bytes(const void * data, size_t size) {
        hash.update(data, size);
    }

    void u32(uint32_t value) {
        uint8_t data[4];
        llama_store_le_u32(data, value);
        bytes(data, sizeof(data));
    }

    void u64(uint64_t value) {
        uint8_t data[8];
        llama_store_le_u64(data, value);
        bytes(data, sizeof(data));
    }

    void string(const void * data, size_t size) {
        u64(size);
        if (size > 0) {
            bytes(data, size);
        }
    }

    std::array<uint8_t, 32> finish() {
        return hash.finish();
    }

private:
    llama_sha256 hash;
};
