#include "llama-sha256.h"

extern "C" {
#include "sha256/sha256.h"
}

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(x)                                                      \
    do {                                                              \
        if (!(x)) {                                                   \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); \
            ++failures;                                               \
        }                                                             \
    } while (0)

static std::string hex(const std::array<uint8_t, 32> & digest) {
    static const char digits[] = "0123456789abcdef";
    std::string       result;
    for (uint8_t byte : digest) {
        result += digits[byte >> 4];
        result += digits[byte & 15];
    }
    return result;
}

// Independent vendored implementation, not a second call through our dispatch.
static std::array<uint8_t, 32> reference(const void * data, size_t size) {
    std::array<uint8_t, 32> result;
    sha256_hash(result.data(), static_cast<const unsigned char *>(data), size);
    return result;
}

static void check_input(const uint8_t * data, size_t size) {
    const auto expected = reference(data, size);
    CHECK(llama_sha256_digest(data, size) == expected);
    for (size_t chunk : { 1u, 3u, 55u, 63u, 64u, 65u, 4093u, 1048576u }) {
        if (size > 4096 && chunk < 55) {
            continue;
        }
        llama_sha256 hash;
        hash.update(nullptr, 0);
        for (size_t offset = 0; offset < size;) {
            const size_t n = std::min(chunk, size - offset);
            hash.update(data + offset, n);
            hash.update(nullptr, 0);  // also after a partial head/tail
            offset += n;
        }
        CHECK(hash.finish() == expected);
    }
    // Prefix snapshots are used by identity/hash builders; copying must still
    // preserve both a partial block and the number of bytes already processed.
    llama_sha256 prefix;
    const size_t split = size / 2;
    prefix.update(data, split);
    auto copy = prefix;
    CHECK(prefix.finish() == reference(data, split));
    copy.update(data + split, size - split);
    CHECK(copy.finish() == expected);
}

int main(int argc, char ** argv) {
    CHECK(hex(llama_sha256_digest(nullptr, 0)) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(hex(llama_sha256_digest("abc", 3)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const std::string long_vector = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(hex(llama_sha256_digest(long_vector.data(), long_vector.size())) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    const std::string million(1000000, 'a');
    CHECK(hex(llama_sha256_digest(million.data(), million.size())) ==
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");

    std::vector<uint8_t> bytes((1u << 20) + 32);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = uint8_t((i * 131) ^ (i >> 7));
    }
    for (size_t alignment : { 0u, 1u, 7u, 15u }) {
        for (size_t size = 0; size <= 130; ++size) {
            check_input(bytes.data() + alignment, size);
        }
        for (size_t size : { 255u, 256u, 257u, 511u, 512u, 513u, 4095u, 4096u, 4097u, 1048575u, 1048576u, 1048577u }) {
            check_input(bytes.data() + alignment, size);
        }
    }
    // Canonical writer serialization remains little-endian and length-delimited.
    llama_sha256_writer writer;
    writer.u32(0x12345678);
    writer.u64(0x0102030405060708ULL);
    writer.string("abc", 3);
    writer.string(nullptr, 0);
    const uint8_t canonical[] = {
        0x78, 0x56, 0x34, 0x12, 8, 7, 6, 5, 4, 3, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 'a', 'b', 'c', 0, 0, 0, 0, 0, 0, 0, 0,
    };
    CHECK(writer.finish() == reference(canonical, sizeof(canonical)));
    if (failures) {
        return 1;
    }
    std::puts("SHA-256 vectors, streaming, unaligned inputs, empty updates, and canonical writer: PASS");
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
        bytes.resize(32u << 20, 0x5a);
        uint8_t    result = 0;
        const auto start  = std::chrono::steady_clock::now();
        for (int i = 0; i < 32; ++i) {
            bytes[0] = uint8_t(i);
            result ^= llama_sha256_digest(bytes.data(), bytes.size())[0];
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("SHA-256: %.3f GB/s (%g seconds, checksum=%u)\n", double(bytes.size()) * 32 / seconds / 1e9,
                    seconds, unsigned(result));
    }
    return 0;
}
