#include "llama-sha256.h"

#include <algorithm>
#include <cstring>

// SHA-NI and direct-block updates adapted from 72eaaed714d865d69270cd413e3f7dceeb71589a
// (spiritbuun, co-authored by Claude Fable 5.1). Keep compression and runtime
// dispatch here so all identity, artifact, and resume callers share one backend.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER) && \
    !defined(LLAMA_SHA256_FORCE_PORTABLE)
#    define LLAMA_SHA256_X86 1
#    include <cpuid.h>
#    include <immintrin.h>
#endif

namespace {

constexpr uint32_t k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

void compress_portable(uint32_t * state, const uint8_t * data, size_t n_blocks) {
    for (; n_blocks > 0; --n_blocks, data += 64) {
        uint32_t w[64];
        for (size_t i = 0; i < 16; ++i) {
            w[i] = (uint32_t(data[4 * i + 0]) << 24) | (uint32_t(data[4 * i + 1]) << 16) |
                   (uint32_t(data[4 * i + 2]) << 8) | (uint32_t(data[4 * i + 3]) << 0);
        }
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i]              = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];

        for (size_t i = 0; i < 64; ++i) {
            const uint32_t s1  = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch  = (e & f) ^ (~e & g);
            const uint32_t t1  = h + s1 + ch + k[i] + w[i];
            const uint32_t s0  = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2  = s0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }
}

#ifdef LLAMA_SHA256_X86
bool have_sha_ext() {
    unsigned a = 0, b = 0, c = 0, d = 0;
    // SSSE3 and SSE4.1, then the SHA extensions
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & (1u << 9)) || !(c & (1u << 19))) {
        return false;
    }
    return __get_cpuid_count(7, 0, &a, &b, &c, &d) && (b & (1u << 29));
}

__attribute__((target("sha,sse4.1,ssse3")))
void compress_sha_ext(uint32_t * state, const uint8_t * data, size_t n_blocks) {
    const __m128i swap   = _mm_set_epi64x(0x0c0d0e0f08090a0bll, 0x0405060700010203ll);
    // the instructions want the state as ABEF and CDGH
    __m128i       tmp    = _mm_shuffle_epi32(_mm_loadu_si128((const __m128i *) &state[0]), 0xB1);
    __m128i       state1 = _mm_shuffle_epi32(_mm_loadu_si128((const __m128i *) &state[4]), 0x1B);
    __m128i       state0 = _mm_alignr_epi8(tmp, state1, 8);
    state1               = _mm_blend_epi16(state1, tmp, 0xF0);

    for (; n_blocks > 0; --n_blocks, data += 64) {
        const __m128i save0 = state0;
        const __m128i save1 = state1;
        __m128i       w[4];
        for (size_t i = 0; i < 16; ++i) {
            if (i < 4) {
                w[i] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) (data + 16 * i)), swap);
            } else {
                const __m128i prev = w[(i + 3) % 4];
                __m128i       next = _mm_sha256msg1_epu32(w[i % 4], w[(i + 1) % 4]);
                next               = _mm_add_epi32(next, _mm_alignr_epi8(prev, w[(i + 2) % 4], 4));
                w[i % 4]           = _mm_sha256msg2_epu32(next, prev);
            }
            __m128i msg = _mm_add_epi32(w[i % 4], _mm_loadu_si128((const __m128i *) &k[4 * i]));
            state1      = _mm_sha256rnds2_epu32(state1, state0, msg);
            msg         = _mm_shuffle_epi32(msg, 0x0E);
            state0      = _mm_sha256rnds2_epu32(state0, state1, msg);
        }
        state0 = _mm_add_epi32(state0, save0);
        state1 = _mm_add_epi32(state1, save1);
    }

    tmp    = _mm_shuffle_epi32(state0, 0x1B);
    state1 = _mm_shuffle_epi32(state1, 0xB1);
    _mm_storeu_si128((__m128i *) &state[0], _mm_blend_epi16(tmp, state1, 0xF0));
    _mm_storeu_si128((__m128i *) &state[4], _mm_alignr_epi8(state1, tmp, 8));
}
#endif

void compress(uint32_t * state, const uint8_t * data, size_t n_blocks) {
#ifdef LLAMA_SHA256_X86
    static const auto implementation = have_sha_ext() ? compress_sha_ext : compress_portable;
    implementation(state, data, n_blocks);
#else
    compress_portable(state, data, n_blocks);
#endif
}

}  // namespace

llama_sha256::llama_sha256() {
    state = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
}

void llama_sha256::update(const void * src, size_t len) {
    if (len == 0) {
        return;
    }

    const uint8_t * data = static_cast<const uint8_t *>(src);
    total_len += len;

    if (block_len > 0) {
        const size_t n = std::min(len, block.size() - block_len);
        memcpy(block.data() + block_len, data, n);
        block_len += n;
        data += n;
        len -= n;
        if (block_len < block.size()) {
            return;
        }
        compress(state.data(), block.data(), 1);
        block_len = 0;
    }
    // whole blocks are hashed where they are
    const size_t n_blocks = len / block.size();
    if (n_blocks > 0) {
        compress(state.data(), data, n_blocks);
        data += n_blocks * block.size();
        len -= n_blocks * block.size();
    }
    if (len > 0) {
        memcpy(block.data(), data, len);
        block_len = len;
    }
}

std::array<uint8_t, 32> llama_sha256::finish() {
    const uint64_t bit_len = total_len * 8;
    const uint8_t  one     = 0x80;
    update(&one, 1);

    const uint8_t zero = 0;
    while (block_len != 56) {
        update(&zero, 1);
    }

    uint8_t len_be[8];
    for (size_t i = 0; i < sizeof(len_be); ++i) {
        len_be[i] = uint8_t(bit_len >> (56 - 8 * i));
    }
    update(len_be, sizeof(len_be));

    std::array<uint8_t, 32> result;
    for (size_t i = 0; i < state.size(); ++i) {
        result[4 * i + 0] = uint8_t(state[i] >> 24);
        result[4 * i + 1] = uint8_t(state[i] >> 16);
        result[4 * i + 2] = uint8_t(state[i] >> 8);
        result[4 * i + 3] = uint8_t(state[i] >> 0);
    }
    return result;
}

std::array<uint8_t, 32> llama_sha256_digest(const void * data, size_t size) {
    llama_sha256 hash;
    hash.update(data, size);
    return hash.finish();
}
