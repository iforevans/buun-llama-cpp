#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include "../ggml/src/ggml-cuda/exl3.cuh"

bool ggml_cuda_marlin_q4_a32_supports_shape(int64_t n, int64_t k, int64_t m, int cc);
bool ggml_cuda_marlin_q8_g128_supports_shape(int64_t n, int64_t k, int64_t m, int cc);

// No GPU needed: validate host dispatch against the CUDA images actually built.
// In particular, SM75 PTX running on Ampere must not admit the BF16 stub.
int main() {
#ifndef __CUDA_ARCH_LIST__
    std::fprintf(stderr, "CUDA compiler does not expose its architecture list; skipping image-admission test\n");
    return 77;
#else
    bool ok = true;
    using supports_fn = bool (*)(int64_t, int64_t, int64_t, int);
    const supports_fn paths[] = {
        ggml_cuda_marlin_q4_a32_supports_shape,
        ggml_cuda_marlin_q8_g128_supports_shape,
    };
    for (const auto supports : paths) {
        for (int cc : {600, 700, 750, 800, 860, 890, 900, 1200}) {
            int compiled = -1;
            for (int image : {__CUDA_ARCH_LIST__}) {
                if (image <= cc && image > compiled) {
                    compiled = image;
                }
            }
            const bool expected = cc >= 800 && cc < 1200 && compiled >= 800;
            const bool actual = supports(256, 128, 1, cc);
            if (actual != expected || supports(255, 128, 1, cc) ||
                supports(256, 127, 1, cc) || supports(256, 128, 0, cc)) {
                std::fprintf(stderr, "Marlin admission mismatch: cc=%d compiled=%d expected=%d actual=%d\n",
                    cc, compiled, int(expected), int(actual));
                ok = false;
            }
        }
    }
    for (int cc : {600, 700, 750, 800, 860, 1200}) {
        for (int image : {600, 610, 700, 750, 800, 860}) {
            for (int m : {1, 8, 9, 16, 32, 33, 64, 128, 129}) {
                for (int n : {5119, 5120, 10239, 10240}) {
                    const bool expected = cc == 750 && image == 750 && m >= 9 &&
                        (m <= 32 || (m <= 128 && n >= 10240));
                    if (ggml_cuda_exl3_turing_gemm_supported(cc, image, m, n, 5120) != expected) {
                        std::fprintf(stderr, "EXL3 admission mismatch: cc=%d image=%d m=%d n=%d\n", cc, image, m, n);
                        ok = false;
                    }
                }
            }
        }
    }
    return ok ? 0 : 1;
#endif
}
