#include "../ggml/src/ggml-cuda/common.cuh"

// Compile this probe with the same CUDA targets as the backend. The physical
// GPU alone does not tell a C++ test which specialization the binary contains.
bool test_cuda_exl3_sm86_image();

bool test_cuda_exl3_sm86_image() {
    return ggml_cuda_highest_compiled_arch(860) == 860;
}
