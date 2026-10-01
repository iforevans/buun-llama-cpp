#include "../ggml/src/ggml-cuda/kernel-params.cuh"

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK_TEST(expr) do { \
    const cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

// Covers a descriptor-sized object and nested, CUTLASS-sized launch arguments.
template <size_t Size> struct alignas(128) test_params {
    unsigned char bytes[Size];
};

template <size_t Size>
static __global__ void consume(const test_params<Size> * params, unsigned char * output) {
    ggml_cuda_acquire_kernel_params(params);
    for (size_t i = threadIdx.x; i < Size; i += blockDim.x) {
        output[i] = params->bytes[i];
    }
}

template <size_t Size>
static void submit(test_params<Size> * params, unsigned char * output, cudaStream_t stream, unsigned char seed) {
    test_params<Size> host;
    for (size_t i = 0; i < Size; ++i) {
        host.bytes[i] = static_cast<unsigned char>(seed + i);
    }
    CUDA_CHECK_TEST(ggml_cuda_upload_kernel_params(host, params, stream));
    consume<<<1, 128, 0, stream>>>(params, output);
    CUDA_CHECK_TEST(cudaGetLastError());
    // This source is deliberately temporary, including during graph capture.
}

template <size_t Size>
static void verify(unsigned char * output, cudaStream_t stream, unsigned char seed) {
    unsigned char host[Size];
    CUDA_CHECK_TEST(cudaMemcpyAsync(host, output, Size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_TEST(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < Size; ++i) {
        if (host[i] != static_cast<unsigned char>(seed + i)) {
            std::fprintf(stderr, "parameter mismatch: size=%zu offset=%zu\n", Size, i);
            std::exit(1);
        }
    }
}

template <size_t Size>
static void test() {
    cudaStream_t streams[2];
    test_params<Size> * params[2];
    unsigned char * outputs[2];
    cudaGraph_t graphs[2];
    cudaGraphExec_t execs[2];
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK_TEST(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        CUDA_CHECK_TEST(cudaMalloc(&params[i], sizeof(test_params<Size>)));
        CUDA_CHECK_TEST(cudaMalloc(&outputs[i], 2 * Size));
        submit(params[i], outputs[i], streams[i], 17 + i);
    }
    for (int i = 0; i < 2; ++i) {
        verify<Size>(outputs[i], streams[i], 17 + i);
        CUDA_CHECK_TEST(cudaStreamBeginCapture(streams[i], cudaStreamCaptureModeThreadLocal));
        submit(params[i], outputs[i], streams[i], 83 + i);
        // A stream pool can recycle the same parameter allocation between nodes.
        submit(params[i], outputs[i] + Size, streams[i], 123 + i);
        CUDA_CHECK_TEST(cudaStreamEndCapture(streams[i], &graphs[i]));
        CUDA_CHECK_TEST(cudaGraphInstantiate(&execs[i], graphs[i], nullptr, nullptr, 0));
    }
    // Reuse each parameter allocation with different contents before replay.
    // Graph replay must restore the captured bytes, not the last host temporary.
    for (int repeat = 0; repeat < 16; ++repeat) {
        for (int i = 0; i < 2; ++i) {
            submit(params[i], outputs[i], streams[i], repeat + i);
            CUDA_CHECK_TEST(cudaGraphLaunch(execs[i], streams[i]));
        }
        for (int i = 0; i < 2; ++i) {
            verify<Size>(outputs[i], streams[i], 83 + i);
            verify<Size>(outputs[i] + Size, streams[i], 123 + i);
        }
    }
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK_TEST(cudaGraphExecDestroy(execs[i]));
        CUDA_CHECK_TEST(cudaGraphDestroy(graphs[i]));
        CUDA_CHECK_TEST(cudaFree(outputs[i]));
        CUDA_CHECK_TEST(cudaFree(params[i]));
        CUDA_CHECK_TEST(cudaStreamDestroy(streams[i]));
    }
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        return 77;
    }
    test<128>();
    test<2048>();
    std::puts("kernel parameter upload: eager, two streams, and graph replay passed");
}
