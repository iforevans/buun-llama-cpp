#pragma once

#include <cuda_runtime.h>

#include <cstring>
#include <type_traits>

// MSVC cannot pass over-aligned CUDA parameters by value. Pack their object
// representation into an unaligned launch argument, then unpack to device
// storage owned by the caller. Unlike an async copy from a host temporary, the
// upload kernel's arguments are also owned by a captured CUDA graph on replay.
template <size_t Size> struct ggml_cuda_kernel_param_bytes {
    unsigned char data[Size];
};

template <size_t Size>
static __global__ void ggml_cuda_unpack_kernel_params(ggml_cuda_kernel_param_bytes<Size> bytes,
                                                     unsigned char * dst) {
    for (size_t i = threadIdx.x; i < Size; i += blockDim.x) {
        dst[i] = bytes.data[i];
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
#endif
}

// TMA does not implicitly acquire device-written descriptors at kernel entry.
// The caller reserves a 128-byte-rounded, 128-byte-aligned parameter region;
// acquire every chunk, including descriptors nested inside CUTLASS Params.
template <typename Params>
static __device__ void ggml_cuda_acquire_kernel_params(const Params * params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    for (size_t offset = 0; offset < sizeof(Params); offset += 128) {
        const auto * chunk = reinterpret_cast<const unsigned char *>(params) + offset;
        asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(chunk) : "memory");
    }
#else
    (void) params;
#endif
}

template <typename Params>
static cudaError_t ggml_cuda_upload_kernel_params(const Params & params, Params * dst, cudaStream_t stream) {
    static_assert(std::is_trivially_copyable<Params>::value, "kernel parameters must be byte-copyable");
    // Keep the upload usable on pre-Volta devices as well.
    static_assert(sizeof(Params) + sizeof(void *) <= 4096, "kernel parameters exceed the launch argument limit");
    ggml_cuda_kernel_param_bytes<sizeof(Params)> bytes;
    std::memcpy(bytes.data, &params, sizeof(params));
    ggml_cuda_unpack_kernel_params<<<1, 128, 0, stream>>>(bytes, reinterpret_cast<unsigned char *>(dst));
    return cudaGetLastError();
}
