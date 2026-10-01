#pragma once

#include "../kernel-params.cuh"

template <typename Kernel>
static __global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void ggml_cuda_cutlass_indirect_kernel(const typename Kernel::Params * params) {
    extern __shared__ char smem[];
    ggml_cuda_acquire_kernel_params(params);
    Kernel{}(*params, smem);
}

// The SM120 provider uses ordinary, single-CTA GEMMs. Retain CUTLASS's argument
// validation and workspace initialization, but avoid its by-value host stub on
// MSVC. Both the descriptors and scratch belong to the caller's stream pool.
template <typename Kernel>
struct ggml_cuda_cutlass_indirect_adapter {
    using Direct    = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using Arguments = typename Kernel::Arguments;
    using Params    = typename Kernel::Params;

    static_assert(cute::size(typename Kernel::DispatchPolicy::ClusterShape{}) == 1,
                  "indirect launch requires a single-CTA cluster");
    static_assert(alignof(Params) <= 128, "CUDA pool alignment must cover kernel parameters");

    static cutlass::Status can_implement(const Arguments & args) {
        if (args.mode != cutlass::gemm::GemmUniversalMode::kGemm) {
            return cutlass::Status::kErrorNotSupported;
        }
        return Direct::can_implement(args);
    }

    static size_t param_offset(const Arguments & args) {
        const size_t bytes = Direct::get_workspace_size(args);
        return (bytes + 255) / 256 * 256;
    }

    static size_t get_workspace_size(const Arguments & args) {
        return param_offset(args) + (sizeof(Params) + 127) / 128 * 128;
    }

    cutlass::Status run(const Arguments & args, void * workspace, cudaStream_t stream) {
        if (!workspace) {
            return cutlass::Status::kErrorWorkspaceNull;
        }
        auto status = Kernel::initialize_workspace(args, workspace, stream, nullptr);
        if (status != cutlass::Status::kSuccess) {
            return status;
        }
        const auto params = Kernel::to_underlying_arguments(args, workspace);
        auto * device_params = reinterpret_cast<Params *>(static_cast<char *>(workspace) + param_offset(args));
        constexpr int smem_size = Kernel::SharedStorageSize;
        if (smem_size >= (48 << 10) &&
            cudaFuncSetAttribute(ggml_cuda_cutlass_indirect_kernel<Kernel>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) != cudaSuccess) {
            return cutlass::Status::kErrorInternal;
        }
        if (ggml_cuda_upload_kernel_params(params, device_params, stream) != cudaSuccess) {
            return cutlass::Status::kErrorInternal;
        }
        ggml_cuda_cutlass_indirect_kernel<Kernel>
            <<<Kernel::get_grid_shape(params), Kernel::get_block_shape(), smem_size, stream>>>(device_params);
        return cudaGetLastError() == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
    }
};

template <typename Kernel>
using ggml_cuda_cutlass_adapter =
#if defined(_MSC_VER)
    ggml_cuda_cutlass_indirect_adapter<Kernel>;
#else
    cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
#endif
