#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

#if defined(GGML_CUDA_USE_CUB) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
namespace {
// Keep score keys in registers across radix passes, then write the k survivors unordered.
constexpr int register_top_k_threads=1024;
__device__ uint32_t register_top_k_ordered(uint32_t b) {
    // CUB treats signed zeros as equivalent. Otherwise preserve radix bit order.
    if (!(b & 0x7fffffffU)) b=0;
    return b ^ ((b >> 31) ? 0xffffffffU : 0x80000000U);
}

template<int items>
__global__ __launch_bounds__(register_top_k_threads, 1)
void register_top_k_select(const float * values, int * dst, int n, int k) {
    using Scan=cub::BlockScan<int,register_top_k_threads>;
    __shared__ union { unsigned hist[32][256]; Scan::TempStorage scan; } scratch;
    __shared__ uint32_t prefix;
    __shared__ int rank;
    const int t=threadIdx.x, row=blockIdx.x;
    uint32_t keys[items];
#pragma unroll
    for(int j=0;j<items;++j) {
        int i=t*items+j;
        keys[j]=i<n ? register_top_k_ordered(__float_as_uint(values[size_t(row)*n+i])) : 0;
    }
    if(!t) {prefix=0;rank=k;}
    uint32_t mask=0;
    __syncthreads();
    for(int shift=24;shift>=0;shift-=8) {
        for(int h=t;h<32*256;h+=register_top_k_threads) scratch.hist[h/256][h%256]=0;
        __syncthreads();
        const uint32_t pre=prefix;
#pragma unroll
        for(int j=0;j<items;++j) {
            if(t*items+j<n && (keys[j]&mask)==pre)
                atomicAdd(&scratch.hist[t/32][(keys[j]>>shift)&255],1U);
        }
        __syncthreads();
        // thread t owns digit 255 - t; a suffix sum from the top digit finds the one holding the rank-th key
        int count=0;
        if(t<256) for(int w=0;w<32;++w) count+=scratch.hist[w][255-t];
        const int need=rank;
        __syncthreads();
        int above;
        Scan(scratch.scan).ExclusiveSum(count,above);
        if(t<256 && above<need && above+count>=need) {
            prefix=pre|uint32_t(255-t)<<shift;
            rank=need-above;
        }
        mask|=0xffU<<shift;
        __syncthreads();
    }
    int ng=0,ne=0;
#pragma unroll
    for(int j=0;j<items;++j) if(t*items+j<n) {ng+=keys[j]>prefix;ne+=keys[j]==prefix;}
    int bg,be,total;
    Scan(scratch.scan).ExclusiveSum(ng,bg,total);
    __syncthreads();
    Scan(scratch.scan).ExclusiveSum(ne,be);
    __syncthreads();
#pragma unroll
    for(int j=0;j<items;++j) {
        const int i=t*items+j;
        if(i>=n) continue;
        int pos=-1;
        if(keys[j]>prefix) pos=bg++;
        else if(keys[j]==prefix) pos=total+be++;
        if(pos>=0 && pos<k) dst[size_t(row)*k+pos]=i;
    }
}
void register_top_k_launch(const float * values,int * dst,int n,int rows,int k,cudaStream_t stream) {
    if(n<=8192)register_top_k_select<8><<<rows,register_top_k_threads,0,stream>>>(values,dst,n,k);
    else if(n<=16384)register_top_k_select<16><<<rows,register_top_k_threads,0,stream>>>(values,dst,n,k);
    else if(n<=32768)register_top_k_select<32><<<rows,register_top_k_threads,0,stream>>>(values,dst,n,k);
    else if(n<=40960)register_top_k_select<40><<<rows,register_top_k_threads,0,stream>>>(values,dst,n,k);
    else GGML_ABORT("register top-k shape exceeds dispatch bound");
}

// One block per row: from four rows up this beats a DeviceTopK launch chain per row (25x on
// 4352-row prefill), while a lone decode row leaves the device idle and stays on DeviceTopK.
// Without a race-free DeviceTopK (CCCL < 3.4.3) the fallback is a full segmented argsort, which loses at any row count.
bool register_top_k_wins(int64_t columns, int64_t rows, int64_t k) {
#ifdef CUB_TOP_K_AVAILABLE
    const int64_t min_rows = 4;
#else
    const int64_t min_rows = 1;
#endif
    return columns > 1024 && columns <= 40960 && rows >= min_rows && rows <= 8192 && k >= 1 && k <= 4096 && k <= columns;
}

} // namespace
#endif

static constexpr int stable_top_k_chunk = 4096;

static __global__ void stable_top_k_collect_ties(
        const float * src,
        const int * dst,
        int * tie_ids,
        int * tie_counts,
        int ncols,
        int k,
        int n_chunks) {
    const int work = blockIdx.x;
    const int row = work / n_chunks;
    const int chunk = work - row * n_chunks;
    const float * row_src = src + row * ncols;
    const int * row_dst = dst + row * k;
    const int lane = threadIdx.x;

    float threshold = FLT_MAX;
    if (lane == 0) {
        for (int i = 0; i < k; ++i) {
            threshold = min(threshold, row_src[row_dst[i]]);
        }
    }
    threshold = __shfl_sync(0xffffffffULL, threshold, 0);

    const int begin = chunk * stable_top_k_chunk;
    const int end = min(begin + stable_top_k_chunk, ncols);
    int stored = 0;
    int * out = tie_ids + (size_t) work * k;
    for (int base = begin; base < end; base += warpSize) {
        const int id = base + lane;
        const unsigned long long mask = __ballot_sync(0xffffffffULL,
                id < end && row_src[id] == threshold);
        const int n_match = __popcll(mask);
        const int rank = __popcll(mask & ((1ULL << lane) - 1ULL));
        if ((mask & (1ULL << lane)) && stored + rank < k) {
            out[stored + rank] = id;
        }
        stored += n_match;
    }
    if (lane == 0) {
        tie_counts[work] = min(stored, k);
    }
}

static __global__ void stable_top_k_finalize(
        const float * src,
        int * dst,
        const int * tie_ids,
        const int * tie_counts,
        int ncols,
        int k,
        int n_chunks) {
    const int row = blockIdx.x;
    const float * row_src = src + row * ncols;
    int * row_dst = dst + row * k;
    __shared__ int ids[64];

    if (threadIdx.x == 0) {
        float threshold = FLT_MAX;
        for (int i = 0; i < k; ++i) {
            threshold = min(threshold, row_src[row_dst[i]]);
        }

        int n_ids = 0;
        for (int i = 0; i < k; ++i) {
            const int id = row_dst[i];
            if (row_src[id] > threshold) {
                ids[n_ids++] = id;
            }
        }
        for (int chunk = 0; chunk < n_chunks && n_ids < k; ++chunk) {
            const int work = row * n_chunks + chunk;
            const int n_ties = tie_counts[work];
            const int * chunk_ids = tie_ids + (size_t) work * k;
            for (int i = 0; i < n_ties && n_ids < k; ++i) {
                ids[n_ids++] = chunk_ids[i];
            }
        }

        for (int i = 1; i < k; ++i) {
            const int id = ids[i];
            const float value = row_src[id];
            int j = i - 1;
            while (j >= 0) {
                const int prev_id = ids[j];
                const float prev = row_src[prev_id];
                if (prev > value || (prev == value && prev_id < id)) {
                    break;
                }
                ids[j + 1] = prev_id;
                --j;
            }
            ids[j + 1] = id;
        }
        for (int i = 0; i < k; ++i) {
            row_dst[i] = ids[i];
        }
    }
}

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols,
                         k, env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if defined(CUB_TOP_K_AVAILABLE) || (!defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP))

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

#endif // Batched radix selection: CUDA with DeviceTopK, or HIP without CUB.

#if defined(GGML_USE_HIP)
// hipMemcpy2DAsync can reject valid VMM-backed graph suballocations. Pack the
// selected prefix of each sorted row directly, without changing its ordering.
static __global__ void top_k_copy_indices(const int * src, int * dst, int64_t ncols, int64_t k, int64_t count) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        dst[i] = src[(i / k) * ncols + i % k];
    }
}
#endif

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    const bool       stable = ggml_top_k_is_stable(dst);
    ggml_cuda_pool & pool  = ctx.pool();
    GGML_ASSERT(!stable || k <= 64);
#if defined(GGML_CUDA_USE_CUB) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (!stable && register_top_k_wins(ncols, nrows, k)) {
        register_top_k_launch(src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }
#endif
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    // SM86 speculative batches amortize radix passes across rows. Bound their
    // per-row workspace; keep DeviceTopK for tiny batches and bulk prefill.
    if (ggml_cuda_info().devices[ctx.device].cc == 860 && nrows >= 7 && nrows <= 32 && ncols >= 65536 && k <= 64) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
        for (int i = 0; i < nrows; i++) {
            top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
        }
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
#if defined(GGML_USE_HIP)
        top_k_copy_indices<<<(k * nrows + 255) / 256, 256, 0, stream>>>(tmp_dst, dst_d, ncols, k, k * nrows);
#else
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#endif
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
    if (stable) {
        const int n_chunks = (ncols + stable_top_k_chunk - 1) / stable_top_k_chunk;
        const size_t n_work = (size_t) nrows * n_chunks;
        ggml_cuda_pool_alloc<int> tie_ids_alloc(pool, n_work * k);
        ggml_cuda_pool_alloc<int> tie_counts_alloc(pool, n_work);
        stable_top_k_collect_ties<<<n_work, WARP_SIZE, 0, stream>>>(
                (const float *) src0->data, (const int *) dst->data,
                tie_ids_alloc.get(), tie_counts_alloc.get(), ncols, k, n_chunks);
        stable_top_k_finalize<<<nrows, 1, 0, stream>>>(
                (const float *) src0->data, (int *) dst->data,
                tie_ids_alloc.get(), tie_counts_alloc.get(), ncols, k, n_chunks);
    }
}
