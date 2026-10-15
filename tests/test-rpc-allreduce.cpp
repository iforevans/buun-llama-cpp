// Run with two loopback rpc-server endpoints. The large case distinguishes
// full-precision replicated sums from rounding only the remote rank to BF16.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-rpc.h"

#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    if (argc == 3 && std::strcmp(argv[1], "--server") == 0) {
        auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(cpu);
        ggml_backend_rpc_start_server(argv[2], nullptr, 1, 1, &cpu);
        return 0;
    }
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s endpoint0 endpoint1 | --server endpoint\n", argv[0]);
        return 1;
    }
    ggml_backend_ptr ranks[] = {
        ggml_backend_ptr(ggml_backend_rpc_init(argv[1], 0)),
        ggml_backend_ptr(ggml_backend_rpc_init(argv[2], 0)),
    };
    GGML_ASSERT(ranks[0] && ranks[1]);
    ggml_backend_t backends[] = {ranks[0].get(), ranks[1].get()};
    auto reg = ggml_backend_rpc_reg();
    auto init = reinterpret_cast<ggml_backend_comm_init_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init"));
    auto reduce = reinterpret_cast<ggml_backend_comm_allreduce_tensor_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor"));
    auto free_comm = reinterpret_cast<ggml_backend_comm_free_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free"));
    GGML_ASSERT(init && reduce && free_comm);
    void * comm = init(backends, 2);
    GGML_ASSERT(comm);

    // A packed transpose occupies contiguous storage but has different logical
    // strides. Refuse it before dispatch: the server reduction uses contiguous
    // scratch/output tensors and cannot safely reduce this view in place.
    {
        ggml_context_ptr contexts[] = {
            ggml_context_ptr(ggml_init({2*ggml_tensor_overhead(), nullptr, true})),
            ggml_context_ptr(ggml_init({2*ggml_tensor_overhead(), nullptr, true})),
        };
        ggml_tensor * bases[2];
        ggml_tensor * transposed[2];
        ggml_backend_buffer_ptr buffers[2];
        const std::vector<float> original[] = {{1, 2, 3, 4}, {11, 13, 17, 19}};
        for (int rank = 0; rank < 2; ++rank) {
            bases[rank] = ggml_new_tensor_2d(contexts[rank].get(), GGML_TYPE_F32, 2, 2);
            transposed[rank] = ggml_transpose(contexts[rank].get(), bases[rank]);
            buffers[rank].reset(ggml_backend_alloc_ctx_tensors(contexts[rank].get(), backends[rank]));
            GGML_ASSERT(buffers[rank]);
            bases[rank]->flags |= GGML_TENSOR_FLAG_COMPUTE;
            transposed[rank]->flags |= GGML_TENSOR_FLAG_COMPUTE;
            GGML_ASSERT(ggml_is_contiguously_allocated(transposed[rank]));
            GGML_ASSERT(!ggml_is_contiguous(transposed[rank]));
            ggml_backend_tensor_set(bases[rank], original[rank].data(), 0, 4*sizeof(float));
        }
        for (int mask : {1, 2, 3}) {
            ggml_tensor * tensors[] = {
                mask & 1 ? transposed[0] : bases[0],
                mask & 2 ? transposed[1] : bases[1],
            };
            GGML_ASSERT(!reduce(comm, tensors));
            for (int rank = 0; rank < 2; ++rank) {
                std::vector<float> output(4);
                ggml_backend_synchronize(backends[rank]);
                ggml_backend_tensor_get(bases[rank], output.data(), 0, 4*sizeof(float));
                GGML_ASSERT(output == original[rank]);
            }
        }
        std::puts("packed transpose refused on either rank without mutation: PASS");
    }

    bool ok = true;
    for (int count : {32767, 32768, 32769, 65536}) {
        ggml_context_ptr contexts[] = {
            ggml_context_ptr(ggml_init({ggml_tensor_overhead(), nullptr, true})),
            ggml_context_ptr(ggml_init({ggml_tensor_overhead(), nullptr, true})),
        };
        ggml_tensor * tensors[] = {
            ggml_new_tensor_1d(contexts[0].get(), GGML_TYPE_F32, count),
            ggml_new_tensor_1d(contexts[1].get(), GGML_TYPE_F32, count),
        };
        ggml_backend_buffer_ptr buffers[] = {
            ggml_backend_buffer_ptr(ggml_backend_alloc_ctx_tensors(contexts[0].get(), backends[0])),
            ggml_backend_buffer_ptr(ggml_backend_alloc_ctx_tensors(contexts[1].get(), backends[1])),
        };
        GGML_ASSERT(buffers[0] && buffers[1]);
        std::vector<float> expected(count), input(count), output(count);
        for (int rank = 0; rank < 2; ++rank) {
            tensors[rank]->flags |= GGML_TENSOR_FLAG_COMPUTE;
            for (int i = 0; i < count; ++i) {
                input[i] = rank == 0 ? 1.0009765625f + (i % 13)*0.0001220703125f : -0.998046875f;
                expected[i] += input[i];
            }
            ggml_backend_tensor_set(tensors[rank], input.data(), 0, count*sizeof(float));
        }
        GGML_ASSERT(reduce(comm, tensors));
        for (int rank = 0; rank < 2; ++rank) {
            ggml_backend_synchronize(backends[rank]);
            ggml_backend_tensor_get(tensors[rank], output.data(), 0, count*sizeof(float));
            ok = ok && output == expected;
        }
        std::printf("count=%d exact replicated F32 sum: %s\n", count, ok ? "PASS" : "FAIL");
    }
    free_comm(comm);
    return ok ? 0 : 1;
}
