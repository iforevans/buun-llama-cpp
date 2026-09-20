#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

int main() {
    if (!ggml_backend_cuda_get_device_count()) return 77;
    auto * backend = ggml_backend_cuda_init(0);
    GGML_ASSERT(backend);
    uint32_t rng = 429;
    auto random = [&] { rng = rng * 1664525u + 1013904223u; return (int(rng >> 16) - 32768) / 32768.0f; };
    for (int m : {1,2,3,4,7,8,9,12,13,16,17}) for (int n : {7,48,64}) for (bool alias : {false,true}) {
      std::vector<float> reference;
      for (bool unfused : {true,false}) {
        rng = 429;
        constexpr int k = 5120;
        auto * ctx = ggml_init({1024*1024,nullptr,true});
        auto * wa = ggml_new_tensor_2d(ctx,GGML_TYPE_F16,k,n);
        auto * wb = ggml_new_tensor_2d(ctx,GGML_TYPE_F16,k,n);
        auto * x = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,m);
        auto * dt = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,1,n);
        auto * a = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,1,n);
        auto * alpha = ggml_mul_mat(ctx,wa,x);
        // An observable intermediate forces the same arithmetic through the
        // ordinary graph instead of eliding it in the gate fusion.
        if (unfused) ggml_set_output(alpha);
        alpha = ggml_reshape_3d(ctx,alpha,1,n,m);
        alpha = ggml_add(ctx,alpha,dt);
        alpha = ggml_softplus(ctx,alpha);
        alpha = ggml_mul(ctx,alpha,a);
        auto * gate = ggml_reshape_2d(ctx,alpha,n,m);
        auto * beta = ggml_mul_mat(ctx,wb,x);
        beta = ggml_reshape_3d(ctx,beta,1,n,m);
        beta = ggml_sigmoid(ctx,beta);
        auto * graph = ggml_new_graph_custom(ctx,128,false);
        ggml_build_forward_expand(graph,gate);
        ggml_build_forward_expand(graph,beta);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx,backend);
        GGML_ASSERT(buffer);
        if (alias) beta->data = x->data;
        std::vector<ggml_fp16_t> weights(k*n);
        for (auto * w : {wa,wb}) {
            for (auto & v : weights) v = ggml_fp32_to_fp16(random()*0.05f);
            ggml_backend_tensor_set(w,weights.data(),0,ggml_nbytes(w));
        }
        std::vector<float> input(k*m), offsets(n), factors(n);
        for (auto & v : input) v = random();
        for (int i=0;i<n;++i) { offsets[i]=(i%3-1)*20.0f; factors[i]=-0.1f-std::abs(random()); }
        ggml_backend_tensor_set(dt,offsets.data(),0,ggml_nbytes(dt));
        ggml_backend_tensor_set(a,factors.data(),0,ggml_nbytes(a));
        for (int repeat=0;repeat<3;++repeat) {
            ggml_backend_tensor_set(x,input.data(),0,ggml_nbytes(x));
            GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
            for (auto * out : {gate,beta}) {
                std::vector<float> values(n*m);
                ggml_backend_tensor_get(out,values.data(),0,ggml_nbytes(out));
                for (float v : values) GGML_ASSERT(std::isfinite(v));
                if (unfused) reference.insert(reference.end(),values.begin(),values.end());
                else {
                    const size_t offset = (2 * repeat + (out == beta)) * values.size();
                    GGML_ASSERT(std::memcmp(reference.data()+offset,values.data(),values.size()*sizeof(float)) == 0);
                }
            }
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
      }
      std::printf("PASS m=%d n=%d alias=%d\n",m,n,alias);
    }
    ggml_backend_free(backend);
}
