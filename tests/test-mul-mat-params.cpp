#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>

static void check_prec(const ggml_tensor * t, ggml_prec acc, ggml_prec src) {
    GGML_ASSERT(ggml_get_op_params_i32(t, 0) == acc);
    GGML_ASSERT(ggml_get_op_params_i32(t, 3) == src);
}

static void check_window(const ggml_tensor * t, int32_t lo, int32_t count) {
    GGML_ASSERT(ggml_mmid_window_lo(t) == lo);
    GGML_ASSERT(ggml_mmid_window_n_local(t) == count);
}

int main() {
    ggml_context * ctx = ggml_init({16 * ggml_tensor_overhead(), nullptr, true});
    GGML_ASSERT(ctx);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 3);
    ggml_tensor * acts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_tensor * mmid = ggml_mul_mat_id(ctx, weights, acts, ids);
    check_prec(mmid, GGML_PREC_UNDEFINED, GGML_PREC_UNDEFINED);
    check_window(mmid, 0, 0);

    for (const auto prec : {GGML_PREC_F32, GGML_PREC_BF16, GGML_PREC_F16, GGML_PREC_Q8, GGML_PREC_Q4}) {
        GGML_ASSERT(ggml_prec_set_src(mmid, prec, 1));
        GGML_ASSERT(ggml_prec_set_acc(mmid, GGML_PREC_F32));
        ggml_mul_mat_id_set_expert_window(mmid, 7, 3);
        check_prec(mmid, GGML_PREC_F32, prec);
        check_window(mmid, 7, 3);

        // Reverse setter order, then clear only the window on a CUDA-style shadow.
        GGML_ASSERT(ggml_prec_set_src(mmid, GGML_PREC_BF16, 1));
        check_window(mmid, 7, 3);
        ggml_tensor shadow = *mmid;
        ggml_mul_mat_id_set_expert_window(&shadow, 0, 0);
        check_prec(&shadow, GGML_PREC_F32, GGML_PREC_BF16);
        check_window(&shadow, 0, 0);
        check_window(mmid, 7, 3);

        // The meta backend sets bounds before wiring per-device source pointers.
        ggml_tensor shard = *mmid;
        shard.src[0] = nullptr;
        ggml_set_op_params_i32(&shard, GGML_MMID_WINDOW_LO, 10);
        ggml_set_op_params_i32(&shard, GGML_MMID_WINDOW_N_LOCAL, 3);
        check_prec(&shard, GGML_PREC_F32, GGML_PREC_BF16);
        check_window(&shard, 10, 3);
    }

    for (int idx = 0; idx < GGML_MAX_SRC; ++idx) {
        if (idx == 1) continue;
        const ggml_tensor before = *mmid;
        GGML_ASSERT(!ggml_prec_set_src(mmid, GGML_PREC_Q8, idx));
        GGML_ASSERT(memcmp(before.op_params, mmid->op_params, sizeof(mmid->op_params)) == 0);
    }
    GGML_ASSERT(ggml_mmid_expert_index(6, 7, 3) == -1);
    GGML_ASSERT(ggml_mmid_expert_index(7, 7, 3) == 0);
    GGML_ASSERT(ggml_mmid_expert_index(9, 7, 3) == 2);
    GGML_ASSERT(ggml_mmid_expert_index(10, 7, 3) == -1);
    GGML_ASSERT(ggml_mmid_expert_index(7, 0, 0) == 7);
    GGML_ASSERT(ggml_mmid_expert_index(-1, 0, 0) == -1);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    ggml_tensor * mm = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_hint(mm, GGML_HINT_FORCE_MMQ);
    GGML_ASSERT(ggml_prec_set_acc(mm, GGML_PREC_F32));
    GGML_ASSERT(ggml_prec_set_src(mm, GGML_PREC_Q8, 1));
    check_prec(mm, GGML_PREC_F32, GGML_PREC_Q8);
    GGML_ASSERT(ggml_get_op_params_i32(mm, 1) == GGML_HINT_FORCE_MMQ);

    ggml_tensor fa = {};
    fa.op = GGML_OP_FLASH_ATTN_EXT;
    GGML_ASSERT(ggml_prec_set_acc(&fa, GGML_PREC_F32));
    GGML_ASSERT(ggml_flash_attn_ext_get_prec(&fa) == GGML_PREC_F32);
    GGML_ASSERT(!ggml_prec_set_src(&fa, GGML_PREC_F16, 1));
    // FA's finite-row bound must never read a boolean hint as a bound of one.
    GGML_ASSERT(ggml_get_op_params_i32(&fa, 4) == 0);
    ggml_flash_attn_ext_set_sparse_mask(&fa, true);
    GGML_ASSERT(ggml_get_op_params_i32(&fa, 4) == 0);
    for (int32_t bound : {1, 640, 4096, 0}) {
        ggml_flash_attn_ext_set_n_kv_max(&fa, bound);
        GGML_ASSERT(ggml_flash_attn_ext_get_sparse_mask(&fa));
        ggml_flash_attn_ext_set_sparse_mask(&fa, false);
        GGML_ASSERT(ggml_get_op_params_i32(&fa, 4) == bound);
        ggml_flash_attn_ext_set_sparse_mask(&fa, true);
        GGML_ASSERT(ggml_get_op_params_i32(&fa, 4) == bound);
        GGML_ASSERT(ggml_flash_attn_ext_get_prec(&fa) == GGML_PREC_F32);
    }
    const ggml_tensor before = *a;
    GGML_ASSERT(!ggml_prec_set_acc(a, GGML_PREC_F32));
    GGML_ASSERT(!ggml_prec_set_src(a, GGML_PREC_F32, 1));
    GGML_ASSERT(memcmp(before.op_params, a->op_params, sizeof(a->op_params)) == 0);

    for (auto op : {GGML_OP_FLASH_ATTN_EXT, GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID,
                   GGML_OP_CUMSUM, GGML_OP_ARGSORT, GGML_OP_TOP_K}) {
        GGML_ASSERT(ggml_op_alloc_size_may_expand(op));
    }
    GGML_ASSERT(!ggml_op_alloc_size_may_expand(GGML_OP_ADD));
    ggml_free(ctx);
    puts("precision/window metadata and allocation cases: PASS");
}
