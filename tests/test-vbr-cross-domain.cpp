#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-turbo-meansub.h"
#include "ggml-vbr.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

// Independent host add-back oracle. Its only extra rounding is the F16
// residual readback; tolerance includes that rounding and the destination's
// encoding. This exercises the registered interface on CUDA AND HIP, with no
// model download or direct dependency on either backend's symbols.
static void run_case(ggml_backend_t backend, const ggml_vbr_backend_iface & iface,
                     const ggml_vbr_cross_domain_iface_v1 & cross,
                     ggml_type source_type, ggml_type target_type, bool value_side,
                     int64_t columns, bool in_place) {
    constexpr int64_t rows = 257; // full tile plus tail; reverse traversal on promotion
    constexpr int model_id = 1;
    int layers = 0, table_columns = 0, live = 0;
    const float * table = ggml_turbo_meansub_table(
        model_id, value_side, &layers, &table_columns, &live);
    check(table && live > 0 && columns <= table_columns, "missing baked mean table");
    const int64_t offset = columns < table_columns ? columns : 0;
    check(offset + columns <= table_columns, "invalid test shard");
    // Pick a nontrivial mean so forgetting add-back cannot pass the tolerance.
    int layer = 0;
    double energy = 0;
    for (int l = 0; l < layers; ++l) {
        double sum = 0;
        for (int64_t c = 0; c < columns; ++c) {
            const double m = table[l*table_columns + offset+c];
            sum += m*m;
        }
        if (sum > energy) { energy = sum; layer = l; }
    }
    check(energy > 0.01, "test requires a nonzero affine mean");
    const float * mean = table + layer*table_columns + offset;

    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({ 12*ggml_tensor_overhead(), nullptr, true }), ggml_free);
    check(bool(ctx), "context allocation");
    auto tensor = [&](ggml_type type) {
        return ggml_new_tensor_2d(ctx.get(), type, columns, rows);
    };
    auto * input = tensor(GGML_TYPE_F16);
    auto * source = tensor(source_type);
    auto * residual = tensor(GGML_TYPE_F16);
    auto * reference_input = tensor(GGML_TYPE_F16);
    auto * reference = tensor(target_type);
    auto * actual = tensor(target_type);
    auto * decoded = tensor(GGML_TYPE_F16);
    auto * expected = tensor(GGML_TYPE_F16);
    // The source is already a residual. A non-model name avoids subtracting a
    // baked mean while constructing these encoded test bytes.
    ggml_set_name(input, value_side ? "cache_v_l0" : "cache_k_l0");
    ggml_set_name(source, ggml_get_name(input));
    ggml_set_name(reference_input, ggml_get_name(input));
    ggml_set_name(actual, ggml_get_name(input));
    ggml_set_name(reference, ggml_get_name(input));
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    check(bool(buffer), "tensor allocation");
    const auto upload = [&](ggml_tensor * dst, const void * data, size_t size) {
        // The synchronous tensor-set API may use the model-upload ring. Direct
        // interface calls bypass graph entry's upload fence, so own this stream.
        ggml_backend_tensor_set_async(backend, dst, data, 0, size);
        ggml_backend_synchronize(backend);
    };
    std::vector<ggml_fp16_t> host(size_t(rows*columns));
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = ggml_fp32_to_fp16(float(0.7*std::sin(i*0.13) + 0.3*std::cos(i*0.037)));
    }
    upload(input, host.data(), ggml_nbytes(input));
    const auto transcode = [&](const ggml_tensor * src, ggml_tensor * dst) {
        const ggml_vbr_transcode_params p {
            src, dst->type, dst->data, dst->buffer, rows, value_side, nullptr, 0, 0,
        };
        iface.kv_transcode(backend, &p);
        ggml_backend_synchronize(backend);
    };
    transcode(input, source);
    // A full-domain destination includes the source TCQ V decode alpha. A
    // sink-stash capture instead normalizes to the same source tier, so it is
    // not the correct reference for a cross-domain V reconstruction.
    transcode(source, residual);
    ggml_backend_tensor_get(residual, host.data(), 0, ggml_nbytes(residual));
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(host[i]) + mean[i%columns]);
    }
    upload(reference_input, host.data(), ggml_nbytes(reference_input));
    if (target_type != GGML_TYPE_F16) {
        transcode(reference_input, reference);
    }

    ggml_tensor source_view = *source;
    if (in_place) {
        std::vector<uint8_t> packed(ggml_nbytes(source));
        ggml_backend_tensor_get(source, packed.data(), 0, packed.size());
        upload(actual, packed.data(), packed.size());
        source_view.data = actual->data;
        source_view.buffer = actual->buffer;
    }
    ggml_vbr_cross_domain_reconstruct_params p {
        { &source_view, target_type, actual->data, actual->buffer, rows,
          value_side, nullptr, 0, 0 }, model_id, layer, uint64_t(offset),
    };
    check(cross.kv_cross_domain_reconstruct(backend, &p), "cross-domain dispatch");
    ggml_backend_synchronize(backend);
    std::vector<ggml_fp16_t> output(host.size());
    if (target_type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(actual, output.data(), 0, ggml_nbytes(actual));
    } else {
        iface.kv_stash_capture(backend, actual, decoded->data, rows, value_side);
        iface.kv_stash_capture(backend, reference, expected->data, rows, value_side);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(decoded, output.data(), 0, ggml_nbytes(decoded));
        ggml_backend_tensor_get(expected, host.data(), 0, ggml_nbytes(expected));
    }
    double error = 0, norm = 0, worst = 0;
    size_t worst_index = 0;
    for (size_t i = 0; i < host.size(); ++i) {
        const double a = ggml_fp16_to_fp32(output[i]);
        const double b = ggml_fp16_to_fp32(host[i]);
        check(std::isfinite(a) && std::isfinite(b), "nonfinite reconstruction");
        error += (a-b)*(a-b);
        norm += b*b;
        if (std::abs(a-b) > worst) { worst = std::abs(a-b); worst_index = i; }
    }
    const double relative = std::sqrt(error/std::max(norm, 1e-30));
    std::printf("%s->%s side=%c cols=%lld inplace=%d rel_rms=%.6f max=%.6f\n",
        ggml_type_name(source_type), ggml_type_name(target_type), value_side ? 'V' : 'K',
        (long long) columns, in_place, relative, worst);
    if (relative > 0.004) {
        std::fprintf(stderr, "worst row=%zu col=%zu actual=%g expected=%g mean=%g layer=%d offset=%lld\n",
            worst_index/columns, worst_index%columns, ggml_fp16_to_fp32(output[worst_index]),
            ggml_fp16_to_fp32(host[worst_index]), mean[worst_index%columns], layer, (long long) offset);
    }
    check(relative < (target_type == GGML_TYPE_F16 ? 0.001 : 0.004), "reconstruction disagrees with host mean-addback oracle");
    p.logical_offset = uint64_t(table_columns);
    check(!cross.kv_cross_domain_reconstruct(backend, &p), "invalid shard accepted");
    p.logical_offset = uint64_t(offset);
    p.meansub_layer = layers;
    check(!cross.kv_cross_domain_reconstruct(backend, &p), "invalid layer accepted");
}

int main() {
    try {
        ggml_backend_load_all();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto device = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
            auto reg = ggml_backend_dev_backend_reg(device);
            auto get = reinterpret_cast<ggml_backend_vbr_iface_fn_t>(
                ggml_backend_reg_get_proc_address(reg, GGML_VBR_BACKEND_IFACE_PROC));
            auto get_cross = reinterpret_cast<ggml_backend_vbr_cross_domain_iface_v1_fn_t>(
                ggml_backend_reg_get_proc_address(reg, GGML_VBR_CROSS_DOMAIN_IFACE_V1_PROC));
            if (!get || !get_cross) { continue; }
            const auto * iface = get();
            const auto * cross = get_cross();
            check(iface && cross && cross->abi_version == GGML_VBR_CROSS_DOMAIN_IFACE_V1_VERSION &&
                  cross->struct_size >= sizeof(*cross), "invalid backend interface");
            std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
                ggml_backend_dev_init(device, nullptr), ggml_backend_free);
            check(bool(backend), "backend init");
            for (auto source : { GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_TCQ,
                                 GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO1_TCQ }) {
                for (auto target : { GGML_TYPE_F16, GGML_TYPE_TURBO8_0 }) {
                    for (bool value_side : { false, true }) {
                        for (int64_t columns : { 128, 256 }) {
                            for (bool in_place : { false, true }) {
                                run_case(backend.get(), *iface, *cross, source, target, value_side, columns, in_place);
                            }
                        }
                    }
                }
            }
            return 0;
        }
        std::puts("SKIP: no VBR cross-domain GPU backend");
        return 77;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
