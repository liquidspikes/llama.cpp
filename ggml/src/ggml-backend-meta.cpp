#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// N-split ssm_out: HIP shard GEMM is bit-exact vs full-W when dest is a *native*
// [K, N/2] tensor (hip-ssm-gemm.cpp NMSE 0). Shrinking a 2560 dest in place
// is not (English loop). Own 1280 buffer + host allgather into the 2560 PARTIAL.
static std::unordered_map<ggml_tensor *, ggml_tensor *> g_innern_gemm;
static std::vector<ggml_backend_buffer_ptr> g_innern_bufs;
// Split-GDN activation AllGather: 3072 3-rep shard → sequential 6144 dest for ssm_out.
static std::unordered_map<ggml_tensor *, ggml_tensor *> g_gdn_x_full;
static std::vector<ggml_backend_buffer_ptr> g_gdn_x_bufs;

// Unallocated / poisoned shard pointers (qwen4exp Flash Next TP SIGSEGV:
// dest 0x555500000032, memset ~29GB). Odd or near-null addresses are never
// valid GPU/host buffers on this path; 16-byte alignment matches malloc/HIP.
static bool ggml_backend_meta_data_ptr_ok(const void * p) {
    const uintptr_t u = (uintptr_t) p;
    return p != nullptr && u >= 0x10000ull && (u & 0xf) == 0;
}

static bool ggml_backend_meta_resid_name(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    return strstr(name, "hc_combine") != nullptr ||
           strstr(name, "hc_mixed") != nullptr ||
           strstr(name, "ffn_moe_out") != nullptr ||
           strstr(name, "l_last") != nullptr ||
           strstr(name, "ple_conv") != nullptr ||
           strstr(name, "linear_attn_out") != nullptr ||
           strstr(name, "innern_gemm") != nullptr;
}

static void ggml_backend_meta_dump_resid(const char * tag, size_t j, ggml_tensor * t) {
    if (t == nullptr) {
        fprintf(stderr, "[RESID] %s j=%zu name=(null)\n", tag, j);
        return;
    }
    if (t->data == nullptr || !ggml_backend_meta_data_ptr_ok(t->data) || t->type != GGML_TYPE_F32) {
        fprintf(stderr, "[RESID] %s j=%zu name=%s op=%s type=%s ne=[%lld,%lld,%lld] data=%p\n",
                tag, j, t->name, ggml_op_name(t->op), ggml_type_name(t->type),
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], t->data);
        return;
    }
    float x[2] = {0, 0};
    const size_t n = std::min(sizeof(x), ggml_nbytes(t));
    ggml_backend_tensor_get(t, x, 0, n);
    fprintf(stderr, "[RESID] %s j=%zu name=%s op=%s ne=[%lld,%lld,%lld] x0=%.5f,%.5f\n",
            tag, j, t->name, ggml_op_name(t->op),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], x[0], x[1]);
}

// QWEN4EXP HC mix: [n_embd, T] strided view of one hc stream of [n_embd, hc, T].
// src->nb[1] == n_embd * esz (one contiguous stream). hc is small (Flash-Next=4).
// Rejects QSA/GDN views such as [64,4] of [64,64,4].
static bool ggml_backend_meta_is_hc_stream_view(const ggml_tensor * t) {
    if (t == nullptr || t->op != GGML_OP_VIEW || t->src[0] == nullptr) {
        return false;
    }
    const ggml_tensor * s = t->src[0];
    const size_t esz = ggml_type_size(t->type);
    return s->ne[1] > 1 && s->ne[1] <= 16 &&
           t->ne[0] == s->ne[0] && t->ne[1] == s->ne[2] &&
           t->nb[1] == s->nb[2] && s->nb[1] == (size_t) t->ne[0] * esz &&
           s->nb[1] > 0 && t->view_offs % s->nb[1] == 0;
}

// Walk VIEW parents to the GATED_DELTA_NET that owns this tensor, if any.
static const ggml_tensor * ggml_backend_meta_gdn_of(const ggml_tensor * t) {
    while (t != nullptr && t->op == GGML_OP_VIEW) {
        t = t->view_src != nullptr ? t->view_src : t->src[0];
    }
    if (t == nullptr || t->op != GGML_OP_GATED_DELTA_NET || t->src[2] == nullptr) {
        return nullptr;
    }
    return t;
}

// Snapshot tail of ggml_gated_delta_net: columns after the attn scores.
// 4D [S_v, S_v, H, n_seqs] or packed [S_v*S_v*H, n_seqs, n_written].
// handle_reshape of AXIS_0 packed S_v*H lands on the inner S_v (AXIS_1) because
// 128*128 > S_v*H; the split must follow heads (V axis 1) or cache_s is scrambled.
static bool ggml_backend_meta_is_gdn_state_view(const ggml_tensor * t, const ggml_tensor ** gdn_out) {
    if (t == nullptr || t->op != GGML_OP_VIEW) {
        return false;
    }
    const ggml_tensor * gdn = ggml_backend_meta_gdn_of(t);
    if (gdn == nullptr) {
        return false;
    }
    const ggml_tensor * v = gdn->src[2];
    const int64_t S_v      = v->ne[0];
    const int64_t H        = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs   = v->ne[3];
    const size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);
    if (tail_off == 0 || t->view_offs != tail_off) {
        return false;
    }
    const bool shape_4d = t->ne[0] == S_v && t->ne[1] == S_v && t->ne[2] == H;
    const bool shape_D  = t->ne[0] == S_v * S_v * H;
    if (!shape_4d && !shape_D) {
        return false;
    }
    if (gdn_out != nullptr) {
        *gdn_out = gdn;
    }
    return true;
}

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
        /* .mmap_support          = */ true,
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    if (meta_dev_ctx->simple_devs.empty()) {
        return false;
    }
    const bool ok = std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
    if (ok) {
        return true;
    }
    // RPC supports_op can reject ADD of HC stream VIEWs (srcs do not
    // round-trip) while the local HIP device accepts them. all_of then
    // forces the sched onto CPU, mix never runs on Meta, NMSE ~1e2.
    // Both GPUs are the same HIP; if the first device can run it, TP can.
    return ggml_backend_dev_supports_op(meta_dev_ctx->simple_devs[0], op);
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
            if (ggml_backend_dev_supports_buft(simple_dev, buft)) {
                return true;
            }
        }
        return false;
    }
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

static size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
struct ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params, const int n_simple) {
        ctxs.reserve(n_simple);
        for (int i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

struct ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    int stc_compute_index      = 0;
    int stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;

    int debug;

    // Device-local zero row for HC stream views this device does not own.
    std::vector<ggml_backend_buffer_ptr> hc_zero_bufs;
    std::vector<void *>                  hc_zero_ptrs;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            const std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return stc_static;
        }
        return stc_compute[stc_compute_index];
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    if (tensor == nullptr) {
        return nullptr;
    }
    if (tensor->buffer == nullptr || !ggml_backend_buffer_is_meta(tensor->buffer)) {
        if (tensor->view_src != nullptr && tensor->view_src->buffer != nullptr &&
                ggml_backend_buffer_is_meta(tensor->view_src->buffer)) {
            tensor = tensor->view_src;
        } else {
            return nullptr;
        }
    }
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    if (index >= buf_ctx->bufs.size()) {
        return nullptr;
    }

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    if (it == stc.simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    if (tensor == nullptr) {
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    }
    if (tensor->buffer == nullptr || !ggml_backend_buffer_is_meta(tensor->buffer)) {
        if (tensor->view_src != nullptr && tensor->view_src->buffer != nullptr &&
                ggml_backend_buffer_is_meta(tensor->view_src->buffer)) {
            tensor = tensor->view_src;
        } else {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
    }
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                // Elementwise: a fully replicated operand does not change the
                // other operand's split (Qwen3.5 GDN gate = softplus × ssm_a).
                if (ret.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                        src_ss[i].axis >= 0 && src_ss[i].axis < GGML_MAX_DIMS) {
                    ret = src_ss[i];
                } else if (src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                        ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                    // keep ret
                } else if (ret.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                        src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    ret = src_ss[i];
                } else if (src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                        ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    // keep PARTIAL
                } else {
                    ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                    break;
                }
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            // No live sources (FILL/ARANGE/etc): every device holds the full tensor.
            ret = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
            GGML_LOG_ERROR("%s: UNKNOWN split for %s [%s] scalar_only=%d\n",
                           __func__, tensor->name, ggml_op_name(tensor->op), (int) scalar_only);
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    continue;
                }
                GGML_LOG_ERROR("%s:   src%zu %s [%s] axis=%d\n",
                               __func__, i, tensor->src[i]->name, ggml_op_name(tensor->src[i]->op),
                               (int) src_ss[i].axis);
            }
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        // Swapped broadcast: src0 replicated, src1 split (gate-0 = a_softplus × ssm_a).
        if (src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS &&
                src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        // HC mix: ADD of per-device stream views (each device holds a subset of
        // hc). The sum is PARTIAL until AllReduce. Delay-AR walks this chain.
        if (tensor->op == GGML_OP_ADD) {
            const bool hc0 = ggml_backend_meta_is_hc_stream_view(tensor->src[0]);
            const bool hc1 = ggml_backend_meta_is_hc_stream_view(tensor->src[1]);
            auto gated_hc_split = [&](const ggml_tensor * view) -> bool {
                if (view == nullptr || view->src[0] == nullptr) {
                    return false;
                }
                const ggml_backend_meta_split_state gss =
                    ggml_backend_meta_get_split_state(stc, view->src[0], assume_sync);
                return gss.axis == GGML_BACKEND_SPLIT_AXIS_1;
            };
            if ((hc0 && hc1 && (gated_hc_split(tensor->src[0]) || gated_hc_split(tensor->src[1]))) ||
                    (hc0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) ||
                    (hc1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)) {
                if (getenv("LLAMA_TP_LOG_HC")) {
                    fprintf(stderr, "[HC_ADD] PARTIAL assume_sync=%d hc0=%d hc1=%d\n",
                            (int) assume_sync, (int) hc0, (int) hc1);
                }
                return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
            }
        }
        // HC combine: wo is PARTIAL (AllReduced in the previous subgraph), then
        // REPEAT/MUL/ADD. Typing ADD(MIRRORED residual, PARTIAL) as PARTIAL
        // poisons the [n_embd,hc,T] residual. After the wo AllReduce both
        // buffers are full — keep MIRRORED.
        if ((src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                    src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) ||
            (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                    src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        // Unfused GDN chunking: ADD(TRI head-split, DIAG identity). The
        // identity is the same on every head; keep the TRI's split.
        if (tensor->op == GGML_OP_ADD &&
                src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS &&
                src_ss[0].axis != src_ss[1].axis) {
            auto is_diag = [](const ggml_tensor * t) -> bool {
                return t != nullptr && (t->op == GGML_OP_DIAG ||
                        (t->src[0] != nullptr && t->src[0]->op == GGML_OP_DIAG));
            };
            if (is_diag(tensor->src[1])) {
                return src_ss[0];
            }
            if (is_diag(tensor->src[0])) {
                return src_ss[1];
            }
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        static const bool nspl_mm = getenv("LLAMA_TP_SSM_OUT_NSPLIT") != nullptr;
        if (nspl_mm && tensor->src[0] != nullptr &&
                strstr(tensor->src[0]->name, "ssm_out") != nullptr &&
                src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 &&
                (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                 src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL ||
                 src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0)) {
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED
                                : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            // Column-parallel ssm_out: full-K GEMM into an N-slice, zero-pad,
            // AllReduce-sum concatenates. Do not treat qkv this way (head split).
            static const bool nspl = getenv("LLAMA_TP_SSM_OUT_NSPLIT") != nullptr;
            if (nspl && tensor->src[0] != nullptr &&
                    strstr(tensor->src[0]->name, "ssm_out") != nullptr) {
                return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED
                                    : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
            }
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            // Keep n_segments/nr. QWEN4EXP qkv/ssm_out use nr=5/3 of 2048 so each
            // device gets Q,K and each V-group half (V-order), not a sequential
            // first-half of 10240 (that is Q+K+V[:8] on device 0).
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        // Inner-K weight split, replicated activation (ssm_out × mirrored GDN out).
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            if (!split_states_equal(src_ss[0], src_ss[1])) {
                fprintf(stderr, "[META_MUL] AXIS_0 size mismatch node=%s src0=%s ne=[%lld,%lld,%lld] src1=%s ne=[%lld,%lld,%lld]\n",
                        tensor->name, tensor->src[0]->name,
                        (long long) tensor->src[0]->ne[0], (long long) tensor->src[0]->ne[1], (long long) tensor->src[0]->ne[2],
                        tensor->src[1]->name,
                        (long long) tensor->src[1]->ne[0], (long long) tensor->src[1]->ne[1], (long long) tensor->src[1]->ne[2]);
                for (size_t j = 0; j < n_bufs; j++) {
                    int64_t a = 0, b = 0;
                    for (size_t s = 0; s < src_ss[0].n_segments; s++) {
                        a += src_ss[0].ne[s * n_bufs + j] * src_ss[0].nr[s];
                    }
                    for (size_t s = 0; s < src_ss[1].n_segments; s++) {
                        b += src_ss[1].ne[s * n_bufs + j] * src_ss[1].nr[s];
                    }
                    fprintf(stderr, "[META_MUL]   j=%zu src0_sum=%lld src1_sum=%lld nseg=%u/%u\n",
                            j, (long long) a, (long long) b, src_ss[0].n_segments, src_ss[1].n_segments);
                }
                GGML_ABORT("mul_mat AXIS_0 split sizes differ");
            }
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            if (!split_states_equal(src_ss[0], src_ss[1])) {
                fprintf(stderr, "[META_MUL] AXIS_%d size mismatch node=%s src0=%s op=%s ne=[%lld,%lld,%lld,%lld] src1=%s op=%s ne=[%lld,%lld,%lld,%lld]\n",
                        (int) src_ss[0].axis, tensor->name,
                        tensor->src[0]->name, ggml_op_name(tensor->src[0]->op),
                        (long long) tensor->src[0]->ne[0], (long long) tensor->src[0]->ne[1],
                        (long long) tensor->src[0]->ne[2], (long long) tensor->src[0]->ne[3],
                        tensor->src[1]->name, ggml_op_name(tensor->src[1]->op),
                        (long long) tensor->src[1]->ne[0], (long long) tensor->src[1]->ne[1],
                        (long long) tensor->src[1]->ne[2], (long long) tensor->src[1]->ne[3]);
                for (size_t j = 0; j < n_bufs; j++) {
                    int64_t a = 0, b = 0;
                    for (size_t s = 0; s < src_ss[0].n_segments; s++) {
                        a += src_ss[0].ne[s * n_bufs + j] * src_ss[0].nr[s];
                    }
                    for (size_t s = 0; s < src_ss[1].n_segments; s++) {
                        b += src_ss[1].ne[s * n_bufs + j] * src_ss[1].nr[s];
                    }
                    fprintf(stderr, "[META_MUL]   j=%zu src0_sum=%lld src1_sum=%lld nseg=%u/%u\n",
                            j, (long long) a, (long long) b, src_ss[0].n_segments, src_ss[1].n_segments);
                }
                GGML_ABORT("mul_mat AXIS_2+ split sizes differ");
            }
            return src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        if (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis, tensor->src[1]->name, (int) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto gdn_state_view_split = [&](const ggml_tensor * gdn) -> ggml_backend_meta_split_state {
        const ggml_tensor * v = gdn->src[2];
        const int64_t S_v = v->ne[0];
        const int64_t H   = v->ne[1];
        const ggml_backend_meta_split_state vss =
            ggml_backend_meta_get_split_state(stc, v, /*assume_sync =*/ true);
        if (vss.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || vss.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return vss;
        }
        ggml_backend_meta_split_state ret = vss;
        const bool is_4d = tensor->ne[0] == S_v && tensor->ne[1] == S_v && tensor->ne[2] == H;
        ret.axis = is_4d ? GGML_BACKEND_SPLIT_AXIS_2 : GGML_BACKEND_SPLIT_AXIS_0;
        if (!is_4d) {
            for (size_t s = 0; s < ret.n_segments; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ret.ne[s * n_bufs + j] *= S_v * S_v;
                }
            }
        }
        {
            static int nlog;
            if (nlog < 8) {
                nlog++;
                fprintf(stderr, "[GDN_STATE] name=%s 4d=%d axis=%d nr=%u nseg=%u ne=[%lld,%lld,%lld] offs=%zu",
                        tensor->name, (int) is_4d, (int) ret.axis, ret.nr[0], ret.n_segments,
                        (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2],
                        (size_t) tensor->view_offs);
                for (size_t j = 0; j < n_bufs; j++) {
                    fprintf(stderr, " j%zu=%lld", j, (long long) ret.ne[j]);
                }
                fprintf(stderr, "\n");
            }
        }
        return ret;
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_tensor * gdn = nullptr;
        if (ggml_backend_meta_is_gdn_state_view(tensor, &gdn)) {
            return gdn_state_view_split(gdn);
        }
        // Sequential inner-K: [3072, 2, T] MIRRORED → one 3072-half per device.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                strstr(tensor->name, "final_output_khalf") != nullptr &&
                tensor->ne[1] == (int64_t) n_bufs) {
            ggml_backend_meta_split_state ret;
            memset(&ret, 0, sizeof(ret));
            ret.axis = GGML_BACKEND_SPLIT_AXIS_1;
            ret.n_segments = 1;
            ret.nr[0] = 1;
            for (size_t j = 0; j < n_bufs; j++) {
                ret.ne[j] = 1;
            }
            fprintf(stderr, "[GDN_KHALF] name=%s ne0=%lld\n", tensor->name, (long long) tensor->ne[0]);
            return ret;
        }
        // Sequential GDN x [6144] MIRRORED → [1024, 2, 3, T] = [d0g0,d1g0, d0g1,d1g1, d0g2,d1g2].
        // Split the device axis so each GPU holds groups 0,1,2 of its 1024 (3-rep V-order).
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                strstr(tensor->name, "final_output_3rep") != nullptr &&
                tensor->ne[0] == 1024 && tensor->ne[1] == (int64_t) n_bufs && tensor->ne[2] > 1) {
            ggml_backend_meta_split_state ret;
            memset(&ret, 0, sizeof(ret));
            ret.axis = GGML_BACKEND_SPLIT_AXIS_1;
            ret.n_segments = 1;
            ret.nr[0] = 1;
            for (size_t j = 0; j < n_bufs; j++) {
                ret.ne[j] = 1;
            }
            {
                static int n3;
                if (n3 < 4) {
                    n3++;
                    fprintf(stderr, "[GDN_3REP] name=%s ne=[%lld,%lld,%lld] axis=1 ne_j=%lld\n",
                            tensor->name, (long long) tensor->ne[0], (long long) tensor->ne[1],
                            (long long) tensor->ne[2], (long long) ret.ne[0]);
                }
            }
            return ret;
        }
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                int64_t base_ne_in = 1;
                for (int dim = 0; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                if (src_ss[0].n_segments == 1) {
                    if (src_ss[0].nr[0] == 0) {
                        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
                    }
                    base_ne_in /= src_ss[0].nr[0];
                    if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1 && src_ss[0].nr[0] == 1) {
                        bool complete_rows = true;
                        for (size_t j = 0; j < n_bufs; j++) {
                            const int64_t ne = src_ss[0].ne[j];
                            complete_rows = complete_rows && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        if (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            // Only for a true vector reshape [D,1] — not [D,1,K] GDN snapshot
                            // views, where dim 2 is the rollback-slot axis.
                            return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    base_ne_out *= tensor->ne[dim];
                    if (base_ne_out % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out/base_ne_in)}, 1};
                    }
                    if (base_ne_out > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss);
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_repeat_back = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        // repeat_back sums src into the smaller dst shape. Reducing the split
        // axis (QWEN4EXP HC mean: [n_embd, hc, T] -> [n_embd, 1, T]) leaves a
        // partial sum per device; AllReduce after this node restores the mean.
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            const int ax = (int) src_ss[0].axis;
            if (tensor->ne[ax] < tensor->src[0]->ne[ax]) {
                return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
            }
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        return src_ss[0];
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_tensor * gdn = nullptr;
        if (ggml_backend_meta_is_gdn_state_view(tensor, &gdn)) {
            return gdn_state_view_split(gdn);
        }
        // GDN attn scores: view [S_v, H, T, B] at offset 0 of packed {S_v*H, T+state}.
        {
            const ggml_tensor * gdn_attn = ggml_backend_meta_gdn_of(tensor);
            if (gdn_attn != nullptr && tensor->view_offs == 0 && gdn_attn->src[2] != nullptr &&
                    tensor->ne[0] == gdn_attn->src[2]->ne[0] &&
                    tensor->ne[0] * tensor->ne[1] == gdn_attn->ne[0] &&
                    src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 &&
                    src_ss[0].n_segments >= 1) {
                ggml_backend_meta_split_state ret = src_ss[0];
                ret.axis = GGML_BACKEND_SPLIT_AXIS_1;
                const int64_t S_v = tensor->ne[0];
                bool ok = true;
                for (size_t s = 0; s < ret.n_segments && ok; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        if (S_v <= 0 || ret.ne[s * n_bufs + j] % S_v != 0) {
                            ok = false;
                            break;
                        }
                        ret.ne[s * n_bufs + j] /= S_v;
                    }
                }
                if (ok) {
                    return ret;
                }
            }
        }
        // QKV/conv channel views: [head_dim, n_heads, T] of packed [n_channels, T]
        // whose parent is nseg=1, nr=5 (Q,K,V0,V1,V2) of key_dim. Take the reps
        // covered by view_offs so V keeps nr=3, not a sequential 24/24 first-half.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 &&
                src_ss[0].n_segments == 1 && src_ss[0].nr[0] > 1 &&
                tensor->src[0] != nullptr && tensor->src[0]->nb[0] > 0 &&
                tensor->ne[0] > 0 && tensor->ne[1] > 1 &&
                tensor->nb[2] == tensor->src[0]->nb[1] &&
                tensor->view_offs % tensor->src[0]->nb[0] == 0) {
            const int64_t parent_n0 = tensor->src[0]->ne[0];
            const uint32_t nr_p = src_ss[0].nr[0];
            if (nr_p > 0 && parent_n0 % nr_p == 0) {
                const int64_t seg = parent_n0 / (int64_t) nr_p;
                const int64_t start = (int64_t) (tensor->view_offs / tensor->src[0]->nb[0]);
                const int64_t count = tensor->ne[0] * tensor->ne[1];
                const int64_t head_dim = tensor->ne[0];
                if (seg > 0 && start % seg == 0 && count % seg == 0 && head_dim > 0) {
                    const uint32_t n_reps = (uint32_t) (count / seg);
                    if (n_reps > 0 && n_reps <= nr_p) {
                        ggml_backend_meta_split_state ret = src_ss[0];
                        ret.axis = GGML_BACKEND_SPLIT_AXIS_1;
                        ret.nr[0] = n_reps;
                        bool ok = true;
                        for (size_t j = 0; j < n_bufs; j++) {
                            if (src_ss[0].ne[j] % head_dim != 0) {
                                ok = false;
                                break;
                            }
                            ret.ne[j] = src_ss[0].ne[j] / head_dim;
                        }
                        if (ok) {
                            static int nvlog;
                            if (nvlog < 12) {
                                nvlog++;
                                fprintf(stderr, "[GDN_VIEW] name=%s start=%lld count=%lld n_reps=%u head=%lld j0=%lld j1=%lld\n",
                                        tensor->name, (long long) start, (long long) count, n_reps, (long long) head_dim,
                                        (long long) ret.ne[0], n_bufs > 1 ? (long long) ret.ne[1] : 0);
                            }
                            return ret;
                        }
                    }
                }
            }
        }
        // GDN rollback snapshots view a 2D RS cache [D, n_rows] as [D, 1, K].
        // That view is contiguous at n_seqs=1, but it is not a reshape: dim 2 is
        // the slot axis. handle_reshape would see ne[1]==1 and move a dim-0 split
        // onto that singleton, dropping slots 1..K on rpc-tensor.
        const bool gdn_snapshot_view =
            ggml_n_dims(tensor->src[0]) == 2 && ggml_n_dims(tensor) >= 3 &&
            tensor->ne[0] == tensor->src[0]->ne[0] && tensor->ne[1] == 1 && tensor->ne[2] > 1;
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0]) && !gdn_snapshot_view) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        if (getenv("LLAMA_TP_LOG_HC") && ggml_backend_meta_is_hc_stream_view(tensor)) {
            fprintf(stderr, "[HC_VIEW] src_axis=%d assume_sync=%d ne=[%lld,%lld] src_ne=[%lld,%lld,%lld]\n",
                    axis, (int) assume_sync,
                    (long long) tensor->ne[0], (long long) tensor->ne[1],
                    (long long) tensor->src[0]->ne[0], (long long) tensor->src[0]->ne[1],
                    (long long) tensor->src[0]->ne[2]);
        }
        // QWEN4EXP HC mix: view one stream [n_embd, T] of [n_embd, hc, T]
        // split on hc. Full n_embd on every device (non-owners read zeros);
        // ADD of these views is PARTIAL and AllReduces the mean.
        if (axis == GGML_BACKEND_SPLIT_AXIS_1 && ggml_backend_meta_is_hc_stream_view(tensor)) {
            // Views stay MIRRORED so they are not AllReduce boundaries; the
            // ADD of the streams is PARTIAL (handle_bin_bcast).
            if (getenv("LLAMA_TP_LOG_HC")) {
                fprintf(stderr, "[HC_VIEW] axis1 stream offs=%zu ne=[%lld,%lld] src_ne1=%lld -> MIRRORED\n",
                        (size_t) tensor->view_offs, (long long) tensor->ne[0], (long long) tensor->ne[1],
                        (long long) tensor->src[0]->ne[1]);
            }
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
                return handle_reshape(src_ss);
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        const bool kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        const bool kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_lightning_indexer = [&](
            const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        for (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            // Kernel holds the qkv channel segments (nr=5). Copy them; only the
            // split axis flips (channels are dim 1 of sx and dim 0 of dst).
            ggml_backend_meta_split_state ret = src_ss[1];
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                ret.axis = GGML_BACKEND_SPLIT_AXIS_1;
                return ret;
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
                return ret;
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        // N-split AllGather: full-sized dest on each GPU. HIP writes local heads
        // into the first half; SEQ concat fills the rest. Check before AXIS_1
        // asserts — split qkv activations are often AXIS_0, not AXIS_1.
        static const bool nspl = getenv("LLAMA_TP_SSM_OUT_NSPLIT") != nullptr;
        if (nspl) {
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED
                                : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        // Result is 2D {S_v*H, T*B + state_rows}. Pack from V (src2), not Q:
        // fused GDN keeps Q at n_k_heads and V at n_v_heads (head_ratio=3).
        // Keep V's nr (3 groups of 16 k-heads) so ssm_out's {{2048,3}} matches.
        ggml_backend_meta_split_state ret = src_ss[2];
        ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
        const int64_t S_v = tensor->src[2]->ne[0];
        for (size_t s = 0; s < ret.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ret.ne[s * n_bufs + j] *= S_v;
            }
        }
        return ret;
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            // Empty views (graph reservation, n_seqs=0) must keep the parent's
            // split so view_offs is scaled onto the shard. UNKNOWN left
            // cache_s_l* views with full-tensor offsets past the RPC shard
            // (qwen4exp TP GRAPH_COMPUTE abort).
            if (tensor->view_src != nullptr) {
                return ggml_backend_meta_get_split_state(stc, tensor->view_src, assume_sync);
            }
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            } else if (ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            return ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS: {
                // Reducing hc after permute. Extra PARTIAL AllReduce of this
                // node deadlocks USB4 GRAPH_COMPUTE on 48-layer Flash-Next.
                // Each device already holds whole streams (hc split), so the
                // local sum is the full mean for its n_embd shard when the
                // split is on n_embd (axis 1 of the un-permuted residual).
                if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                    split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
                } else {
                    split_state = handle_per_row(src_ss);
                }
            } break;
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT: {
                if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
                } else {
                    split_state = handle_generic(src_ss, /*scalar_only =*/ false);
                }
            } break;
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_repeat_back(src_ss);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                // Unfused GDN writes per-chunk outputs into a head-split buffer.
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
                } else {
                    split_state = handle_reshape(src_ss);
                }
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG: {
                // Elementwise; unfused GDN chunking builds an identity via FILL+DIAG
                // on a head-split tensor (axis 3 after permute).
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                // Scores may be split on heads while the mask/sinks are mirrored.
                if (src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN &&
                        (tensor->src[1] == nullptr || src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) &&
                        (tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED)) {
                    split_state = src_ss[0];
                } else {
                    split_state = handle_generic(src_ss, /*scalar_only =*/ false);
                }
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                // Elementwise triangular mask; unfused GDN decay_mask is head-split.
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOLVE_TRI: {
                // Batched triangular solve; unfused GDN chunking runs it per head (axis 3).
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } break;
            case GGML_OP_DSV4_HC_COMB:
            case GGML_OP_DSV4_HC_PRE:
            case GGML_OP_DSV4_HC_POST: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
            int64_t ne_already = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ne_already += split_state.ne[s * n_bufs + j] * split_state.nr[s];
                }
            }
            const bool hc_stream_view_filled = (ne_already == tensor->ne[split_state.axis]);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (hc_stream_view_filled) {
                    break;
                }
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            if (div == 0) {
                                split_state.ne[j] = 0;
                                continue;
                            }
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            if (!hc_stream_view_filled) {
                GGML_ASSERT(!first_src_split_by_axis);
            }
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }

    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state split_state =
                        ggml_backend_meta_get_split_state(tensor->src[i], true);
                GGML_ASSERT(split_state.n_segments == 1);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                const ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[key].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t s = 0; s < ret.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    if (tensor == nullptr) {
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    }
    if (tensor->buffer == nullptr || !ggml_backend_buffer_is_meta(tensor->buffer)) {
        if (tensor->view_src != nullptr && tensor->view_src->buffer != nullptr &&
                ggml_backend_buffer_is_meta(tensor->view_src->buffer)) {
            tensor = tensor->view_src;
        } else {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
    }
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    return ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if ((simple_buf != nullptr) && ggml_backend_buffer_is_multi_buffer(simple_buf)) {
            // see https://github.com/ggml-org/llama.cpp/issues/22197
            GGML_ABORT("multi buffers are not supported by the meta backend");
        }

        const bool pack_3rep = tensor->op == GGML_OP_RESHAPE && split_dim == 1 &&
                (strstr(tensor->name, "final_output_3rep") != nullptr ||
                 strstr(tensor->name, "final_output_khalf") != nullptr);
        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    // 3-rep pack: keep the full key_dim group stride so we skip
                    // the other device's 1024 of each 2048 group.
                    if (pack_3rep && i > split_dim) {
                        continue;
                    }
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (pack_3rep && tensor->src[0] != nullptr) {
            ggml_tensor * p = tensor->src[0];
            if (ggml_backend_buffer_is_meta(p->buffer)) {
                p = ggml_backend_meta_buffer_simple_tensor(p, j);
            }
            t_ij->view_src = p;
            // Skip the other device's 1024 in each (device, group) pair.
            t_ij->view_offs = (size_t) j * tensor->nb[1];
            t_ij->nb[1] = tensor->nb[1];
            t_ij->nb[2] = tensor->nb[2];
            t_ij->nb[3] = tensor->nb[3];
            static int n3v;
            if (n3v < 4) {
                n3v++;
                fprintf(stderr, "[GDN_3REP_VIEW] j=%zu ne=[%lld,%lld,%lld] offs=%zu nb1=%zu nb2=%zu pne0=%lld\n",
                        j, (long long) t_ij->ne[0], (long long) t_ij->ne[1], (long long) t_ij->ne[2],
                        (size_t) t_ij->view_offs, t_ij->nb[1], t_ij->nb[2],
                        p ? (long long) p->ne[0] : -1);
            }
        }
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                if (split_dim_view_src < 0 || split_dim_view_src >= GGML_MAX_DIMS) {
                    // PARTIAL/MIRRORED parent is full-sized; keep view_offs.
                } else {

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
                {
                    static int nvoff;
                    if (nvoff < 12 && tensor->ne[1] > 1 &&
                            (tensor->ne[1] == 16 || tensor->ne[1] == 48 || tensor->ne[1] == 24)) {
                        nvoff++;
                        fprintf(stderr, "[GDN_OFF] name=%s j=%zu split=%d internal=%d offs %zu -> %zu ne1=%lld/%lld pne0=%lld\n",
                                tensor->name, j, split_dim, (int) split_internal_offset,
                                (size_t) tensor->view_offs, (size_t) t_ij->view_offs,
                                (long long) t_ij->ne[1], (long long) tensor->ne[1],
                                t_ij->view_src ? (long long) t_ij->view_src->ne[0] : -1);
                    }
                }
                }
            }
        }
        if (t_ij->view_src != nullptr && tensor->op == GGML_OP_VIEW &&
                tensor->view_src != nullptr && tensor->view_src->nb[1] > 0 &&
                tensor->ne[0] == tensor->view_src->ne[0] &&
                tensor->ne[1] == tensor->view_src->ne[2] &&
                tensor->nb[1] == tensor->view_src->nb[2] &&
                tensor->view_offs % tensor->view_src->nb[1] == 0) {
            const ggml_backend_meta_split_state pss =
                ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true);
            if (pss.axis == GGML_BACKEND_SPLIT_AXIS_1) {
                const int64_t c = (int64_t) (tensor->view_offs / tensor->view_src->nb[1]);
                int64_t pos = 0;
                for (size_t jj = 0; jj < j; jj++) {
                    for (size_t s = 0; s < pss.n_segments; s++) {
                        pos += pss.ne[s * n_simple_bufs + jj] * pss.nr[s];
                    }
                }
                int64_t nloc = 0;
                for (size_t s = 0; s < pss.n_segments; s++) {
                    nloc += pss.ne[s * n_simple_bufs + j] * pss.nr[s];
                }
                if (c >= pos && c < pos + nloc && t_ij->view_src->data != nullptr) {
                    const int64_t local_c = c - pos;
                    t_ij->nb[1] = t_ij->view_src->nb[2];
                    t_ij->nb[2] = t_ij->nb[1] * (size_t) t_ij->ne[1];
                    t_ij->nb[3] = t_ij->nb[2];
                    t_ij->view_offs = (size_t) local_c * t_ij->view_src->nb[1];
                } else {
                    // This device does not own stream c: broadcast a zero row.
                    if (buf_ctx->hc_zero_ptrs.size() < n_simple_bufs) {
                        buf_ctx->hc_zero_bufs.resize(n_simple_bufs);
                        buf_ctx->hc_zero_ptrs.resize(n_simple_bufs, nullptr);
                    }
                    if (buf_ctx->hc_zero_ptrs[j] == nullptr && simple_buf != nullptr) {
                        const size_t zsz = 8192 * sizeof(float);
                        ggml_backend_buffer_t zb = ggml_backend_buft_alloc_buffer(
                                ggml_backend_buffer_get_type(simple_buf), zsz);
                        ggml_backend_buffer_clear(zb, 0);
                        buf_ctx->hc_zero_bufs[j].reset(zb);
                        buf_ctx->hc_zero_ptrs[j] = ggml_backend_buffer_get_base(zb);
                    }
                    t_ij->view_src  = nullptr;
                    t_ij->view_offs = 0;
                    t_ij->nb[1]     = 0;
                    t_ij->nb[2]     = 0;
                    t_ij->nb[3]     = 0;
                    t_ij->data      = buf_ctx->hc_zero_ptrs[j];
                    GGML_ASSERT(t_ij->data != nullptr);
                }
            }
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }

        if (simple_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_buffer_init_tensor(simple_buf, t_ij);
        } else {
            t_ij->extra = tensor->extra;
        }
        if (getenv("LLAMA_TP_LOG_HC") && ggml_backend_meta_is_hc_stream_view(tensor)) {
            fprintf(stderr, "[HC_INIT] j=%zu ne0=%lld nb1=%zu offs=%zu data=%p pne=[%lld,%lld,%lld] pnb=[%zu,%zu,%zu]\n",
                    j, (long long) t_ij->ne[0], t_ij->nb[1], (size_t) t_ij->view_offs, t_ij->data,
                    t_ij->view_src ? (long long) t_ij->view_src->ne[0] : -1,
                    t_ij->view_src ? (long long) t_ij->view_src->ne[1] : -1,
                    t_ij->view_src ? (long long) t_ij->view_src->ne[2] : -1,
                    t_ij->view_src ? t_ij->view_src->nb[0] : 0,
                    t_ij->view_src ? t_ij->view_src->nb[1] : 0,
                    t_ij->view_src ? t_ij->view_src->nb[2] : 0);
        }

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = tensor->src[i];
            t_ij->src[i] = src;
            if (src == tensor) {
                t_ij->src[i] = t_ij;
                continue;
            }
            if (src == nullptr) {
                continue;
            }
            // HC mix ADD(view_c, view_k): the VIEW is often not view_init'd
            // yet (no-op, not a sched leaf). simple_tensor then walks to the
            // parent and HIP ADD sees [hc_dim,T] (NMSE ~1e2). Init the VIEW
            // shard first. Do not do this for sigmoid/unary (needs contiguous).
            if (tensor->op == GGML_OP_ADD && ggml_backend_meta_is_hc_stream_view(src)) {
                if (src->buffer == nullptr && src->view_src != nullptr &&
                        src->view_src->data != nullptr &&
                        src->view_src->buffer != nullptr &&
                        ggml_backend_buffer_is_meta(src->view_src->buffer)) {
                    ggml_backend_view_init(src);
                }
                if (src->buffer != nullptr && ggml_backend_buffer_is_meta(src->buffer)) {
                    ggml_backend_meta_simple_tensor_container & stc_src =
                        ((ggml_backend_meta_buffer_context *) src->buffer->context)
                            ->get_simple_tensor_container(src);
                    if (stc_src.simple_tensors.find(src) == stc_src.simple_tensors.end()) {
                        ggml_backend_meta_buffer_init_tensor_impl(stc_src, src);
                    }
                }
            }
            if (src->buffer != nullptr && ggml_backend_buffer_is_meta(src->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(src, j);
            }
        }

        // Inner-K: W is AXIS_0 sequential (nseg=1,nr=1) and x is a full mirrored
        // activation. HIP mmq uses src1->ne[0] as K — a 6144 x with a 3072 W
        // overruns or takes the wrong slice. View x at this device's K offset
        // keeping the parent row stride so token t is x[k0:k0+K, t].
        if ((tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_MUL_MAT_ID) &&
                t_ij->src[0] != nullptr && t_ij->src[1] != nullptr) {
            ggml_tensor * w = t_ij->src[0];
            ggml_tensor * x = t_ij->src[1];
            const ggml_backend_meta_split_state wss =
                ggml_backend_meta_get_split_state(stc, tensor->src[0], /*assume_sync =*/ true);
            if (wss.axis == GGML_BACKEND_SPLIT_AXIS_0 && wss.n_segments == 1 && wss.nr[0] == 1 &&
                    x->type == GGML_TYPE_F32 && w->ne[0] > 0 && x->ne[0] > w->ne[0] &&
                    x->nb[0] == sizeof(float) && x->data != nullptr) {
                int64_t k0 = 0;
                for (size_t jj = 0; jj < j; jj++) {
                    ggml_tensor * wj = ggml_backend_meta_buffer_simple_tensor(tensor->src[0], jj);
                    if (wj != nullptr) {
                        k0 += wj->ne[0];
                    }
                }
                GGML_ASSERT(k0 + w->ne[0] <= x->ne[0]);
                int64_t xne[GGML_MAX_DIMS];
                for (int d = 0; d < GGML_MAX_DIMS; d++) {
                    xne[d] = x->ne[d];
                }
                xne[0] = w->ne[0];
                ggml_tensor * xv = ggml_new_tensor(simple_ctx, x->type, GGML_MAX_DIMS, xne);
                xv->op = GGML_OP_VIEW;
                for (int d = 0; d < GGML_MAX_DIMS; d++) {
                    xv->nb[d] = x->nb[d];
                }
                ggml_set_name(xv, "innerk_x");
                xv->buffer   = x->buffer;
                xv->extra    = x->extra;
                xv->view_src = x->view_src != nullptr ? x->view_src : x;
                xv->view_offs = (x->view_src != nullptr ? x->view_offs : 0) + (size_t) k0 * x->nb[0];
                xv->data = (char *) xv->view_src->data + xv->view_offs;
                t_ij->src[1] = xv;
                static int ninnerk;
                if (ninnerk < 8) {
                    ninnerk++;
                    fprintf(stderr, "[INNERK] %s j=%zu k0=%lld wK=%lld xK=%lld xnb1=%zu\n",
                            tensor->name, j, (long long) k0, (long long) w->ne[0],
                            (long long) x->ne[0], x->nb[1]);
                }
            }
        }

        // N-split: W AXIS_1 (full K, half N), dest is full n_embd (PARTIAL).
        // HIP MUL_MAT into a native 1280 dest (own buffer). Host allgather fills t_ij.
        if ((tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_MUL_MAT_ID) &&
                t_ij->src[0] != nullptr && t_ij->src[1] != nullptr && simple_buf != nullptr &&
                getenv("LLAMA_TP_SSM_OUT_NSPLIT") != nullptr &&
                (strstr(tensor->name, "linear_attn_out") != nullptr ||
                 (tensor->src[0]->name[0] && strstr(tensor->src[0]->name, "ssm_out") != nullptr))) {
            ggml_tensor * w = t_ij->src[0];
            ggml_tensor * x = t_ij->src[1];
            {
                auto itx = g_gdn_x_full.find(x);
                if (itx != g_gdn_x_full.end() && itx->second != nullptr) {
                    x = itx->second;
                    t_ij->src[1] = x;
                }
            }
            const ggml_backend_meta_split_state wss =
                ggml_backend_meta_get_split_state(stc, tensor->src[0], /*assume_sync =*/ true);
            if (wss.axis == GGML_BACKEND_SPLIT_AXIS_1 && wss.n_segments == 1 && wss.nr[0] == 1 &&
                    t_ij->type == GGML_TYPE_F32 && w->ne[1] > 0 && t_ij->ne[0] > w->ne[1] &&
                    x->type == GGML_TYPE_F32 && x->ne[0] == w->ne[0]) {
                if (j == 0 && strstr(tensor->name, "linear_attn_out-0") != nullptr) {
                    g_innern_gemm.clear();
                    g_innern_bufs.clear();
                }
                int64_t n0 = 0;
                for (size_t jj = 0; jj < j; jj++) {
                    ggml_tensor * wj = ggml_backend_meta_buffer_simple_tensor(tensor->src[0], jj);
                    if (wj != nullptr) {
                        n0 += wj->ne[1];
                    }
                }
                GGML_ASSERT(n0 + w->ne[1] <= t_ij->ne[0]);
                int64_t gne[GGML_MAX_DIMS];
                for (int d = 0; d < GGML_MAX_DIMS; d++) {
                    gne[d] = t_ij->ne[d];
                }
                gne[0] = w->ne[1];
                ggml_tensor * gemm = ggml_new_tensor(simple_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, gne);
                gemm->op = GGML_OP_MUL_MAT;
                memcpy(gemm->op_params, tensor->op_params, sizeof(tensor->op_params));
                gemm->nb[0] = sizeof(float);
                gemm->nb[1] = gemm->nb[0] * (size_t) gemm->ne[0];
                gemm->nb[2] = gemm->nb[1] * (size_t) gemm->ne[1];
                gemm->nb[3] = gemm->nb[2] * (size_t) gemm->ne[2];
                gemm->flags = GGML_TENSOR_FLAG_COMPUTE;
                gemm->src[0] = w;
                gemm->src[1] = x;
                ggml_set_name(gemm, "innern_gemm");
                const size_t gbytes = ggml_nbytes(gemm);
                ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(simple_buf);
                g_innern_bufs.emplace_back(ggml_backend_buft_alloc_buffer(buft, gbytes));
                ggml_backend_buffer_t gb = g_innern_bufs.back().get();
                GGML_ASSERT(gb != nullptr);
                gemm->buffer = gb;
                gemm->data = ggml_backend_buffer_get_base(gb);
                ggml_backend_buffer_init_tensor(gb, gemm);
                t_ij->op = GGML_OP_NONE;
                t_ij->src[0] = nullptr;
                t_ij->src[1] = nullptr;
                g_innern_gemm[t_ij] = gemm;
                static int ninnern;
                if (ninnern < 8) {
                    ninnern++;
                    fprintf(stderr, "[INNERN] %s j=%zu n0=%lld wN=%lld dstN=%lld wK=%lld xK=%lld gbytes=%zu\n",
                            tensor->name, j, (long long) n0, (long long) w->ne[1],
                            (long long) t_ij->ne[0], (long long) w->ne[0],
                            (long long) x->ne[0], gbytes);
                }
            }
        }

        if (simple_buf != nullptr && t_ij != nullptr && t_ij->type == GGML_TYPE_F32 &&
                getenv("LLAMA_TP_SSM_OUT_NSPLIT") != nullptr &&
                tensor->name[0] && strstr(tensor->name, "final_output") != nullptr &&
                strstr(tensor->name, "3rep") == nullptr && strstr(tensor->name, "khalf") == nullptr &&
                tensor->ne[0] == 6144 && t_ij->ne[0] == 3072) {
            if (j == 0 && strstr(tensor->name, "final_output-0") != nullptr) {
                g_gdn_x_full.clear();
                g_gdn_x_bufs.clear();
            }
            int64_t dne[GGML_MAX_DIMS];
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                dne[d] = tensor->ne[d];
            }
            ggml_tensor * dest = ggml_new_tensor(simple_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, dne);
            dest->op = GGML_OP_NONE;
            dest->nb[0] = sizeof(float);
            dest->nb[1] = dest->nb[0] * (size_t) dest->ne[0];
            dest->nb[2] = dest->nb[1] * (size_t) dest->ne[1];
            dest->nb[3] = dest->nb[2] * (size_t) dest->ne[2];
            ggml_set_name(dest, "gdn_x_full");
            const size_t dbytes = ggml_nbytes(dest);
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(simple_buf);
            g_gdn_x_bufs.emplace_back(ggml_backend_buft_alloc_buffer(buft, dbytes));
            ggml_backend_buffer_t db = g_gdn_x_bufs.back().get();
            GGML_ASSERT(db != nullptr);
            dest->buffer = db;
            dest->data = ggml_backend_buffer_get_base(db);
            ggml_backend_buffer_init_tensor(db, dest);
            g_gdn_x_full[t_ij] = dest;
            static int ngdnx;
            if (ngdnx < 8) {
                ngdnx++;
                fprintf(stderr, "[GDN_X] %s j=%zu shardN=%lld destN=%lld dbytes=%zu\n",
                        tensor->name, j, (long long) t_ij->ne[0], (long long) dest->ne[0], dbytes);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0 && tensor->op != GGML_OP_ADD && tensor->op != GGML_OP_MUL &&
                    tensor->op != GGML_OP_SCALE) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_meta_xfer_split_chunks(
        const ggml_tensor * tensor,
        const ggml_backend_meta_split_state & split_state,
        size_t n_bufs,
        void * data, size_t offset, size_t size,
        bool is_set);

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    return ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

static void ggml_backend_meta_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        for (int64_t row = 0; row < row_count; row++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (row_start + row)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            return;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    for (int64_t row = 0; row < row_count; row++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (row_start + row)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            if (offset % chunk_size_full != 0 || size % chunk_size_full != 0) {
                std::vector<uint8_t> tmp(size);
                ggml_backend_meta_xfer_split_chunks(tensor, split_state, n_bufs, tmp.data(), offset, size, /*is_set=*/false);
                memset(tmp.data(), value, size);
                ggml_backend_meta_xfer_split_chunks(tensor, split_state, n_bufs, tmp.data(), offset, size, /*is_set=*/true);
                break;
            }
            const int64_t i_start =  offset        / chunk_size_full;
            const int64_t i_stop  = (offset + size) / chunk_size_full;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size == 0) {
                    continue;
                }
                for (int64_t i = i_start; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_size, chunk_size);
                }
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

// Copy a contiguous byte range of a tensor-parallel (split) tensor by splicing
// per-device chunks. The scheduler often requests a size that is not a multiple
// of one full split-slice (nb[axis+1]) — e.g. MoE expert rows, views, logits.
// The aligned path below assumes that and used to GGML_ASSERT.
static void ggml_backend_meta_xfer_split_chunks(
        const ggml_tensor * tensor,
        const ggml_backend_meta_split_state & split_state,
        size_t n_bufs,
        void * data, size_t offset, size_t size,
        bool is_set) {
    GGML_ASSERT(split_state.axis >= GGML_BACKEND_SPLIT_AXIS_0);
    GGML_ASSERT(split_state.axis <= GGML_BACKEND_SPLIT_AXIS_2);
    if (size == 0) {
        return;
    }
    const int axis = (int) split_state.axis;
    GGML_ASSERT(axis + 1 < GGML_MAX_DIMS);
    const size_t chunk_full = tensor->nb[axis + 1];
    GGML_ASSERT(chunk_full > 0);

    auto xfer_aligned = [&](size_t off, size_t sz, void * ptr, bool do_set) {
        GGML_ASSERT(off % chunk_full == 0);
        GGML_ASSERT(sz  % chunk_full == 0);
        const int64_t i_start  = (int64_t) (off / chunk_full);
        const int64_t n_copies = (int64_t) (sz  / chunk_full);
        if (n_copies <= 0) {
            return;
        }
        size_t offset_j = 0;
        for (size_t j = 0; j < n_bufs; j++) {
            ggml_tensor * st = ggml_backend_meta_buffer_simple_tensor(tensor, j);
            const size_t chunk_j = st->nb[axis + 1];
            if (chunk_j == 0) {
                continue;
            }
            const size_t simple_offset = (size_t) i_start * chunk_j;
            if (do_set) {
                ggml_backend_tensor_set_2d(st, (const char *) ptr + offset_j, simple_offset, chunk_j,
                    (size_t) n_copies, chunk_j, chunk_full);
            } else {
                ggml_backend_tensor_get_2d(st, (char *) ptr + offset_j, simple_offset, chunk_j,
                    (size_t) n_copies, chunk_j, chunk_full);
            }
            offset_j += chunk_j;
        }
        GGML_ASSERT(offset_j == chunk_full);
    };

    if (offset % chunk_full == 0 && size % chunk_full == 0) {
        xfer_aligned(offset, size, data, is_set);
        return;
    }

    const int64_t i0 = (int64_t) (offset / chunk_full);
    const int64_t i1 = (int64_t) ((offset + size + chunk_full - 1) / chunk_full);
    std::vector<uint8_t> slice(chunk_full);
    for (int64_t i = i0; i < i1; ++i) {
        const size_t slice_begin = (size_t) i * chunk_full;
        const size_t c0 = std::max(offset, slice_begin);
        const size_t c1 = std::min(offset + size, slice_begin + chunk_full);
        if (c0 >= c1) {
            continue;
        }
        const bool partial = (c0 != slice_begin) || (c1 != slice_begin + chunk_full);
        if (is_set) {
            if (partial) {
                xfer_aligned(slice_begin, chunk_full, slice.data(), /*do_set=*/false);
            }
            memcpy(slice.data() + (c0 - slice_begin), (const char *) data + (c0 - offset), c1 - c0);
            xfer_aligned(slice_begin, chunk_full, slice.data(), /*do_set=*/true);
        } else {
            xfer_aligned(slice_begin, chunk_full, slice.data(), /*do_set=*/false);
            memcpy((char *) data + (c0 - offset), slice.data() + (c0 - slice_begin), c1 - c0);
        }
    }
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    if (!ggml_is_contiguous(tensor) && split_state.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        // Split views (Qwen3.5 cache_r / GDN) are not packed; ship each shard as a compact block.
        if (offset == 0 && size == ggml_nbytes(tensor)) {
            size_t off = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * st = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                if (st == nullptr) {
                    continue;
                }
                const size_t n = ggml_nbytes(st);
                if (n == 0) {
                    continue;
                }
                ggml_backend_tensor_set(st, (const char *) data + off, 0, n);
                off += n;
            }
            GGML_ASSERT(off == size);
            return;
        }
    }
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        {
            static int nset;
            if (nset < 16) {
                nset++;
                fprintf(stderr, "[SET_NR] name=%s axis=%d nseg=%u nr0=%u ne=[%lld,%lld,%lld] type=%s size=%zu\n",
                        tensor->name, (int) split_state.axis, split_state.n_segments, split_state.nr[0],
                        (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2],
                        ggml_type_name(tensor->type), size);
            }
        }

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            // Host-pack each device's nr-rep K slices, then one contiguous SET.
            // RPC has no set_tensor_2d; 2560 strided Q4_K superblock SETs for
            // ssm_out {{2048,3}} were a TP quality suspect.
            std::vector<size_t> col_bytes(n_bufs, 0);
            for (size_t j = 0; j < n_bufs; j++) {
                for (size_t s = 0; s < split_state.n_segments; s++) {
                    GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                    col_bytes[j] += (size_t) (split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0]) * split_state.nr[s];
                }
            }
            std::vector<std::vector<char>> packed(n_bufs);
            for (size_t j = 0; j < n_bufs; j++) {
                packed[j].assign((size_t) row_count * col_bytes[j], 0);
            }
            std::vector<size_t> woff(n_bufs, 0);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        for (int64_t c = 0; c < row_count; c++) {
                            memcpy(packed[j].data() + (size_t) c * col_bytes[j] + woff[j],
                                   (const char *) data + offset_data + (size_t) c * tensor->nb[1],
                                   nbytes);
                        }
                        woff[j] += nbytes;
                        offset_data += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*(size_t) row_count == size);
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * st = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                if (st == nullptr || packed[j].empty()) {
                    continue;
                }
                ggml_backend_tensor_set(st, packed[j].data(),
                        (size_t) row_start * st->nb[1], packed[j].size());
            }
            // Confirm 3-rep (not first-half). dest0 group1 = K[2048:3072]; dest1 group1 = K[3072:4096].
            if (split_state.nr[0] == 3 && n_bufs >= 1 && row_count >= 1 &&
                    strstr(tensor->name, "ssm_out") != nullptr) {
                static int nchk_ssm;
                if (nchk_ssm < 2) {
                    nchk_ssm++;
                    const size_t nbytes = (size_t) (split_state.ne[0] / blck_size * tensor->nb[0]);
                    for (size_t j = 0; j < n_bufs && j < 2; j++) {
                        ggml_tensor * stj = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        if (stj == nullptr || packed[j].size() < 2 * nbytes) {
                            fprintf(stderr, "[SET_NR_CHK] %s dest%zu skipped\n", tensor->name, j);
                            continue;
                        }
                        std::vector<char> got(nbytes);
                        const size_t got_off = (size_t) row_start * stj->nb[1] + nbytes;
                        ggml_backend_tensor_get(stj, got.data(), got_off, nbytes);
                        const char * k_3rep = (const char *) data + (2 + j) * nbytes;
                        const char * k_half = (const char *) data + nbytes;
                        const int eq_3 = memcmp(got.data(), k_3rep, nbytes) == 0;
                        const int eq_h = memcmp(got.data(), k_half, nbytes) == 0;
                        fprintf(stderr, "[SET_NR_CHK] %s dest%zu[1]==3rep %d dest%zu[1]==half %d chunk=%zu packed_eq3=%d\n",
                                tensor->name, j, eq_3, j, eq_h, nbytes,
                                memcmp(packed[j].data() + nbytes, k_3rep, nbytes) == 0);
                    }
                }
            }
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        // Host-pack nr-rep N slices, one contiguous SET per device.
        // RPC set_tensor_2d is NULL; Q6_K qkv {{2048,5}} was 5 offset SETs.
        std::vector<size_t> n_bytes(n_bufs, 0);
        for (size_t j = 0; j < n_bufs; j++) {
            for (size_t s = 0; s < split_state.n_segments; s++) {
                n_bytes[j] += (size_t) split_state.ne[s*n_bufs + j] * tensor->nb[1] * split_state.nr[s];
            }
        }
        std::vector<std::vector<char>> packed1(n_bufs);
        for (size_t j = 0; j < n_bufs; j++) {
            packed1[j].assign((size_t) row_count * n_bytes[j], 0);
        }
        std::vector<size_t> woff1(n_bufs, 0);
        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const size_t nbytes = (size_t) split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    for (int64_t c = 0; c < row_count; c++) {
                        memcpy(packed1[j].data() + (size_t) c * n_bytes[j] + woff1[j],
                               (const char *) data + offset_data + (size_t) c * tensor->nb[2],
                               nbytes);
                    }
                    woff1[j] += nbytes;
                    offset_data += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*(size_t) row_count == size);
        for (size_t j = 0; j < n_bufs; j++) {
            ggml_tensor * st = ggml_backend_meta_buffer_simple_tensor(tensor, j);
            if (st == nullptr || packed1[j].empty()) {
                continue;
            }
            ggml_backend_tensor_set(st, packed1[j].data(),
                    (size_t) row_start * st->nb[2], packed1[j].size());
        }
        // Confirm 5-rep (not first-half): dest j group1 must be K[0:1024], not Q[1024:2048].
        if (split_state.nr[0] == 5 && n_bufs >= 1 && row_count == 1 &&
                strstr(tensor->name, "attn_qkv.weight") != nullptr) {
            static int nchk;
            if (nchk < 2) {
                nchk++;
                const size_t chunk = (size_t) split_state.ne[0] * tensor->nb[1];
                // Source order is (r,j): Q0,Q1,K0,K1,... dest0[1]=K0 at 2*chunk, dest1[1]=K1 at 3*chunk.
                for (size_t j = 0; j < n_bufs && j < 2; j++) {
                    ggml_tensor * stj = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    if (stj == nullptr || packed1[j].size() < 2 * chunk) {
                        continue;
                    }
                    std::vector<char> got(chunk);
                    ggml_backend_tensor_get(stj, got.data(), chunk, chunk);
                    const char * k = (const char *) data + (2 + j) * chunk;
                    const char * q1 = (const char *) data + chunk;
                    const int eq_k = memcmp(got.data(), k, chunk) == 0;
                    const int eq_q = memcmp(got.data(), q1, chunk) == 0;
                    const int pk = memcmp(packed1[j].data() + chunk, k, chunk) == 0;
                    fprintf(stderr, "[SET_NR_CHK] %s dest%zu[1]==K%zu %d dest%zu[1]==Q1 %d packed_eqK %d chunk=%zu\n",
                            tensor->name, j, j, eq_k, j, eq_q, pk, chunk);
                }
            }
        }
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            ggml_backend_meta_xfer_split_chunks(tensor, split_state, n_bufs, const_cast<void *>(data), offset, size, /*is_set=*/true);
            if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_1 && n_bufs >= 2 &&
                    strstr(tensor->name, "ssm_out") != nullptr && offset == 0 &&
                    tensor->nb[1] > 0) {
                static int nchk_a1;
                if (nchk_a1 < 2) {
                    nchk_a1++;
                    const size_t col = tensor->nb[1];
                    ggml_tensor * d0 = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
                    ggml_tensor * d1 = ggml_backend_meta_buffer_simple_tensor(tensor, 1);
                    if (d0 && d1 && col <= ggml_nbytes(d0) && col <= ggml_nbytes(d1) &&
                            size >= col * (size_t) tensor->ne[1]) {
                        std::vector<char> g0(col), g1(col);
                        ggml_backend_tensor_get(d0, g0.data(), 0, col);
                        ggml_backend_tensor_get(d1, g1.data(), 0, col);
                        const char * src = (const char *) data;
                        const int eq0 = memcmp(g0.data(), src, col) == 0;
                        const int eq1_n = memcmp(g1.data(), src + (size_t) (tensor->ne[1] / 2) * col, col) == 0;
                        const int eq1_c1 = memcmp(g1.data(), src + col, col) == 0;
                        fprintf(stderr, "[SET_AXIS1_CHK] %s dest0==src0 %d dest1==srcN/2 %d dest1==src1 %d col=%zu N=%lld\n",
                                tensor->name, eq0, eq1_n, eq1_c1, col, (long long) tensor->ne[1]);
                    }
                }
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % sizeof(float) == 0);
            GGML_ASSERT(size   % sizeof(float) == 0);
            const size_t n_values = size / sizeof(float);
            size_t n_contributors = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                n_contributors += split_state.ne[j] != 0;
            }
            const bool has_contributor_mask = n_contributors != 0;
            if (!has_contributor_mask) {
                n_contributors = n_bufs;
            }
            std::vector<float> tmp(n_values);
            for (size_t i = 0; i < n_values; i++) {
                tmp[i] = ((const float *) data)[i] / n_contributors;
            }
            std::vector<float> zero;
            if (has_contributor_mask) {
                zero.resize(n_values, 0.0f);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const float * partial = has_contributor_mask && split_state.ne[j] == 0 ? zero.data() : tmp.data();
                ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    if (!ggml_is_contiguous(tensor) && split_state.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        if (offset == 0 && size == ggml_nbytes(tensor)) {
            size_t off = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * st = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                if (st == nullptr) {
                    continue;
                }
                const size_t n = ggml_nbytes(st);
                if (n == 0) {
                    continue;
                }
                ggml_backend_tensor_get(st, (char *) data + off, 0, n);
                off += n;
            }
            GGML_ASSERT(off == size);
            return;
        }
    }
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            ggml_backend_meta_xfer_split_chunks(tensor, split_state, n_bufs, data, offset, size, /*is_set=*/false);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
        case GGML_BACKEND_SPLIT_AXIS_NONE: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % sizeof(float) == 0);
            GGML_ASSERT(size   % sizeof(float) == 0);
            const size_t n_values = size / sizeof(float);
            std::vector<float> sum(n_values, 0.0f);
            std::vector<float> tmp(n_values);
            for (size_t j = 0; j < n_bufs; j++) {
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_get(simple_tensor, tmp.data(), offset, size);
                for (size_t i = 0; i < n_values; i++) {
                    sum[i] += tmp[i];
                }
            }
            memcpy(data, sum.data(), size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

void ggml_backend_meta_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        if (buf_ctx->bufs[i]) {
            ggml_backend_buffer_set_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    const ggml_init_params params_static = {
        /*.mem_size   =*/ ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom*ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_nonzero_slice = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                break;
            }
        }
        if (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } else {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;
    uint64_t                    graph_sig     = 0;

    using comm_graph_seq_t = bool (*)(
            void * comm_ctx,
            ggml_backend_t local_backend,
            ggml_cgraph ** local_graphs,
            ggml_tensor ** local_ar,
            uint64_t * rpc_uids,
            ggml_tensor ** rpc_ar,
            size_t n);
    using comm_set_pending_ar_t = void (*)(void * comm_ctx, ggml_tensor * rpc_ar);
    using comm_finish_ar_t      = bool (*)(void * comm_ctx, ggml_tensor * local_ar);
    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;
    comm_graph_seq_t                     comm_graph_seq = nullptr;
    comm_set_pending_ar_t                comm_set_pending_ar = nullptr;
    comm_finish_ar_t                     comm_finish_ar      = nullptr;
    bool                                 seq_plan_valid = false;
    size_t                               seq_j_local    = 0;
    size_t                               seq_j_rpc      = 0;
    std::vector<ggml_cgraph *>           seq_local_gs;
    std::vector<ggml_tensor *>           seq_local_ar;
    std::vector<uint64_t>                seq_rpc_uids;
    std::vector<ggml_tensor *>           seq_rpc_ar;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        if (n_devs > 1) {
            ggml_backend_reg_t comm_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0]));
            for (auto & b : simple_backends) {
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(b));
                if (reg && strncmp(ggml_backend_reg_name(reg), "RPC", 3) == 0) {
                    comm_reg = reg; // If any backend is RPC, use the RPC registry for comms
                    break;
                }
            }
            if (comm_reg != nullptr) {
                ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(comm_reg, "ggml_backend_comm_init");
                if (comm_init != nullptr) {
                    comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
                }
            }
        }
        if (comm_ctx != nullptr) {
            ggml_backend_reg_t comm_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0]));
            for (auto & b : simple_backends) {
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(b));
                if (reg && strncmp(ggml_backend_reg_name(reg), "RPC", 3) == 0) {
                    comm_reg = reg;
                    break;
                }
            }
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(comm_reg, "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
            comm_graph_seq = (comm_graph_seq_t)
                ggml_backend_reg_get_proc_address(comm_reg, "ggml_backend_comm_graph_seq");
            comm_set_pending_ar = (comm_set_pending_ar_t)
                ggml_backend_reg_get_proc_address(comm_reg, "ggml_backend_comm_set_pending_ar");
            comm_finish_ar = (comm_finish_ar_t)
                ggml_backend_reg_get_proc_address(comm_reg, "ggml_backend_comm_finish_ar");
            GGML_LOG_INFO("%s: initialized comm_ctx=%p comm_allreduce=%p for meta backend (n_devs=%zu)\n",
                          __func__, comm_ctx, (void*)comm_allreduce, n_devs);
            fprintf(stderr, "[META_COMM] initialized comm_ctx=%p comm_allreduce=%p n_devs=%zu\n",
                    comm_ctx, (void*)comm_allreduce, n_devs);
            fflush(stderr);
        } else {
            GGML_LOG_INFO("%s: comm_ctx is NULL for meta backend (n_devs=%zu), using allreduce_fallback\n",
                          __func__, n_devs);
            fprintf(stderr, "[META_COMM] comm_ctx is NULL for meta backend (n_devs=%zu), using allreduce_fallback\n",
                    n_devs);
            fflush(stderr);
        }
    }

    ~ggml_backend_meta_context() {
        if (comm_ctx != nullptr) {
            ggml_backend_reg_t comm_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend));
            for (auto & bc : backend_configs) {
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(bc.backend));
                if (reg && strncmp(ggml_backend_reg_name(reg), "RPC", 3) == 0) {
                    comm_reg = reg;
                    break;
                }
            }
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(comm_reg, "ggml_backend_comm_free");
            if (comm_free != nullptr) {
                comm_free(comm_ctx);
            }
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            if (offset % chunk_size_full != 0 || size % chunk_size_full != 0) {
                ggml_backend_synchronize(backend);
                ggml_backend_meta_xfer_split_chunks(tensor, split_state, n_backends, const_cast<void *>(data), offset, size, /*is_set=*/true);
                break;
            }
            const int64_t i_start = offset / chunk_size_full;
            const int64_t i_stop  = (offset + size) / chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = (size_t) i_start * chunk_size_j;
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            if (offset % chunk_size_full != 0 || size % chunk_size_full != 0) {
                ggml_backend_synchronize(backend);
                ggml_backend_meta_xfer_split_chunks(tensor, split_state, n_backends, data, offset, size, /*is_set=*/false);
                break;
            }
            const int64_t i_start = offset / chunk_size_full;
            const int64_t i_stop  = (offset + size) / chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = (size_t) i_start * chunk_size_j;
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // llama.cpp assigns a new cgraph->uid on every sched pass, which forced a
    // full USB4 GRAPH_COMPUTE serialize (~1s/token at 96 subgraphs). Reuse the
    // per-device subgraphs (and their RPC uids → GRAPH_RECOMPUTE) when the
    // op/shape signature matches.
    uint64_t graph_sig = (uint64_t) cgraph->n_nodes * 0x9e3779b97f4a7c15ull;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        graph_sig ^= (uint64_t) n->op + 0x9e3779b97f4a7c15ull + (graph_sig << 6) + (graph_sig >> 2);
        graph_sig ^= (uint64_t) n->ne[0] + ((uint64_t) n->ne[1] << 20) + ((uint64_t) n->ne[2] << 40);
    }
    const bool needs_rebuild = (backend_ctx->n_subgraphs == 0) || (graph_sig != backend_ctx->graph_sig);
    backend_ctx->graph_sig = graph_sig;
    if (needs_rebuild) {
        backend_ctx->seq_plan_valid = false;
    }
    if (!needs_rebuild) {
        static int nreuse;
        if (nreuse < 8) {
            nreuse++;
            fprintf(stderr, "[META_GC] reuse n_nodes=%d n_subgraphs=%zu\n",
                    cgraph->n_nodes, backend_ctx->n_subgraphs);
        }
    }

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        for (int i = 0; i < cgraph->n_leafs; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        for (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
            for (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->buffer != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                    // For regular usage this doesn't matter since it's a noop but trying to call ggml_backend_meta_buffer_simple_tensor results in a crash.
                    bcj.nodes[i] = node;
                    continue;
                }
                // Leave nullptr when the node has no per-device shard. Aliasing
                // the original meta tensor here mutates the live graph (COMPUTE
                // flags) and later AllReduce of nodes[n_nodes-1] memset's garbage
                // (qwen4exp Flash Next SIGSEGV, dest=0x...0032 size ~29GB).
                ggml_tensor * shard = ggml_backend_meta_buffer_simple_tensor(node, j);
                if (shard != nullptr && node->op == GGML_OP_ADD) {
                    for (int s = 0; s < 2; s++) {
                        if (!ggml_backend_meta_is_hc_stream_view(node->src[s])) {
                            continue;
                        }
                        ggml_tensor * v = ggml_backend_meta_buffer_simple_tensor(node->src[s], j);
                        if (v != nullptr && v->op == GGML_OP_VIEW &&
                                ggml_backend_meta_data_ptr_ok(v->data)) {
                            shard->src[s] = v;
                        }
                    }
                }
                if (shard != nullptr && !ggml_backend_meta_data_ptr_ok(shard->data)) {
                    static int nbad;
                    if (nbad < 8 && (node->op == GGML_OP_ADD || node->op == GGML_OP_VIEW || node->op == GGML_OP_SCALE)) {
                        nbad++;
                        fprintf(stderr, "[META_BADPTR] j=%zu op=%s name=%s data=%p\n",
                                j, ggml_op_name(node->op), node->name, shard->data);
                    }
                    shard = nullptr;
                }
                bcj.nodes[i] = shard;
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed_branch = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }
                // HC mix: delay AllReduce through ADD of per-device stream views
                // and the following 1/hc SCALE so one AR sums all hc streams.
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (ggml_node_get_use_count(cgraph, id+1) != 1) {
                        break;
                    }
                    if (next->op == GGML_OP_ADD && next->src[0] == node &&
                            ggml_backend_meta_is_hc_stream_view(next->src[1])) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                        continue;
                    }
                    if (next->op == GGML_OP_SCALE && next->src[0] == node) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                        continue;
                    }
                    break;
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            auto get_i_delayed = [&](const int i) -> int {
                const int i_delayed = get_i_delayed_branch(i);
                ggml_tensor * node = cgraph->nodes[i_delayed];

                if (ggml_node_get_use_count(cgraph, i_delayed) != 1) {
                    return i_delayed;
                }

                for (int id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * next = cgraph->nodes[id];
                    if (next->view_src == node) {
                        return i_delayed;
                    }
                    for (int s = 0; s < GGML_MAX_SRC; s++) {
                        if (next->src[s] == node) {
                            return i_delayed;
                        }
                    }

                    if (next->view_src != nullptr && next->view_src->buffer != nullptr && next->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(next->view_src->buffer)) {
                        continue;
                    }
                    if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        continue;
                    }

                    const int i_other = id;
                    const int i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    if (ggml_node_get_use_count(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        return i_delayed;
                    }

                    ggml_tensor * sum = cgraph->nodes[i_other_delayed + 1];
                    if (sum->op != GGML_OP_ADD ||
                            !ggml_are_same_shape(node, other) || node->type != other->type || sum->type != node->type ||
                            !((sum->src[0] == node && sum->src[1] == other) ||
                              (sum->src[0] == other && sum->src[1] == node)) ||
                            ggml_backend_meta_get_split_state(sum, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        return i_delayed;
                    }

                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if (bcj.nodes[i] == nullptr || bcj.nodes[i_other] == nullptr) {
                            return i_delayed;
                        }
                        const bool compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        const bool compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        if (compute != compute_other) {
                            return i_delayed;
                        }
                    }
                    return i_other_delayed + 1;
                }
                return i_delayed;
            };

            int i_start = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->buffer != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    const size_t nb = ggml_nbytes(node);
                    // Poisoned PARTIAL nodes have reported ~29GB (Flash Next SIGSEGV).
                    // Real TP activations here are << 256 MiB (n_embd=2560, ubatch<=2048).
                    if (nb > 0 && nb <= (256ull << 20)) {
                        max_tmp_size = std::max(max_tmp_size, nb);
                    }
                }
                static const bool nspl_split = getenv("LLAMA_TP_SSM_OUT_NSPLIT") != nullptr;
                const bool gdn_x_ag = nspl_split && node->name[0] &&
                        strstr(node->name, "final_output") != nullptr &&
                        strstr(node->name, "3rep") == nullptr &&
                        strstr(node->name, "khalf") == nullptr;
                const bool new_subgraph = i + 1 == cgraph->n_nodes ||
                        split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL || gdn_x_ag;
                if (!new_subgraph) {
                    continue;
                }

                // LLAMA_TP_NO_DELAY_AR=1: AllReduce at every PARTIAL (no MoE delay).
                // QWEN4EXP HC combine is residual + block; delaying AR through that ADD
                // would AllReduce(mirrored_res + partial) = 2*res + full_block.
                // Do not delay GDN-x concat through ssm_out (that GEMMs 3072 K).
                const bool no_delay_ar = getenv("LLAMA_TP_NO_DELAY_AR") != nullptr;
                const int i_delayed = (no_delay_ar || gdn_x_ag) ? i : get_i_delayed(i);
                static const bool log_ar = getenv("GGML_META_LOG_AR") != nullptr;
                if (log_ar && split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    fprintf(stderr, "[META_AR] i=%d delay=%d name=%s op=%s nbytes=%zu axis=%s",
                            i, i_delayed, node->name, ggml_op_name(node->op),
                            ggml_nbytes(node), ggml_backend_meta_split_axis_name(split_state.axis));
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        ggml_tensor * sh = (i < (int) bcj.nodes.size()) ? bcj.nodes[i] : nullptr;
                        if (sh == nullptr) {
                            fprintf(stderr, " j%zu=null", j);
                        } else {
                            fprintf(stderr, " j%zu=ne=[%lld,%lld],cmp=%d", j,
                                    (long long) sh->ne[0], (long long) sh->ne[1],
                                    (int) !!(sh->flags & GGML_TENSOR_FLAG_COMPUTE));
                        }
                    }
                    fprintf(stderr, "\n");
                }

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if (bcj.nodes[i] == nullptr) {
                            continue;
                        }
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                if (bcj.nodes[ii] != nullptr) {
                                    bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                                }
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;
        if (cgraph->n_nodes <= 0 || cgraph->n_nodes > (1 << 22)) {
            fprintf(stderr, "[META_GRAPH] refusing n_nodes=%d n_subgraphs=%zu\n",
                    cgraph->n_nodes, n_subgraphs);
            return GGML_STATUS_FAILED;
        }

        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        bool missing_cgraph = false;
        for (size_t j = 0; j < n_backends && !missing_cgraph; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i = 0; i < n_subgraphs; i++) {
                if (i >= bcj.cgraphs.size() || bcj.cgraphs[i].cgraph_main == nullptr) {
                    missing_cgraph = true;
                    break;
                }
            }
        }
        // ctx.reset() frees every prior subgraph graph. Only filling 0..n_subgraphs-1
        // leaves higher slots dangling; a later graph with more subgraphs then
        // hash_set_reset/nodes[]-store SIGSEGVs (Flash Next first decode).
        if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs || missing_cgraph) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            const size_t graph_cap = (size_t) cgraph->n_nodes;
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (auto & cg : bcj.cgraphs) {
                    cg.cgraph_main = nullptr;
                }
                for (size_t i = 0; i < n_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), graph_cap, /*grads =*/ false);
                    if (bcj.cgraphs[i].cgraph_main == nullptr) {
                        fprintf(stderr, "[META_GRAPH] ggml_new_graph_custom failed j=%zu i=%zu cap=%zu\n",
                                j, i, graph_cap);
                        return GGML_STATUS_FAILED;
                    }
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                // hash.used is a uint32_t bitset (4-byte aligned). data_ptr_ok
                // requires 16-byte GPU buffers; using it here silently skipped
                // every Flash-Next subgraph (7182 nodes → uniform 1/vocab logits).
                if (cgraph_ij == nullptr ||
                        cgraph_ij->nodes == nullptr ||
                        cgraph_ij->visited_hash_set.used == nullptr) {
                    continue;
                }
                // Flash Next TP: a dangling subgraph graph can report hash size 2^32+1
                // and ggml_hash_set_reset then memset ~512MiB into a small pool (SIGSEGV).
                if (cgraph_ij->visited_hash_set.size == 0 ||
                        cgraph_ij->visited_hash_set.size > (1ull << 22) ||
                        cgraph_ij->size <= 0) {
                    fprintf(stderr, "[META_GRAPH] skip subgraph j=%zu i=%zu hash_size=%zu graph_size=%d parent_n_nodes=%d\n",
                            j, i_graph, cgraph_ij->visited_hash_set.size, cgraph_ij->size, cgraph->n_nodes);
                    continue;
                }
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                int n_kept = 0;
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    ggml_tensor * node_ij = bcj.nodes[i_node];
                    if (node_ij == nullptr) {
                        ggml_tensor * orig = cgraph->nodes[i_node];
                        static int nskip;
                        if (j == 1 && nskip < 12 && orig &&
                                (orig->op == GGML_OP_ADD || orig->op == GGML_OP_VIEW || orig->op == GGML_OP_SCALE)) {
                            nskip++;
                            fprintf(stderr, "[META_SKIP] j=1 i=%zu op=%s name=%s\n",
                                    i_node, ggml_op_name(orig->op), orig->name);
                        }
                        continue;
                    }
                    // Host/meta views copied through as the original tensor have no
                    // device buffer (Qwen3.5 cache_r reshape/view). HRX then fails
                    // with "external value N is not bound".
                    if (node_ij->buffer == nullptr || ggml_backend_buffer_is_host(node_ij->buffer) ||
                            ggml_backend_buffer_is_meta(node_ij->buffer) ||
                            !ggml_backend_meta_data_ptr_ok(node_ij->data)) {
                        continue;
                    }
                    cgraph_ij->nodes[n_kept] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    if (hash_pos_orig != GGML_HASHSET_FULL && hash_pos_ij != GGML_HASHSET_FULL) {
                        cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                    }
                    n_kept++;
                    auto it_g = g_innern_gemm.find(node_ij);
                    if (it_g != g_innern_gemm.end() && it_g->second != nullptr &&
                            n_kept < cgraph_ij->size &&
                            it_g->second->data != nullptr &&
                            ggml_backend_meta_data_ptr_ok(it_g->second->data)) {
                        cgraph_ij->nodes[n_kept] = it_g->second;
                        ggml_hash_insert(&cgraph_ij->visited_hash_set, it_g->second);
                        n_kept++;
                    }
                    auto it_x = g_gdn_x_full.find(node_ij);
                    if (it_x != g_gdn_x_full.end() && it_x->second != nullptr &&
                            n_kept < cgraph_ij->size &&
                            it_x->second->data != nullptr &&
                            ggml_backend_meta_data_ptr_ok(it_x->second->data)) {
                        cgraph_ij->nodes[n_kept] = it_x->second;
                        ggml_hash_insert(&cgraph_ij->visited_hash_set, it_x->second);
                        n_kept++;
                    }
                }
                cgraph_ij->n_nodes = n_kept;
                // Fresh uid on rebuild so T=8 prefill graphs are not GRAPH_RECOMPUTE'd
                // onto T=1 decode buffers (k_set_rows aperture fault). Skip-rebuild
                // graph_sig keeps these uids for token 2+ of the same shape.
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        auto last_node = [&](size_t j) -> ggml_tensor * {
            ggml_cgraph * cg = backend_ctx->backend_configs[j].cgraphs[i].cgraph_main;
            if (cg == nullptr || cg->n_nodes <= 0) {
                return nullptr;
            }
            ggml_tensor * n = cg->nodes[cg->n_nodes - 1];
            if (n == nullptr || !ggml_backend_meta_data_ptr_ok(n->data)) {
                return nullptr;
            }
            return n;
        };
        for (size_t j = 0; j < n_backends; j++) {
            if (last_node(j) == nullptr) {
                return GGML_STATUS_SUCCESS;
            }
        }

        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_cgraph * cg = bcj.cgraphs[i].cgraph_main;
            if (cg == nullptr || cg->n_nodes <= 0 || cg->nodes[cg->n_nodes - 1] == nullptr) {
                continue;
            }
            ggml_tensor * node = cg->nodes[cg->n_nodes - 1];
            if (!ggml_backend_meta_data_ptr_ok(node->data)) {
                continue;
            }
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            ggml_tensor * node_zero = get_node_aux(node);
            node_zero->op = GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
            node_zero->src[0] = node;
            ggml_set_op_params_f32(node_zero, 0, 0.0f);
            node_zero->data = node->data;
            node_zero->buffer = node->buffer;
            node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

            step_cgraphs[j] = get_cgraph_aux();
            step_cgraphs[j]->nodes[0] = node_zero;
            step_cgraphs[j]->n_nodes = 1;
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            push_data(j_src, j_dst, i_buf);
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                push_data(j, j_other, i_buf);
            }

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
        }

        return GGML_STATUS_SUCCESS;
    };


    {
        static int nlogg;
        if (nlogg < 6) {
            nlogg++;
            int nempty = 0, npos = 0, sumk = 0, maxk = 0;
            for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
                ggml_cgraph * cg = backend_ctx->backend_configs[0].cgraphs[i].cgraph_main;
                const int n = cg ? cg->n_nodes : -1;
                if (n <= 0) {
                    nempty++;
                } else {
                    npos++;
                    sumk += n;
                    if (n > maxk) {
                        maxk = n;
                    }
                }
            }
            fprintf(stderr, "[META_GC] g=%d parent_n=%d n_subgraphs=%zu n_backends=%zu nempty=%d npos=%d sumk=%d maxk=%d\n",
                    nlogg, cgraph->n_nodes, backend_ctx->n_subgraphs, n_backends, nempty, npos, sumk, maxk);
            const size_t nshow = std::min(backend_ctx->n_subgraphs, (size_t) 4);
            for (size_t i = 0; i < nshow; i++) {
                for (size_t j = 0; j < n_backends; j++) {
                    ggml_cgraph * cg = backend_ctx->backend_configs[j].cgraphs[i].cgraph_main;
                    const char * last = (cg && cg->n_nodes > 0 && cg->nodes[cg->n_nodes - 1])
                            ? cg->nodes[cg->n_nodes - 1]->name : "";
                    fprintf(stderr, "[META_GC] g=%d sub=%zu j=%zu n=%d last=%s\n",
                            nlogg, i, j, cg ? cg->n_nodes : -1, last);
                    if (j == 0 && cg != nullptr) {
                        const int nprint = std::min(cg->n_nodes, 12);
                        for (int ni = 0; ni < nprint; ni++) {
                            ggml_tensor * nd = cg->nodes[ni];
                            if (nd == nullptr) {
                                continue;
                            }
                            fprintf(stderr, "[META_GC]   sub=%zu i=%d op=%s name=%s ne=[%lld,%lld,%lld]\n",
                                    i, ni, ggml_op_name(nd->op), nd->name,
                                    (long long) nd->ne[0], (long long) nd->ne[1], (long long) nd->ne[2]);
                        }
                    }
                }
            }
        }
    }

    if (backend_ctx->comm_graph_seq != nullptr && backend_ctx->comm_ctx != nullptr &&
            backend_ctx->n_subgraphs >= 2 && n_backends >= 2 &&
            getenv("LLAMA_TP_NO_SEQ") == nullptr) {
        size_t j_local = n_backends, j_rpc = n_backends;
        for (size_t j = 0; j < n_backends; j++) {
            const char * bname = ggml_backend_name(backend_ctx->backend_configs[j].backend);
            const bool is_rpc = bname != nullptr && strncmp(bname, "RPC", 3) == 0;
            if (is_rpc) {
                j_rpc = j;
            } else if (j_local == n_backends) {
                j_local = j;
            }
        }
        if (j_local < n_backends && j_rpc < n_backends) {
            const size_t ns = backend_ctx->n_subgraphs;
            if (!backend_ctx->seq_plan_valid || backend_ctx->seq_local_gs.size() != ns ||
                    backend_ctx->seq_j_local != j_local || backend_ctx->seq_j_rpc != j_rpc) {
                backend_ctx->seq_local_gs.assign(ns, nullptr);
                backend_ctx->seq_local_ar.assign(ns, nullptr);
                backend_ctx->seq_rpc_uids.assign(ns, 0);
                backend_ctx->seq_rpc_ar.assign(ns, nullptr);
                bool seq_ok = true;
                for (size_t i = 0; i < ns; i++) {
                    ggml_cgraph * lg = backend_ctx->backend_configs[j_local].cgraphs[i].cgraph_main;
                    ggml_cgraph * rg = backend_ctx->backend_configs[j_rpc].cgraphs[i].cgraph_main;
                    backend_ctx->seq_local_gs[i] = lg;
                    if (rg == nullptr || rg->uid == 0) {
                        seq_ok = false;
                        break;
                    }
                    backend_ctx->seq_rpc_uids[i] = rg->uid;
                    if (i + 1 >= ns) {
                        continue;
                    }
                    const size_t i_node_stop = backend_ctx->backend_configs[0].cgraphs[i + 1].offset;
                    if (i_node_stop == 0) {
                        continue;
                    }
                    ggml_tensor * orig_last = cgraph->nodes[i_node_stop - 1];
                    if (orig_last == nullptr) {
                        continue;
                    }
                    const ggml_backend_meta_split_axis last_axis =
                        ggml_backend_meta_get_split_state(orig_last, /*assume_sync =*/ false).axis;
                    const bool last_partial = last_axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                    const bool last_ffn = strstr(orig_last->name, "ffn_moe") != nullptr;
                    ggml_tensor * loc = ggml_backend_meta_buffer_simple_tensor(orig_last, j_local);
                    static const bool nspl = getenv("LLAMA_TP_SSM_OUT_NSPLIT") != nullptr;
                    auto it_x = (loc != nullptr) ? g_gdn_x_full.find(loc) : g_gdn_x_full.end();
                    auto it_g = (loc != nullptr) ? g_innern_gemm.find(loc) : g_innern_gemm.end();
                    if (it_x != g_gdn_x_full.end() && it_x->second != nullptr) {
                        // 3072 3-rep shard → sequential 6144 (interleave 1024).
                        backend_ctx->seq_local_ar[i] = loc;
                        backend_ctx->seq_rpc_ar[i] = it_x->second;
                    } else if (it_g != g_innern_gemm.end() && it_g->second != nullptr) {
                        // INNERN: xchg 1280 gemm, concat into 2560 dest.
                        backend_ctx->seq_local_ar[i] = it_g->second;
                        backend_ctx->seq_rpc_ar[i] = loc;
                    } else if (nspl && last_partial && !last_ffn && loc != nullptr &&
                            loc->type == GGML_TYPE_F32 && loc->ne[0] > 1) {
                        // Split GDN: HIP wrote local heads into the first half of a
                        // full-sized dest. Concat that half with the peer's.
                        backend_ctx->seq_local_ar[i] = loc;
                        backend_ctx->seq_rpc_ar[i] = loc;
                    } else if (last_partial || last_ffn) {
                        backend_ctx->seq_local_ar[i] = loc;
                        backend_ctx->seq_rpc_ar[i] = nullptr; // ADD
                    }
                }
                if (seq_ok) {
                    backend_ctx->seq_plan_valid = true;
                    backend_ctx->seq_j_local = j_local;
                    backend_ctx->seq_j_rpc = j_rpc;
                } else {
                    backend_ctx->seq_plan_valid = false;
                }
            }
            if (backend_ctx->seq_plan_valid) {
                static int nseq;
                if (nseq < 4) {
                    nseq++;
                    fprintf(stderr, "[META_SEQ] n_subgraphs=%zu cached=%d\n",
                            ns, nseq > 1 ? 1 : 0);
                }
                if (backend_ctx->comm_graph_seq(backend_ctx->comm_ctx,
                        backend_ctx->backend_configs[j_local].backend,
                        backend_ctx->seq_local_gs.data(), backend_ctx->seq_local_ar.data(),
                        backend_ctx->seq_rpc_uids.data(), backend_ctx->seq_rpc_ar.data(), ns)) {
                    return GGML_STATUS_SUCCESS;
                }
                backend_ctx->seq_plan_valid = false;
                static int nseq_skip;
                if (nseq_skip < 4) {
                    nseq_skip++;
                    fprintf(stderr, "[META_SEQ] skip (uids not cached yet)\n");
                }
            }
        }
    }

    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        bool any_nodes = false;
        bool innern_ag = false;
        int64_t innern_wN = 0;
        int64_t innern_nT = 0;
        std::vector<std::vector<float>> innern_parts(n_backends);
        std::vector<ggml_tensor *> innern_dev(n_backends, nullptr);
        // Local HIP graph_compute queues kernels and returns. RPC RECOMPUTE
        // blocks on the USB4 ACK (worker HIP). Launch non-RPC first so local
        // kernels overlap the RPC round-trip. Syncing inside the j-loop
        // serialized them (~1ms * 97 subgraphs).
        // Piggyback FFN AllReduce onto GRAPH_RECOMPUTE (one USB4 RTT instead of
        // RECOMPUTE+ALL_REDUCE). Skip when INNERN/NSPLIT owns the residual.
        ggml_tensor * piggy_local = nullptr;
        ggml_tensor * piggy_rpc   = nullptr;
        static const bool skip_piggy = getenv("LLAMA_TP_NO_PIGGY") != nullptr ||
                                getenv("LLAMA_TP_SSM_OUT_NSPLIT") != nullptr ||
                                getenv("LLAMA_TP_SKIP_AR") != nullptr;
        if (!skip_piggy && n_backends > 1 && i < backend_ctx->n_subgraphs - 1 &&
                backend_ctx->comm_set_pending_ar != nullptr && backend_ctx->comm_ctx != nullptr) {
            const size_t i_node_stop = backend_ctx->backend_configs[0].cgraphs[i + 1].offset;
            if (i_node_stop != 0) {
                ggml_tensor * orig_last = cgraph->nodes[i_node_stop - 1];
                ggml_backend_meta_split_axis last_axis = GGML_BACKEND_SPLIT_AXIS_UNKNOWN;
                if (orig_last != nullptr) {
                    last_axis = ggml_backend_meta_get_split_state(orig_last, /*assume_sync =*/ false).axis;
                }
                const bool last_partial = last_axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                const bool last_ffn = orig_last != nullptr && strstr(orig_last->name, "ffn_moe") != nullptr;
                if (orig_last != nullptr && (last_partial || last_ffn)) {
                    for (size_t j = 0; j < n_backends; j++) {
                        ggml_tensor * shard = ggml_backend_meta_buffer_simple_tensor(orig_last, j);
                        if (shard == nullptr || shard->data == nullptr ||
                                !ggml_backend_meta_data_ptr_ok(shard->data)) {
                            piggy_local = nullptr;
                            piggy_rpc = nullptr;
                            break;
                        }
                        const char * bname = ggml_backend_name(backend_ctx->backend_configs[j].backend);
                        const bool is_rpc = bname != nullptr && strncmp(bname, "RPC", 3) == 0;
                        if (is_rpc) {
                            piggy_rpc = shard;
                        } else if (piggy_local == nullptr) {
                            piggy_local = shard;
                        }
                    }
                    if (piggy_local != nullptr && piggy_rpc != nullptr) {
                        bool rpc_has_nodes = false;
                        for (size_t j = 0; j < n_backends; j++) {
                            const char * bname = ggml_backend_name(backend_ctx->backend_configs[j].backend);
                            const bool is_rpc = bname != nullptr && strncmp(bname, "RPC", 3) == 0;
                            ggml_cgraph * rg = backend_ctx->backend_configs[j].cgraphs[i].cgraph_main;
                            if (is_rpc && rg != nullptr && rg->n_nodes > 0) {
                                rpc_has_nodes = true;
                                break;
                            }
                        }
                        if (rpc_has_nodes) {
                            backend_ctx->comm_set_pending_ar(backend_ctx->comm_ctx, piggy_rpc);
                        } else {
                            piggy_local = nullptr;
                            piggy_rpc = nullptr;
                        }
                    } else {
                        piggy_local = nullptr;
                        piggy_rpc = nullptr;
                    }
                }
            }
        }
        ggml_status compute_status = GGML_STATUS_SUCCESS;
        auto launch = [&](bool rpc) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                const char * bname = ggml_backend_name(bcj.backend);
                const bool is_rpc = bname != nullptr && strncmp(bname, "RPC", 3) == 0;
                if (is_rpc != rpc) {
                    continue;
                }
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                if (cgraph_ij == nullptr || cgraph_ij->n_nodes <= 0) {
                    continue;
                }
                any_nodes = true;
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, cgraph_ij);
                if (status != GGML_STATUS_SUCCESS) {
                    compute_status = status;
                }
            }
        };
        launch(false);
        launch(true);
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
            if (cgraph_ij == nullptr || cgraph_ij->n_nodes <= 0) {
                continue;
            }
            ggml_backend_synchronize(bcj.backend);
        }
        if (compute_status != GGML_STATUS_SUCCESS) {
            return compute_status;
        }
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
            if (cgraph_ij == nullptr || cgraph_ij->n_nodes <= 0) {
                continue;
            }
            for (int ni = 0; ni < cgraph_ij->n_nodes; ni++) {
                ggml_tensor * nd = cgraph_ij->nodes[ni];
                if (nd == nullptr) {
                    continue;
                }
                auto it_g = g_innern_gemm.find(nd);
                if (it_g == g_innern_gemm.end() || it_g->second == nullptr) {
                    continue;
                }
                ggml_tensor * gemm = it_g->second;
                const size_t nb = ggml_nbytes(gemm);
                innern_parts[j].assign(nb / sizeof(float), 0.f);
                ggml_backend_tensor_get(gemm, innern_parts[j].data(), 0, nb);
                innern_wN = gemm->ne[0];
                innern_nT = ggml_nrows(gemm);
                innern_dev[j] = nd; // 2560 PARTIAL dest
                innern_ag = true;
                static int nag;
                if (nag < 24) {
                    nag++;
                    float xh[4] = {0, 0, 0, 0};
                    float yh[4] = {0, 0, 0, 0};
                    if (gemm->src[1] != nullptr && ggml_nbytes(gemm->src[1]) >= 16) {
                        ggml_backend_tensor_get(gemm->src[1], xh, 0, 16);
                    }
                    if (nb >= 16) {
                        memcpy(yh, innern_parts[j].data(), 16);
                    }
                    fprintf(stderr, "[INNERN_AG] j=%zu gemm=%s dst=%s wN=%lld nT=%lld nbytes=%zu x0=%.5f,%.5f y0=%.5f,%.5f\n",
                            j, gemm->name, nd->name, (long long) innern_wN,
                            (long long) innern_nT, nb, xh[0], xh[1], yh[0], yh[1]);
                }
            }
        }

        if (!any_nodes) {
            continue;
        }

        if (innern_ag && n_backends >= 2 && innern_wN > 0 && innern_nT > 0 &&
                innern_parts[0].size() == (size_t) innern_wN * (size_t) innern_nT &&
                innern_parts[1].size() == innern_parts[0].size() &&
                innern_dev[0] != nullptr && innern_dev[1] != nullptr) {
            const int64_t N = innern_wN * 2;
            std::vector<float> full((size_t) N * (size_t) innern_nT);
            for (int64_t t = 0; t < innern_nT; t++) {
                memcpy(full.data() + t * N,
                       innern_parts[0].data() + t * innern_wN,
                       (size_t) innern_wN * sizeof(float));
                memcpy(full.data() + t * N + innern_wN,
                       innern_parts[1].data() + t * innern_wN,
                       (size_t) innern_wN * sizeof(float));
            }
            const size_t full_nb = full.size() * sizeof(float);
            const size_t dst_nb0 = ggml_nbytes(innern_dev[0]);
            const size_t dst_nb1 = ggml_nbytes(innern_dev[1]);
            static int nset;
            if (nset < 4) {
                nset++;
                fprintf(stderr, "[INNERN_SET] full_nb=%zu dst_nb=%zu,%zu ne0=%lld,%lld,%lld wN=%lld nT=%lld y0=%.5f yN=%.5f\n",
                        full_nb, dst_nb0, dst_nb1,
                        (long long) innern_dev[0]->ne[0], (long long) innern_dev[0]->ne[1],
                        (long long) innern_dev[0]->ne[2], (long long) innern_wN, (long long) innern_nT,
                        innern_parts[0][0], innern_parts[1][0]);
            }
            GGML_ASSERT(full_nb == dst_nb0 && full_nb == dst_nb1);
            ggml_backend_tensor_set(innern_dev[0], full.data(), 0, full_nb);
            ggml_backend_tensor_set(innern_dev[1], full.data(), 0, full_nb);
            {
                float got0[2] = {0, 0}, gotN[2] = {0, 0};
                ggml_backend_tensor_get(innern_dev[0], got0, 0, 8);
                ggml_backend_tensor_get(innern_dev[1], gotN, (size_t) innern_wN * sizeof(float), 8);
                static int nchk;
                if (nchk < 4) {
                    nchk++;
                    fprintf(stderr, "[INNERN_SETCHK] dst0[0]=%.5f expect=%.5f dst1[wN]=%.5f expect=%.5f\n",
                            got0[0], innern_parts[0][0], gotN[0], innern_parts[1][0]);
                }
            }
            {
                static int nresid_set;
                if (nresid_set < 4) {
                    nresid_set++;
                    for (size_t j = 0; j < n_backends; j++) {
                        ggml_backend_meta_dump_resid("after_set", j, innern_dev[j]);
                    }
                    if (i + 1 < backend_ctx->n_subgraphs) {
                        for (size_t j = 0; j < n_backends; j++) {
                            ggml_cgraph * cg = backend_ctx->backend_configs[j].cgraphs[i + 1].cgraph_main;
                            if (cg == nullptr) {
                                continue;
                            }
                            int ndump = 0;
                            for (int ni = 0; ni < cg->n_nodes && ndump < 8; ni++) {
                                ggml_tensor * nd = cg->nodes[ni];
                                if (nd == nullptr || !ggml_backend_meta_resid_name(nd->name)) {
                                    continue;
                                }
                                ggml_backend_meta_dump_resid("next_pre", j, nd);
                                ndump++;
                            }
                        }
                    }
                }
            }
            continue; // already concatenated; AllReduce would double
        }

        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1 && getenv("LLAMA_TP_SKIP_AR") == nullptr) {
            const size_t i_node_stop = i + 1 < backend_ctx->n_subgraphs
                    ? backend_ctx->backend_configs[0].cgraphs[i + 1].offset
                    : (size_t) cgraph->n_nodes;
            if (i_node_stop == 0) {
                continue;
            }
            ggml_tensor * orig_last = cgraph->nodes[i_node_stop - 1];
            // Delay-AR of ffn_moe_down through ADD/GET_ROWS types ffn_moe_out
            // MIRRORED (ADD(PARTIAL, MIRRORED) handler) without ever reducing
            // the expert down-proj. INNERN then skips GDN AR and layer-1 GDN x
            // diverges. AllReduce ffn_moe_* even when the delayed last is
            // labelled MIRRORED; the buffers still hold a partial sum.
            ggml_backend_meta_split_axis last_axis = GGML_BACKEND_SPLIT_AXIS_UNKNOWN;
            if (orig_last != nullptr) {
                last_axis = ggml_backend_meta_get_split_state(orig_last, /*assume_sync =*/ false).axis;
            }
            const bool last_partial = last_axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
            const bool last_ffn = orig_last != nullptr && strstr(orig_last->name, "ffn_moe") != nullptr;
            static const bool nspl_fb = getenv("LLAMA_TP_SSM_OUT_NSPLIT") != nullptr;
            if (nspl_fb && orig_last != nullptr && orig_last->name[0] &&
                    strstr(orig_last->name, "final_output") != nullptr) {
                ggml_tensor * s0 = ggml_backend_meta_buffer_simple_tensor(orig_last, 0);
                ggml_tensor * s1 = ggml_backend_meta_buffer_simple_tensor(orig_last, 1);
                auto ix0 = (s0 != nullptr) ? g_gdn_x_full.find(s0) : g_gdn_x_full.end();
                auto ix1 = (s1 != nullptr) ? g_gdn_x_full.find(s1) : g_gdn_x_full.end();
                if (ix0 != g_gdn_x_full.end() && ix1 != g_gdn_x_full.end() &&
                        ix0->second != nullptr && ix1->second != nullptr) {
                    const size_t shard_nb = ggml_nbytes(s0);
                    const size_t full_nb = ggml_nbytes(ix0->second);
                    const int64_t nT = std::max((int64_t) 1, ggml_nrows(ix0->second));
                    if (s0->ne[0] == 3072 && ix0->second->ne[0] == 6144 &&
                            shard_nb * 2 == full_nb) {
                        std::vector<float> left(shard_nb / sizeof(float));
                        std::vector<float> right(shard_nb / sizeof(float));
                        std::vector<float> full(full_nb / sizeof(float));
                        ggml_backend_tensor_get(s0, left.data(), 0, shard_nb);
                        ggml_backend_tensor_get(s1, right.data(), 0, shard_nb);
                        const int64_t chunk = 1024;
                        const int64_t n_chunks = 3;
                        const int64_t shard = chunk * n_chunks;
                        for (int64_t t = 0; t < nT; t++) {
                            for (int64_t g = 0; g < n_chunks; g++) {
                                memcpy(full.data() + t * 2 * shard + (2 * g) * chunk,
                                       left.data() + t * shard + g * chunk, (size_t) chunk * sizeof(float));
                                memcpy(full.data() + t * 2 * shard + (2 * g + 1) * chunk,
                                       right.data() + t * shard + g * chunk, (size_t) chunk * sizeof(float));
                            }
                        }
                        ggml_backend_tensor_set(ix0->second, full.data(), 0, full_nb);
                        ggml_backend_tensor_set(ix1->second, full.data(), 0, full_nb);
                        static int nxag;
                        if (nxag < 4) {
                            nxag++;
                            fprintf(stderr, "[GDN_X_AG] fallback interleave last=%s shard=%zu full=%zu nT=%lld\n",
                                    orig_last->name, shard_nb, full_nb, (long long) nT);
                        }
                        continue;
                    }
                }
            }
            if (nspl_fb && last_partial && !last_ffn && orig_last != nullptr &&
                    orig_last->type == GGML_TYPE_F32) {
                ggml_tensor * d0 = ggml_backend_meta_buffer_simple_tensor(orig_last, 0);
                ggml_tensor * d1 = ggml_backend_meta_buffer_simple_tensor(orig_last, 1);
                if (d0 != nullptr && d1 != nullptr && ggml_nbytes(d0) == ggml_nbytes(d1) &&
                        d0->ne[0] > 1 && (d0->ne[0] % 2) == 0) {
                    const size_t full_nb = ggml_nbytes(d0);
                    const size_t half_nb = full_nb / 2;
                    const int64_t nT = std::max((int64_t) 1, ggml_nrows(d0));
                    const int64_t wN = d0->ne[0] / 2;
                    if ((size_t) wN * (size_t) nT * sizeof(float) == half_nb) {
                        std::vector<float> left(half_nb / sizeof(float));
                        std::vector<float> right(half_nb / sizeof(float));
                        std::vector<float> full(full_nb / sizeof(float));
                        ggml_backend_tensor_get(d0, left.data(), 0, half_nb);
                        ggml_backend_tensor_get(d1, right.data(), 0, half_nb);
                        for (int64_t t = 0; t < nT; t++) {
                            memcpy(full.data() + t * (2 * wN), left.data() + t * wN, (size_t) wN * sizeof(float));
                            memcpy(full.data() + t * (2 * wN) + wN, right.data() + t * wN, (size_t) wN * sizeof(float));
                        }
                        ggml_backend_tensor_set(d0, full.data(), 0, full_nb);
                        ggml_backend_tensor_set(d1, full.data(), 0, full_nb);
                        static int nag_gdn;
                        if (nag_gdn < 4) {
                            nag_gdn++;
                            fprintf(stderr, "[GDN_AG] fallback concat last=%s half=%zu wN=%lld nT=%lld\n",
                                    orig_last->name, half_nb, (long long) wN, (long long) nT);
                        }
                        continue;
                    }
                }
            }
            if (orig_last == nullptr || (!last_partial && !last_ffn)) {
                static int nskipar;
                if (nskipar < 8) {
                    nskipar++;
                    fprintf(stderr, "[RESID] skip_ar sub=%zu last=%s axis=%s\n",
                            i, orig_last ? orig_last->name : "(null)",
                            ggml_backend_meta_split_axis_name(last_axis));
                }
                continue;
            }
            bool all_ok = true;
            std::vector<ggml_tensor *> nodes;
            nodes.reserve(n_backends);
            for (size_t j = 0; j < n_backends; j++) {
                ggml_tensor * shard = ggml_backend_meta_buffer_simple_tensor(orig_last, j);
                if (shard == nullptr || shard->data == nullptr) {
                    all_ok = false;
                    break;
                }
                if (!ggml_backend_meta_data_ptr_ok(shard->data)) {
                    all_ok = false;
                    break;
                }
                nodes.push_back(shard);
            }
            if (!all_ok) {
                continue;
            }
            {
                static int nresid_ar;
                if (nresid_ar < 6) {
                    nresid_ar++;
                    fprintf(stderr, "[RESID] ar_pre sub=%zu last=%s axis=%s\n",
                            i, orig_last->name,
                            ggml_backend_meta_split_axis_name(
                                ggml_backend_meta_get_split_state(orig_last, /*assume_sync =*/ false).axis));
                    for (size_t j = 0; j < nodes.size(); j++) {
                        ggml_backend_meta_dump_resid("ar_pre", j, nodes[j]);
                    }
                }
            }
            bool backend_allreduce_success = false;
            if (backend_ctx->comm_finish_ar != nullptr && piggy_local != nullptr) {
                backend_allreduce_success = backend_ctx->comm_finish_ar(backend_ctx->comm_ctx, piggy_local);
            }
            if (!backend_allreduce_success && backend_ctx->comm_ctx && getenv("LLAMA_TP_AR_FALLBACK") == nullptr) {
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            {
                static int nresid_ar2;
                if (nresid_ar2 < 6) {
                    nresid_ar2++;
                    for (size_t j = 0; j < nodes.size(); j++) {
                        ggml_backend_meta_dump_resid("ar_post", j, nodes[j]);
                    }
                    for (size_t j = 0; j < n_backends; j++) {
                        ggml_cgraph * cg = backend_ctx->backend_configs[j].cgraphs[i].cgraph_main;
                        if (cg == nullptr) {
                            continue;
                        }
                        int ndump = 0;
                        for (int ni = 0; ni < cg->n_nodes && ndump < 6; ni++) {
                            ggml_tensor * nd = cg->nodes[ni];
                            if (nd == nullptr || !ggml_backend_meta_resid_name(nd->name)) {
                                continue;
                            }
                            ggml_backend_meta_dump_resid("sub_post", j, nd);
                            ndump++;
                        }
                    }
                }
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}
