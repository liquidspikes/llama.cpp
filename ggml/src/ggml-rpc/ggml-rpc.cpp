#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"

#include <array>
#include <cinttypes>
#include <optional>
#include <string>
#include <vector>
#include <queue>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <thread>
#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#endif

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

/* IMP-24: AVX-512 F32 add + F16C F16 add. This CPU has avx512f/f16c, not avx512fp16,
 * so F16 uses cvtph/ps + add_ps (never _mm512_add_ph). */
static void allreduce_add_f32(float * dst, const float * src, size_t count) {
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 64 <= count; i += 64) {
        __m512 d0 = _mm512_loadu_ps(dst + i);
        __m512 s0 = _mm512_loadu_ps(src + i);
        __m512 d1 = _mm512_loadu_ps(dst + i + 16);
        __m512 s1 = _mm512_loadu_ps(src + i + 16);
        __m512 d2 = _mm512_loadu_ps(dst + i + 32);
        __m512 s2 = _mm512_loadu_ps(src + i + 32);
        __m512 d3 = _mm512_loadu_ps(dst + i + 48);
        __m512 s3 = _mm512_loadu_ps(src + i + 48);
        _mm512_storeu_ps(dst + i,      _mm512_add_ps(d0, s0));
        _mm512_storeu_ps(dst + i + 16, _mm512_add_ps(d1, s1));
        _mm512_storeu_ps(dst + i + 32, _mm512_add_ps(d2, s2));
        _mm512_storeu_ps(dst + i + 48, _mm512_add_ps(d3, s3));
    }
    for (; i + 16 <= count; i += 16) {
        _mm512_storeu_ps(dst + i, _mm512_add_ps(_mm512_loadu_ps(dst + i),
                                                _mm512_loadu_ps(src + i)));
    }
#elif defined(__AVX2__)
    for (; i + 8 <= count; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i),
                                                _mm256_loadu_ps(src + i)));
    }
#endif
    for (; i < count; i++) {
        dst[i] += src[i];
    }
}

static void allreduce_add_f16(ggml_fp16_t * dst, const ggml_fp16_t * src, size_t count) {
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 64 <= count; i += 64) {
        for (int k = 0; k < 64; k += 16) {
            const __m256i h = _mm256_loadu_si256((const __m256i *)(dst + i + (size_t)k));
            const __m256i s = _mm256_loadu_si256((const __m256i *)(src + i + (size_t)k));
            const __m512 sum = _mm512_add_ps(_mm512_cvtph_ps(h), _mm512_cvtph_ps(s));
            _mm256_storeu_si256((__m256i *)(dst + i + (size_t)k),
                                _mm512_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }
    for (; i + 16 <= count; i += 16) {
        const __m256i h = _mm256_loadu_si256((const __m256i *)(dst + i));
        const __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
        const __m512 sum = _mm512_add_ps(_mm512_cvtph_ps(h), _mm512_cvtph_ps(s));
        _mm256_storeu_si256((__m256i *)(dst + i),
                            _mm512_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
#elif defined(__AVX2__) && defined(__F16C__)
    for (; i + 8 <= count; i += 8) {
        const __m128i h = _mm_loadu_si128((const __m128i *)(dst + i));
        const __m128i s = _mm_loadu_si128((const __m128i *)(src + i));
        const __m256 sum = _mm256_add_ps(_mm256_cvtph_ps(h), _mm256_cvtph_ps(s));
        _mm_storeu_si128((__m128i *)(dst + i), _mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
#endif
    for (; i < count; i++) {
        dst[i] = ggml_compute_fp32_to_fp16(ggml_compute_fp16_to_fp32(dst[i]) +
                                           ggml_compute_fp16_to_fp32(src[i]));
    }
}

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)


namespace fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    int32_t use_count;
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_ALL_REDUCE,
    RPC_CMD_NONE,
    RPC_CMD_COUNT,
};

// Try RPC_CMD_SET_TENSOR_HASH first for all tensors
// Skip SET_TENSOR_HASH for USB4STREAM bulk load. Hash-then-body on the same
// tbstripe pipe desynchronized 4KiB frames (magic mismatch / cmd=7 abort)
// when meta TP shards ~2MiB SET_TENSOR payloads. Cache still helps after a
// successful first fill if HASH_THRESHOLD is left 0 via env.
static size_t rpc_hash_threshold() {
    static size_t thr = []() {
        const char * e = std::getenv("GGML_RPC_HASH_THRESHOLD");
        if (e && *e) {
            return (size_t) std::strtoull(e, nullptr, 10);
        }
        return (size_t) (64ull * 1024ull * 1024ull);
    }();
    return thr;
}

// USB4STREAM drops frames in large tbs_send. Combined SET_TENSOR must
// fit in one 4064-byte payload (rpc_tensor+hdr ~321 bytes). Default
// 2048. 4096 is two frames and deadlocks after ~11k tensors. Override
// with GGML_RPC_SET_TENSOR_CHUNK (min 256).
static size_t rpc_set_tensor_chunk() {
    static size_t chunk = []() {
        const char * e = std::getenv("GGML_RPC_SET_TENSOR_CHUNK");
        if (e && *e) {
            size_t v = (size_t) std::strtoull(e, nullptr, 10);
            if (v >= 256) {
                return v;
            }
        }
        return (size_t) 2048;
    }();
    return chunk;
}

static inline bool rpc_perf_enabled() {
    static int e = -1;
    if (e == -1) {
        const char * v = getenv("GGML_RPC_PERF");
        e = (v && strcmp(v, "1") == 0) ? 1 : 0;
    }
    return e == 1;
}

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
};

#pragma pack(pop)

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
    uint64_t    last_graph_uid;
};

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

class rpc_dispatcher;
struct ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    uint32_t                        device;
    std::string                     name;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    void                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

static inline bool tb_debug_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * e = std::getenv("GGML_TB_DEBUG");
        enabled = (e && strcmp(e, "0") != 0 && strcmp(e, "") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    const uint64_t fnv_prime = 0x100000001b3ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static inline uint32_t rpc_cmd_to_channel(enum rpc_cmd cmd) {
    switch (cmd) {
        case RPC_CMD_SET_TENSOR:
        case RPC_CMD_SET_TENSOR_HASH:
        case RPC_CMD_GET_TENSOR:
        case RPC_CMD_GRAPH_COMPUTE:
        case RPC_CMD_ALL_REDUCE:
            return RPC_CHANNEL_STRIPED;
        default:
            return RPC_CHANNEL_CONTROL;
    }
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size, uint32_t channel = RPC_CHANNEL_CONTROL) {
    uint64_t sz = msg_size;
    if (sock->is_tbstripe()) {
        std::vector<uint8_t> buf(sizeof(sz) + msg_size);
        memcpy(buf.data(), &sz, sizeof(sz));
        if (msg && msg_size > 0) {
            memcpy(buf.data() + sizeof(sz), msg, msg_size);
        }
        if (!sock->send_data(buf.data(), buf.size())) {
            return false;
        }
        return sock->flush();
    }
    if (!sock->send_data_channel(RPC_CHANNEL_CONTROL, &sz, sizeof(sz))) {
        return false;
    }
    if (msg && msg_size > 0) {
        if (!sock->send_data_channel(channel, msg, msg_size)) {
            return false;
        }
    }
    return sock->flush();
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size, uint32_t channel = RPC_CHANNEL_CONTROL) {
    uint64_t size;
    if (!sock->recv_data_channel(RPC_CHANNEL_CONTROL, &size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    if (msg_size > 0) {
        return sock->recv_data_channel(channel, msg, msg_size);
    }
    return true;
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input, uint32_t channel = RPC_CHANNEL_CONTROL) {
    uint64_t size;
    if (!sock->recv_data_channel(RPC_CHANNEL_CONTROL, &size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        GGML_LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    if (size > 0) {
        return sock->recv_data_channel(channel, input.data(), size);
    }
    return true;
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    rpc_endpoint_info info;
    if (!socket_t::parse_endpoint(endpoint, info)) {
        return false;
    }
    if (info.kind == rpc_transport_kind::STREAM) {
        host = endpoint;
        port = 0;
        return true;
    }
    host = info.host;
    port = info.port;
    return true;
}

// RPC request : | rpc_cmd (1 byte) on CTRL | request_size (8 bytes) on CTRL | request_data on channel |
// No response
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    if (tb_debug_enabled()) fprintf(stderr, "[RPC_CMD send cmd=%d in_sz=%zu]\n", (int)cmd, input_size);
    uint32_t channel = rpc_cmd_to_channel(cmd);
    uint8_t opcode = static_cast<uint8_t>(cmd);
    if (sock->is_tbstripe()) {
        uint64_t sz = input_size;
        std::vector<uint8_t> buf(1 + sizeof(sz) + input_size);
        buf[0] = opcode;
        memcpy(buf.data() + 1, &sz, sizeof(sz));
        if (input && input_size > 0) {
            memcpy(buf.data() + 1 + sizeof(sz), input, input_size);
        }
        if (!sock->send_data(buf.data(), buf.size())) {
            return false;
        }
        return sock->flush();
    }
    if (!sock->send_data_channel(RPC_CHANNEL_CONTROL, &opcode, 1)) {
        return false;
    }
    if (!send_msg(sock, input, input_size, channel)) {
        return false;
    }
    return sock->flush();
}

// RPC request : | rpc_cmd (1 byte) on CTRL | request_size (8 bytes) on CTRL | request_data on channel |
// RPC response: | response_size (8 bytes) on CTRL | response_data on channel |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size,
                         int timeout_ms = -2, int max_try = -1) {
    if (tb_debug_enabled()) fprintf(stderr, "[RPC_CMD send_with_rsp cmd=%d in_sz=%zu out_sz=%zu]\n", (int)cmd, input_size, output_size);
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        GGML_LOG_ERROR("send_rpc_cmd: failed to send input for cmd=%d (size=%zu)\n", (int)cmd, input_size);
        return false;
    }
    uint32_t channel = rpc_cmd_to_channel(cmd);
    if (cmd == RPC_CMD_GET_TENSOR) {
        if (!sock->recv_data_channel(channel, output, output_size)) {
            GGML_LOG_ERROR("send_rpc_cmd: failed to recv output payload for cmd=%d (size=%zu)\n", (int)cmd, output_size);
            return false;
        }
        if (tb_debug_enabled()) fprintf(stderr, "[RPC_CMD GET_TENSOR recv completed!]\n");
        return true;
    }
    if (sock->is_tbstripe()) {
        std::vector<uint8_t> buf(sizeof(uint64_t) + output_size);
        if (timeout_ms == -2) {
            timeout_ms = (cmd == RPC_CMD_SET_TENSOR) ? 100 : -1;
        }
        if (max_try < 1) {
            max_try = (cmd == RPC_CMD_SET_TENSOR) ? 64 : 1;
        }
        for (int attempt = 0; attempt < max_try; attempt++) {
            if (attempt > 0) {
                if (!send_rpc_cmd(sock, cmd, input, input_size)) {
                    GGML_LOG_ERROR("send_rpc_cmd: retry send failed cmd=%d\n", (int)cmd);
                    return false;
                }
            }
            const bool ok = (timeout_ms < 0)
                ? sock->recv_data(buf.data(), buf.size())
                : sock->recv_data_timeout(buf.data(), buf.size(), timeout_ms);
            if (!ok) {
                continue;
            }
            uint64_t out_size = 0;
            memcpy(&out_size, buf.data(), sizeof(out_size));
            if (out_size != output_size) {
                GGML_LOG_ERROR("send_rpc_cmd: out_size mismatch for cmd=%d (got %" PRIu64 ", expected %zu)\n",
                               (int)cmd, out_size, output_size);
                return false;
            }
            if (output_size > 0) {
                memcpy(output, buf.data() + sizeof(out_size), output_size);
            }
            return true;
        }
        GGML_LOG_ERROR("send_rpc_cmd: tbstripe response timeout cmd=%d tries=%d\n",
                       (int)cmd, max_try);
        return false;
    }
    uint64_t out_size = 0;
    if (!sock->recv_data_channel(RPC_CHANNEL_CONTROL, &out_size, sizeof(out_size))) {
        GGML_LOG_ERROR("send_rpc_cmd: failed to recv out_size for cmd=%d on CTRL channel\n", (int)cmd);
        return false;
    }
    if (out_size != output_size) {
        GGML_LOG_ERROR("send_rpc_cmd: out_size mismatch for cmd=%d (got %" PRIu64 ", expected %zu)\n",
                       (int)cmd, out_size, output_size);
        return false;
    }
    if (!sock->recv_data_channel(channel, output, output_size)) {
        GGML_LOG_ERROR("send_rpc_cmd: failed to recv output payload for cmd=%d (size=%zu)\n", (int)cmd, output_size);
        return false;
    }
    return true;
}

// RPC client-side implementation

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock) {
    rpc_msg_hello_req request = {};
    sock->get_caps(request.conn_caps);
    rpc_msg_hello_rsp response = {};

    if (!send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response))) {
        return false;
    }

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        GGML_LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        return false;
    }

    sock->update_caps(response.conn_caps);
    return true;
}

template <typename T>
class message_queue {
public:
    message_queue() {}

    bool push(const T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        if (interrupted) {
            return false;
        }
        queue.push(value);
        cvar.notify_all();
        return true;
    }

    bool pop(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [this] { return !queue.empty() || interrupted; });
        if (interrupted) {
            return false;
        }
        *out = queue.front();
        queue.pop();
        return true;
    }

    void interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = true;
        lock.unlock();
        cvar.notify_all();
    }

private:
    bool interrupted = false;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

class rpc_dispatcher {
public:
    rpc_dispatcher() {
    }

    void send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size);
    void send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size);
    void send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size,
              int timeout_ms, int max_try);
    void send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size);
    void send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size);
    // Send cmd+header, then tbs_xchg on the dispatcher thread (owns the pipe).
    void send_xchg(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size,
                   const void * xchg_out, void * xchg_in, size_t xchg_n);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    void event_free(ggml_backend_event_t event);
    void event_synchronize(ggml_backend_event_t event);
    void event_record(ggml_backend_event_t event);
    void synchronize();

    void start(const std::string & endpoint);
    void work();

    ~rpc_dispatcher();

private:
    struct rpc_msg {
        rpc_cmd                       cmd;
        std::shared_ptr<const void>   input;
        size_t                        input_size;
        void                        * output;
        size_t                        output_size;
        int                           timeout_ms = -2;
        int                           max_try    = -1;
        const void                  * xchg_out = nullptr;
        void                        * xchg_in  = nullptr;
        size_t                        xchg_n   = 0;
        std::promise<void>            completion;
    };
    using rpc_msg_ptr   = std::shared_ptr<rpc_msg>;
    using rpc_msg_queue = message_queue<rpc_msg_ptr>;
    struct rpc_event {
        rpc_msg_ptr              msg;
        std::shared_future<void> sf;
    };
    rpc_msg_queue    queue;
    socket_ptr       sock;
    std::atomic_bool running;
    std::thread      thread;
};

static void rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

void rpc_dispatcher::send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = nullptr;
    msg->output_size = 0;
    GGML_ASSERT(queue.push(msg));
    auto future = msg->completion.get_future();
    future.wait();
}

void rpc_dispatcher::send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = nullptr;
    msg->output_size = 0;
    GGML_ASSERT(queue.push(msg));
}

void rpc_dispatcher::send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size) {
    send(cmd, input, input_size, output, output_size, -2, -1);
}

void rpc_dispatcher::send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size,
                          int timeout_ms, int max_try) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = output;
    msg->output_size = output_size;
    msg->timeout_ms = timeout_ms;
    msg->max_try = max_try;
    GGML_ASSERT(queue.push(msg));
    auto future = msg->completion.get_future();
    future.wait();
}

void rpc_dispatcher::send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = output;
    msg->output_size = output_size;
    GGML_ASSERT(queue.push(msg));
}

void rpc_dispatcher::send_xchg(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size,
                               const void * xchg_out, void * xchg_in, size_t xchg_n) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = nullptr;
    msg->output_size = 0;
    msg->xchg_out = xchg_out;
    msg->xchg_in = xchg_in;
    msg->xchg_n = xchg_n;
    GGML_ASSERT(queue.push(msg));
    auto future = msg->completion.get_future();
    future.wait();
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = new rpc_event;
    ev->msg = std::make_shared<rpc_msg>();
    ev->msg->cmd = RPC_CMD_NONE;
    ev->sf = ev->msg->completion.get_future().share();
    GGML_ASSERT(queue.push(ev->msg));
    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

void rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    delete ev;
}

void rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

void rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->msg = std::make_shared<rpc_msg>();
    ev->msg->cmd = RPC_CMD_NONE;
    ev->sf = ev->msg->completion.get_future().share();
    GGML_ASSERT(queue.push(ev->msg));
}

void rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.push(msg));
    msg->completion.get_future().wait();
}

void rpc_dispatcher::start(const std::string & endpoint) {
    if (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect_endpoint(endpoint.c_str());
    if (sock == nullptr) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    if (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = true;
    thread = std::thread(rpc_dispatcher_trampoline, this);
}

void rpc_dispatcher::work() {
    while (running) {
        rpc_msg_ptr msg_ptr;
        if (!queue.pop(&msg_ptr)) {
            break;
        }
        if (msg_ptr->cmd != RPC_CMD_NONE) {
            if (msg_ptr->xchg_n > 0) {
                bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size);
                RPC_STATUS_ASSERT(status);
                status = sock->xchg(msg_ptr->xchg_out, msg_ptr->xchg_in, msg_ptr->xchg_n);
                RPC_STATUS_ASSERT(status);
            } else if (msg_ptr->output) {
                bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size, msg_ptr->output, msg_ptr->output_size,
                                           msg_ptr->timeout_ms, msg_ptr->max_try);
                RPC_STATUS_ASSERT(status);
            } else {
                bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size);
                RPC_STATUS_ASSERT(status);
            }
        }
        msg_ptr->completion.set_value();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = false;
    queue.interrupt();
    sock = nullptr;
    if (thread.joinable()) {
        thread.join();
    }
}

static std::shared_ptr<rpc_dispatcher> get_dispatcher(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::shared_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    if (it != dispatchers.end() && it->second != nullptr) {
        return it->second;
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    return dispatcher;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_free_buffer_req>();
    request->remote_ptr = ctx->remote_ptr;
    ctx->dispatcher->send(RPC_CMD_FREE_BUFFER, request, sizeof(*request));
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    auto request = std::make_shared<rpc_msg_buffer_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    rpc_msg_buffer_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_BUFFER_GET_BASE, request, sizeof(*request), &response, sizeof(response));
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor, const std::shared_ptr<rpc_dispatcher> & dispatcher = nullptr) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    ggml_backend_buffer_t buffer = tensor->buffer;
    const ggml_tensor * root = tensor;
    size_t root_offset = 0;
    while (buffer == nullptr && root->view_src != nullptr) {
        root_offset += root->view_offs;
        root = root->view_src;
        buffer = root->buffer;
    }
    if (buffer && ggml_backend_buffer_is_rpc(buffer)) {
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        // Keep view-src offset when tensor->data is null (USB4 / meta views).
        if (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher == dispatcher)) {
            result.buffer = ctx->remote_ptr;
            if (tensor->data) {
                result.data = reinterpret_cast<uint64_t>(tensor->data);
            } else if (root->data) {
                result.data = reinterpret_cast<uint64_t>((const char *)root->data + root_offset);
            } else {
                result.data = 0;
            }
        } else {
            result.buffer = 0;
            result.data = 0;
        }
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    result.use_count = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        auto request = std::make_shared<rpc_msg_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, sizeof(*request));
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, sizeof(*request));
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    int64_t t0 = rpc_perf_enabled() ? ggml_time_us() : 0;
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    if (size > rpc_hash_threshold()) {
        auto request = std::make_shared<rpc_msg_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, sizeof(*request), &response, sizeof(response));
        if (response.result) {
            // the server has the same data, no need to send it
            if (t0) {
                fprintf(stderr, "[RPC_PERF set_tensor hash_match] '%s' size=%zu time=%.3f ms\n",
                        tensor->name, size, (ggml_time_us() - t0) / 1000.0);
            }
            return;
        }
    }
    const uint8_t * in_ptr = static_cast<const uint8_t *>(data);
    size_t transferred = 0;
    while (transferred < size) {
        size_t chunk = std::min(size - transferred, rpc_set_tensor_chunk());
        size_t cur_offset = offset + transferred;
        size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + chunk;
        uint8_t * input = new uint8_t[input_size]();
        memcpy(input, &rpc_tensor, sizeof(rpc_tensor));
        memcpy(input + sizeof(rpc_tensor), &cur_offset, sizeof(cur_offset));
        memcpy(input + sizeof(rpc_tensor) + sizeof(cur_offset), in_ptr + transferred, chunk);
        std::shared_ptr<uint8_t> input_chunk(input, std::default_delete<uint8_t[]>());
        uint8_t ack = 0;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR, input_chunk, input_size, &ack, sizeof(ack));
        transferred += chunk;
    }
    if (t0) {
        fprintf(stderr, "[RPC_PERF set_tensor] '%s' size=%zu time=%.3f ms\n",
                tensor->name, size, (ggml_time_us() - t0) / 1000.0);
    }
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    int64_t t0 = rpc_perf_enabled() ? ggml_time_us() : 0;
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    uint8_t * out = static_cast<uint8_t *>(data);
    size_t transferred = 0;
    if (tb_debug_enabled()) {
        fprintf(stderr, "[RPC_BUF_GET_TENSOR] name=%s size=%zu offset=%zu\n",
                tensor->name, size, offset);
    }
    while (transferred < size) {
        size_t chunk = std::min(size - transferred, (size_t)2097152);
        auto request = std::make_shared<rpc_msg_get_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        request->offset = offset + transferred;
        request->size = chunk;
        if (tb_debug_enabled()) {
            fprintf(stderr, "  [RPC_BUF_GET_TENSOR chunk] transferred=%zu/%zu chunk=%zu\n",
                    transferred, size, chunk);
        }
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, sizeof(*request), out + transferred, chunk);
        transferred += chunk;
    }
    if (tb_debug_enabled()) {
        fprintf(stderr, "[RPC_BUF_GET_TENSOR DONE] name=%s size=%zu\n", tensor->name, size);
    }
    if (t0) {
        fprintf(stderr, "[RPC_PERF get_tensor] '%s' size=%zu time=%.3f ms\n",
                tensor->name, size, (ggml_time_us() - t0) / 1000.0);
    }
}


static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->dispatcher != dst_ctx->dispatcher) {
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        auto request = std::make_shared<rpc_msg_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, sizeof(*request), &response, sizeof(response));
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_buffer_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    ctx->dispatcher->send(RPC_CMD_BUFFER_CLEAR, request, sizeof(*request));
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    auto request = std::make_shared<rpc_msg_alloc_buffer_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_msg_alloc_buffer_rsp response;

    auto dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_BUFFER, request, sizeof(*request), &response, sizeof(response));
    if (response.remote_ptr != 0) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_msg_get_alignment_req>();
    request->device = device;
    rpc_msg_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, sizeof(*request), &response, sizeof(response));
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_msg_get_max_size_req>();
    request->device = device;
    rpc_msg_get_max_size_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_SIZE, request, sizeof(*request), &response, sizeof(response));
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        struct alloc_size_cache_key {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
        };

        alloc_size_cache_key key = {};
        key.device = buft_ctx->device;
        key.type = tensor->type;
        key.op = tensor->op;
        memcpy(key.op_params, tensor->op_params, sizeof(key.op_params));
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            key.ne[i] = (uint32_t)tensor->ne[i];
        }

        uint64_t cache_hash = fnv_hash((const uint8_t *)&key, sizeof(key));
        cache_hash = fnv_hash((const uint8_t *)buft_ctx->endpoint.data(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        static std::mutex cache_mutex;
        static std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            if (it != cache.end()) {
                return it->second;
            }
        }

        auto request = std::make_shared<rpc_msg_get_alloc_size_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        rpc_msg_get_alloc_size_rsp response;
        auto dispatcher = get_dispatcher(buft_ctx->endpoint);
        if (tb_debug_enabled()) fprintf(stderr, "[RPC_CLI get_alloc_size start] tensor '%s' (type=%d)\n", tensor->name, (int)tensor->type);
        dispatcher->send(RPC_CMD_GET_ALLOC_SIZE, request, sizeof(*request), &response, sizeof(response));
        if (tb_debug_enabled()) fprintf(stderr, "[RPC_CLI get_alloc_size done] tensor '%s' alloc_size=%" PRIu64 "\n", tensor->name, response.alloc_size);

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        return response.alloc_size;
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    if (size > rpc_hash_threshold()) {
        auto request = std::make_shared<rpc_msg_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, sizeof(*request), &response, sizeof(response));
        if (response.result) {
            // the server has the same data, no need to send it
            return;
        }
    }
    const uint8_t * in_ptr = static_cast<const uint8_t *>(data);
    size_t transferred = 0;
    while (transferred < size) {
        size_t chunk = std::min(size - transferred, rpc_set_tensor_chunk());
        size_t cur_offset = offset + transferred;
        size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + chunk;
        uint8_t * input = new uint8_t[input_size]();
        memcpy(input, &rpc_tensor, sizeof(rpc_tensor));
        memcpy(input + sizeof(rpc_tensor), &cur_offset, sizeof(cur_offset));
        memcpy(input + sizeof(rpc_tensor) + sizeof(cur_offset), in_ptr + transferred, chunk);
        std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
        uint8_t ack = 0;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR, input_ptr, input_size, &ack, sizeof(ack));
        transferred += chunk;
    }
}


static void ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    auto request = std::make_shared<rpc_msg_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, sizeof(*request), data, size);
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    int64_t t0 = rpc_perf_enabled() ? ggml_time_us() : 0;
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
    if (t0) {
        fprintf(stderr, "[RPC_PERF synchronize] time=%.3f ms\n", (ggml_time_us() - t0) / 1000.0);
    }
}

static void add_tensor(ggml_tensor * tensor, const ggml_cgraph * cgraph, const std::shared_ptr<rpc_dispatcher> & dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    const size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    if (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_count = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

static uint8_t * serialize_graph(uint32_t device, const ggml_cgraph * cgraph, const std::shared_ptr<rpc_dispatcher> & dispatcher, size_t * output_size) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_size = 2*sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    uint8_t * output = new uint8_t[*output_size]();
    uint8_t * dest = output;
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
    return output;
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    int64_t t0 = rpc_perf_enabled() ? ggml_time_us() : 0;
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_dev_t rpc_dev = ggml_backend_get_device(backend);
    ggml_backend_rpc_device_context * rpc_dev_ctx = (ggml_backend_rpc_device_context *)rpc_dev->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    // Wait for a 1-byte ACK so allreduce tbs_xchg cannot start while the
    // remote GPU is still inside GRAPH_COMPUTE on the same tbstripe pipe.
    // USB4STREAM drops frames in a large tbs_send (SET_TENSOR already chunks
    // to 2048). Prefix each GRAPH_COMPUTE RPC with | total (8) | offset (8) |
    // data | so every command fits in one 4KiB frame.
    uint8_t ack = 0;
    bool reuse = cgraph->uid != 0 && rpc_dev_ctx->last_graph_uid == cgraph->uid;
    if (reuse) {
        auto request = std::make_shared<rpc_msg_graph_recompute_req>();
        request->device = rpc_ctx->device;
        rpc_ctx->dispatcher->send(RPC_CMD_GRAPH_RECOMPUTE, request, sizeof(*request), &ack, sizeof(ack));
    } else {
        rpc_dev_ctx->last_graph_uid = cgraph->uid;
        size_t input_size = 0;
        uint8_t * input = serialize_graph(rpc_ctx->device, cgraph, rpc_ctx->dispatcher, &input_size);
        const size_t chunk = rpc_set_tensor_chunk();
        size_t transferred = 0;
        size_t nchunk = 0;
        while (transferred < input_size) {
            const size_t n = std::min(input_size - transferred, chunk);
            const bool last = transferred + n >= input_size;
            const size_t msg_size = 2 * sizeof(uint64_t) + n;
            uint8_t * msg = new uint8_t[msg_size]();
            const uint64_t total = input_size;
            const uint64_t offset = transferred;
            memcpy(msg, &total, sizeof(total));
            memcpy(msg + sizeof(total), &offset, sizeof(offset));
            memcpy(msg + 2 * sizeof(uint64_t), input + transferred, n);
            std::shared_ptr<uint8_t> msg_ptr(msg, std::default_delete<uint8_t[]>());
            ack = 0;
            // Intermediate chunks are SET_TENSOR-sized USB4 frames: 100ms ACK
            // retry. The last chunk runs HIP graph_compute before ACK — wait.
            const int timeout_ms = last ? -1 : 100;
            const int max_try    = last ? 1  : 64;
            rpc_ctx->dispatcher->send(RPC_CMD_GRAPH_COMPUTE, msg_ptr, msg_size, &ack, sizeof(ack),
                                      timeout_ms, max_try);
            if (ack != 1) {
                delete[] input;
                GGML_LOG_ERROR("rpc graph_compute: bad ack %u chunk off=%zu/%zu last=%d\n",
                               (unsigned) ack, transferred, input_size, (int) last);
                return GGML_STATUS_FAILED;
            }
            transferred += n;
            nchunk++;
            if (!last && (nchunk % 32u) == 0u) {
                fprintf(stderr, "[rpc graph_compute] chunks %zu off=%zu/%zu\n",
                        nchunk, transferred, input_size);
            }
        }
        delete[] input;
        if (t0) {
            fprintf(stderr, "[RPC_PERF graph_compute] n_nodes=%d bytes=%zu chunks=%zu\n",
                    cgraph->n_nodes, input_size, (input_size + chunk - 1) / chunk);
        }
    }
    if (ack != 1) {
        GGML_LOG_ERROR("rpc graph_compute: bad ack %u reuse=%d\n", (unsigned) ack, (int) reuse);
        return GGML_STATUS_FAILED;
    }
    if (t0) {
        fprintf(stderr, "[RPC_PERF graph_compute] n_nodes=%d reuse=%d time=%.3f ms\n",
                cgraph->n_nodes, (int)reuse, (ggml_time_us() - t0) / 1000.0);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

static void ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_size(dispatcher, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto dispatcher = get_dispatcher(endpoint);
    auto request = std::make_shared<rpc_msg_get_device_memory_req>();
    request->device = device;
    rpc_msg_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, sizeof(*request), &response, sizeof(response));
    *free = response.free_mem;
    *total = response.total_mem;
}

// RPC server-side implementation

class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool memset_tensor(const rpc_msg_memset_tensor_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_compute_chunk(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

private:
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    std::vector<ggml_backend_t> backends;
    const char * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // store the last computed graph for each backend
    std::vector<stored_graph> stored_graphs;
    // USB4STREAM GRAPH_COMPUTE is split into SET_TENSOR-sized chunks so each
    // tbs_send is one 4KiB frame. Accumulate here until total_size is filled.
    std::vector<uint8_t> gc_pending;
    uint64_t             gc_total  = 0;
    uint64_t             gc_filled = 0;
};

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG("[%s] device: %d, buffer: %p, data: %p\n", __func__, dev_id, (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);
    if (tb_debug_enabled()) fprintf(stderr, "[RPC_SRV get_alloc_size] tensor '%s' dev=%u alloc_size=%" PRIu64 "\n", tensor->name, dev_id, response.alloc_size);

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n",
            __func__, dev_id, request.size, response.remote_ptr, response.remote_size);
        buffers.insert(buffer);
    } else {
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> failed\n", __func__, dev_id, request.size);
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG("[%s] device: %d, alignment: %lu\n", __func__, dev_id, alignment);
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG("[%s] device: %d, max_size: %lu\n", __func__, dev_id, max_size);
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

bool rpc_server::memset_tensor(const rpc_msg_memset_tensor_req & request) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    const uint64_t tensor_size = ggml_nbytes(tensor);
    if (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        GGML_LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        return false;
    }

    const uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    const uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    if (request.tensor.data < buffer_start) {
        GGML_LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        return false;
    }
    const uint64_t data_offset = request.tensor.data - buffer_start;
    if (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        GGML_LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        return false;
    }
    if (tensor->buffer->iface.memset_tensor == nullptr) {
        GGML_LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        return false;
    }

    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 ", value: %u\n",
            __func__, (void *) tensor->buffer, tensor->data, request.offset, request.size, request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        GGML_LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        GGML_LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        const bool overflow = tensor->data + tensor_size < tensor->data;
        const bool oob = tensor->data < buffer_start ||
                tensor->data + tensor_size > buffer_start + buffer_size;
        if (overflow || oob) {
            GGML_LOG_ERROR("[%s] tensor '%s' op=%s type=%d data=%#" PRIx64 " nbytes=%" PRIu64
                           " buf=[%#" PRIx64 " + %" PRIu64 "] ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]"
                           " nb=[%zu,%zu,%zu,%zu]\n",
                    __func__, tensor->name, ggml_op_name((enum ggml_op) tensor->op), (int) tensor->type,
                    (uint64_t) tensor->data, tensor_size, buffer_start, buffer_size,
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    (size_t) tensor->nb[0], (size_t) tensor->nb[1], (size_t) tensor->nb[2], (size_t) tensor->nb[3]);
            return nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);
    //

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            return false;
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    ggml_backend_tensor_set(tensor, data, offset, size);
    return true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    if (!cache_dir) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    if (!fs::exists(cache_file, ec)) {
        return false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char *)data.data(), size);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        response.result = 0;
        return true;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    long us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    printf("DEBUG: get_cached_file took %ld us\n", us); fflush(stdout);
    size_t size = cached_file.size();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        response.result = 0;
        return true;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu, hash: %" PRIx64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, request.offset, size, request.hash);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            response.result = 0;
            return true;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p\n", __func__, (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            GGML_LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }

    LOG_DBG("[%s] src->buffer: %p, dst->buffer: %p\n",
            __func__, (void*) src->buffer, (void*) dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    if (result->buffer == nullptr && result->data != nullptr && tensor->view_src == 0) {
        GGML_LOG_ERROR("[%s] invalid data ptr", __func__);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } else {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                GGML_LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    } else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            GGML_LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
        if (result->buffer == nullptr && result->view_src->buffer != nullptr) {
            result->buffer = result->view_src->buffer;
        }
        result->view_offs = tensor->view_offs;
        // Only HC mix stream views: [n_embd, T] of [n_embd, hc, T] with
        // parent nb[1] == n_embd*esz. Rebasing every VIEW also rewrote QSA
        // [64,4] of [64,64,4] to nb[1]=parent.nb[2] and double-offset ROPE
        // srcs; worker then HSA-faulted in rope_multi (tiny STREAM hang).
        const ggml_tensor * p = result->view_src;
        const size_t esz = ggml_type_size(result->type);
        const bool hc_stream =
            p->ne[1] > 1 && p->ne[1] <= 16 &&
            result->ne[0] == p->ne[0] && result->ne[1] == p->ne[2] &&
            result->nb[1] == p->nb[2] &&
            p->nb[1] == (size_t) result->ne[0] * esz &&
            p->nb[1] > 0 && result->view_offs % p->nb[1] == 0;
        if (hc_stream && p->data != nullptr) {
            result->data = (char *) p->data + result->view_offs;
            result->nb[0] = p->nb[0];
            result->nb[1] = p->nb[2];
            result->nb[2] = result->nb[1] * (size_t) result->ne[1];
            result->nb[3] = result->nb[2];
        }
    }
    if (tensor->view_src == 0) {
        result->view_offs = tensor->view_offs;
    }
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < 2*sizeof(uint32_t)) {
        return false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, sizeof(device));
    src += sizeof(device);
    if (device >= backends.size()) {
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, n_nodes: %u, n_tensors: %u\n", __func__, device, n_nodes, n_tensors);

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    if (stored_graphs[device].buffer.size() < buf_size) {
        stored_graphs[device].buffer.resize(buf_size);
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ stored_graphs[device].buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);

        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            GGML_LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
        if (graph->nodes[i] != nullptr) {
            const size_t hash_pos = ggml_hash_insert(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_count;
        }
    }
    {
        static int ngraphs;
        if (ngraphs < 30) {
            ngraphs++;
            int nadd = 0, nviewsrc = 0, nstrided = 0;
            for (uint32_t i = 0; i < n_nodes; i++) {
                ggml_tensor * node = graph->nodes[i];
                if (!node || node->op != GGML_OP_ADD) {
                    continue;
                }
                nadd++;
                for (int s = 0; s < 2; s++) {
                    ggml_tensor * src = node->src[s];
                    if (!src) {
                        continue;
                    }
                    if (src->op == GGML_OP_VIEW) {
                        nviewsrc++;
                    }
                    const bool strided = src->nb[1] != (size_t) src->ne[0] * ggml_type_size(src->type)
                            && src->ne[1] > 1;
                    if ((strided || nadd <= 3) && nstrided < 8) {
                        fprintf(stderr, "[RPC_ADD] g=%d i=%u src%d op=%s ne=[%lld,%lld,%lld] nb=[%zu,%zu,%zu] offs=%zu cont=%d\n",
                                ngraphs, i, s, ggml_op_name(src->op),
                                (long long) src->ne[0], (long long) src->ne[1], (long long) src->ne[2],
                                src->nb[0], src->nb[1], src->nb[2],
                                (size_t) src->view_offs, (int) ggml_is_contiguous(src));
                    }
                    nstrided += (int) strided;
                }
            }
            fprintf(stderr, "[RPC_GC] n_nodes=%u n_tensors=%u nadd=%d nviewsrc=%d nstrided_add_src=%d\n",
                    n_nodes, n_tensors, nadd, nviewsrc, nstrided);
            int nrope = 0;
            for (uint32_t i = 0; i < n_nodes && nrope < 4; i++) {
                ggml_tensor * node = graph->nodes[i];
                if (!node || node->op != GGML_OP_ROPE) {
                    continue;
                }
                nrope++;
                const ggml_tensor * s0 = node->src[0];
                const ggml_tensor * s1 = node->src[1];
                fprintf(stderr, "[RPC_ROPE] g=%d i=%u dst ne=[%lld,%lld,%lld] nb=[%zu,%zu] data=%p src0=%s ne=[%lld,%lld,%lld] nb=[%zu,%zu] data=%p src1=%s ne=[%lld,%lld] data=%p\n",
                        ngraphs, i,
                        (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2],
                        node->nb[0], node->nb[1], node->data,
                        s0 ? ggml_op_name(s0->op) : "nil",
                        s0 ? (long long) s0->ne[0] : -1, s0 ? (long long) s0->ne[1] : -1, s0 ? (long long) s0->ne[2] : -1,
                        s0 ? s0->nb[0] : 0, s0 ? s0->nb[1] : 0, s0 ? s0->data : nullptr,
                        s1 ? ggml_op_name(s1->op) : "nil",
                        s1 ? (long long) s1->ne[0] : -1, s1 ? (long long) s1->ne[1] : -1,
                        s1 ? s1->data : nullptr);
            }
        }
    }
    if (tb_debug_enabled()) {
        for (uint32_t i = 0; i < n_nodes; i++) {
            ggml_tensor * node = graph->nodes[i];
            if (!node) continue;
            fprintf(stderr, "[RPC-GRAPH] node[%u]: name=%s op=%d data=%p buffer=%p\n", i, node->name, node->op, node->data, (void*)node->buffer);
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                if (node->src[s]) {
                    fprintf(stderr, "   src[%d]: name=%s data=%p buffer=%p\n", s, node->src[s]->name, node->src[s]->data, (void*)node->src[s]->buffer);
                }
            }
        }
    }
    extern void hrx_shim_set_debug(int) __attribute__((weak));
    if (hrx_shim_set_debug) hrx_shim_set_debug(1);
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[RPC-SRV graph_compute FAILED] status=%d, n_nodes=%u\n", (int)status, n_nodes);
        for (uint32_t i = 0; i < n_nodes; i++) {
            ggml_tensor * node = graph->nodes[i];
            if (!node) continue;
            fprintf(stderr, "  node[%u]: '%s' op=%d data=%p buffer=%p view_src=%p\n",
                    i, node->name, (int)node->op, node->data, (void*)node->buffer, (void*)node->view_src);
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                if (node->src[s]) {
                    fprintf(stderr, "    src[%d]: '%s' op=%d data=%p buffer=%p view_src=%p\n",
                            s, node->src[s]->name, (int)node->src[s]->op, node->src[s]->data, (void*)node->src[s]->buffer, (void*)node->src[s]->view_src);
                }
            }
        }
    }
    if (status != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("[%s] graph_compute failed status=%d n_nodes=%u (keeping RPC session)\n",
                       __func__, (int) status, n_nodes);
        return false;
    }
    stored_graphs[device].graph = graph;
    return true;
}

bool rpc_server::graph_compute_chunk(const std::vector<uint8_t> & input) {
    // | total_size (8) | offset (8) | data |
    if (input.size() < 2 * sizeof(uint64_t)) {
        return false;
    }
    uint64_t total = 0;
    uint64_t offset = 0;
    memcpy(&total, input.data(), sizeof(total));
    memcpy(&offset, input.data() + sizeof(total), sizeof(offset));
    const size_t n = input.size() - 2 * sizeof(uint64_t);
    const uint8_t * data = input.data() + 2 * sizeof(uint64_t);

    constexpr uint64_t GC_MAX = 64ull * 1024ull * 1024ull;
    if (total == 0 || total > GC_MAX || n == 0) {
        GGML_LOG_ERROR("[%s] invalid GRAPH_COMPUTE chunk total=%" PRIu64 " n=%zu\n",
                       __func__, total, n);
        gc_pending.clear();
        gc_total = 0;
        gc_filled = 0;
        return false;
    }
    if (offset == 0) {
        gc_pending.resize(total);
        gc_total = total;
        gc_filled = 0;
    }
    if (gc_total != total || offset + n > gc_total) {
        GGML_LOG_ERROR("[%s] GRAPH_COMPUTE chunk mismatch expect_total=%" PRIu64
                       " got_total=%" PRIu64 " offset=%" PRIu64 " filled=%" PRIu64 " n=%zu\n",
                       __func__, gc_total, total, offset, gc_filled, n);
        gc_pending.clear();
        gc_total = 0;
        gc_filled = 0;
        return false;
    }
    // Retransmit of an already-applied chunk (USB4 ACK drop): rewrite in place.
    if (offset + n <= gc_filled) {
        memcpy(gc_pending.data() + offset, data, n);
        if (gc_filled < gc_total) {
            return true;
        }
    } else if (offset == gc_filled) {
        memcpy(gc_pending.data() + offset, data, n);
        gc_filled += n;
        if (gc_filled < gc_total) {
            return true;
        }
    } else {
        GGML_LOG_ERROR("[%s] GRAPH_COMPUTE chunk gap offset=%" PRIu64 " filled=%" PRIu64 " n=%zu\n",
                       __func__, offset, gc_filled, n);
        gc_pending.clear();
        gc_total = 0;
        gc_filled = 0;
        return false;
    }
    const bool ok = graph_compute(gc_pending);
    gc_pending.clear();
    gc_total = 0;
    gc_filled = 0;
    return ok;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    if (stored_graphs[device].graph == nullptr) {
        return false;
    }
    ggml_cgraph * graph = stored_graphs[device].graph;
    LOG_DBG("[%s] device: %u\n", __func__, device);
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    if (status != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("[%s] graph_recompute failed status=%d (keeping RPC session)\n",
                       __func__, (int) status);
        return false;
    }
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        response.free_mem = 0;
        response.total_mem = 0;
        return true;
    }
    size_t free = 0, total = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    if (dev) {
        ggml_backend_dev_memory(dev, &free, &total);
    }
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG("[%s] device: %u, free_mem: %" PRIu64 ", total_mem: %" PRIu64 "\n", __func__, dev_id, response.free_mem, response.total_mem);
    return true;
}

rpc_server::~rpc_server() {
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

static bool handle_hello_handshake(rpc_server & server, socket_ptr sock) {
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return false;
    }
    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return false;
    }
    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return false;
    }
    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return false;
    }
    sock->update_caps(req.conn_caps);
    return true;
}

static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    uint32_t ch = RPC_CHANNEL_CONTROL;
    if (!sock->recv_cmd(&cmd, &ch)) {
        return;
    }
    while (cmd != RPC_CMD_HELLO) {
        GGML_LOG_WARN("Ignoring stale cmd %d before HELLO handshake\n", (int)cmd);
        if (!sock->recv_cmd(&cmd, &ch)) {
            return;
        }
    }

    if (!handle_hello_handshake(server, sock)) {
        return;
    }
    while (true) {
        uint32_t active_ch = RPC_CHANNEL_CONTROL;
        if (!sock->recv_cmd(&cmd, &active_ch)) {
            fprintf(stderr, "[RPC_SRV] recv_cmd returned false! Client disconnected or read error.\n");
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        //
        uint32_t cmd_channel = rpc_cmd_to_channel((enum rpc_cmd)cmd);
        if (tb_debug_enabled()) fprintf(stderr, "[RPC_SRV recv cmd=%d]\n", (int)cmd);
        switch (cmd) {
            case RPC_CMD_HELLO: {
                if (!handle_hello_handshake(server, sock)) {
                    return;
                }
                break;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0, cmd_channel)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = backends.size();
                if (!send_msg(sock, &response, sizeof(response), cmd_channel)) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response), cmd_channel)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    fprintf(stderr, "[RPC_SRV] GET_ALLOC_SIZE recv_msg failed!\n");
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    fprintf(stderr, "[RPC_SRV] GET_ALLOC_SIZE server.get_alloc_size failed!\n");
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response), cmd_channel)) {
                    fprintf(stderr, "[RPC_SRV] GET_ALLOC_SIZE send_msg failed!\n");
                    return;
                }
                if (tb_debug_enabled()) fprintf(stderr, "[RPC_SRV] GET_ALLOC_SIZE response sent successfully\n");
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response), cmd_channel)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response), cmd_channel)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response), cmd_channel)) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_MEMSET_TENSOR: {
                rpc_msg_memset_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    return;
                }
                if (!server.memset_tensor(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input, cmd_channel)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                uint8_t ack = 1;
                if (!send_msg(sock, &ack, sizeof(ack), cmd_channel)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response), cmd_channel)) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    if (tb_debug_enabled()) fprintf(stderr, "[RPC_SRV] GET_TENSOR recv_msg failed!\n");
                    return;
                }
                if (tb_debug_enabled()) {
                    fprintf(stderr, "[RPC_SRV] GET_TENSOR req offset=%" PRIu64 " size=%" PRIu64 "\n",
                            request.offset, request.size);
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    if (tb_debug_enabled()) fprintf(stderr, "[RPC_SRV] GET_TENSOR server.get_tensor failed!\n");
                    return;
                }
                if (tb_debug_enabled()) {
                    fprintf(stderr, "[RPC_SRV] GET_TENSOR sending resp size=%zu\n", response.size());
                }
                if (!sock->send_data_channel(cmd_channel, response.data(), response.size())) {
                    if (tb_debug_enabled()) fprintf(stderr, "[RPC_SRV] GET_TENSOR send_data_channel failed!\n");
                    return;
                }
                sock->flush();
                if (tb_debug_enabled()) {
                    fprintf(stderr, "[RPC_SRV] GET_TENSOR sent response successfully!\n");
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response), cmd_channel)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input, cmd_channel)) {
                    return;
                }
                const uint8_t ack = server.graph_compute_chunk(input) ? 1 : 0;
                if (!send_msg(sock, &ack, sizeof(ack), cmd_channel)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    return;
                }
                const uint8_t ack = server.graph_recompute(request) ? 1 : 0;
                if (!send_msg(sock, &ack, sizeof(ack), cmd_channel)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response), cmd_channel)) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALL_REDUCE: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request), cmd_channel)) {
                    return;
                }
                size_t bytes = request.size;
                if (bytes == 0) {
                    return;
                }
                std::vector<uint8_t> local_data;
                if (!server.get_tensor(request, local_data)) {
                    return;
                }
                if (local_data.size() != bytes) {
                    GGML_LOG_ERROR("RPC_CMD_ALL_REDUCE size mismatch: tensor=%zu header=%zu\n",
                                   local_data.size(), bytes);
                    return;
                }
                std::vector<uint8_t> peer_data(bytes);
                if (!sock->xchg(local_data.data(), peer_data.data(), bytes)) {
                    GGML_LOG_ERROR("RPC_CMD_ALL_REDUCE tbs_xchg failed (bytes=%zu)\n", bytes);
                    return;
                }
                if (request.tensor.type == GGML_TYPE_F32) {
                    allreduce_add_f32((float *)local_data.data(),
                                      (const float *)peer_data.data(),
                                      bytes / sizeof(float));
                } else if (request.tensor.type == GGML_TYPE_F16) {
                    allreduce_add_f16((ggml_fp16_t *)local_data.data(),
                                      (const ggml_fp16_t *)peer_data.data(),
                                      bytes / sizeof(ggml_fp16_t));
                } else {
                    GGML_LOG_ERROR("Unsupported tensor type for RPC_CMD_ALL_REDUCE: %d\n", request.tensor.type);
                    return;
                }
                std::vector<uint8_t> set_input(sizeof(rpc_tensor) + sizeof(uint64_t) + bytes);
                memcpy(set_input.data(), &request.tensor, sizeof(rpc_tensor));
                uint64_t offset = request.offset;
                memcpy(set_input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
                memcpy(set_input.data() + sizeof(rpc_tensor) + sizeof(offset), local_data.data(), bytes);
                if (!server.set_tensor(set_input)) {
                    return;
                }
                break;
            }
            default: {
                GGML_LOG_ERROR("Unknown command: %d\n", cmd);
                return;
            }
        }
    }
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    rpc_endpoint_info ep_info;
    if (!socket_t::parse_endpoint(endpoint, ep_info)) {
        fprintf(stderr, "Failed to parse endpoint: %s\n", endpoint);
        return;
    }

    if (ep_info.kind == rpc_transport_kind::STREAM) {
        printf("  transport      : USB4STREAM (Zero-Network-Stack Dual-Stream Character Devices)\n");
        printf("  data stream    : %s\n", ep_info.data_path.c_str());
        printf("  control stream : %s\n", ep_info.ctrl_path.c_str());
    } else {
#ifdef GGML_RPC_RDMA
        printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
        printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    }

    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server_endpoint(endpoint);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket for endpoint %s\n", endpoint);
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        rpc_serve_client(backends, cache_dir, client_socket);
        printf("Client connection closed\n");
        fflush(stdout);
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    return true;
}


static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint;
}

static ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    return dispatcher->event_new(dev);
}

static void ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

static void ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static void init_rpc_devices(ggml_backend_rpc_reg_context * ctx);

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx && reg == ggml_backend_rpc_reg()) init_rpc_devices(ctx);
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx && reg == ggml_backend_rpc_reg()) init_rpc_devices(ctx);
    if (ctx == nullptr || index >= ctx->devices.size()) {
        return nullptr;
    }
    return ctx->devices[index];
}


struct rpc_comm_context {
    std::vector<ggml_backend_t> backends;
};

GGML_BACKEND_API void * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    rpc_comm_context * ctx = new rpc_comm_context;
    for (size_t i = 0; i < n_backends; i++) {
        ctx->backends.push_back(backends[i]);
    }
    return ctx;
}

GGML_BACKEND_API void ggml_backend_rpc_comm_free(void * comm_ctx) {
    rpc_comm_context * ctx = (rpc_comm_context *)comm_ctx;
    delete ctx;
}

GGML_BACKEND_API bool ggml_backend_rpc_comm_allreduce_tensor(void * comm_ctx, struct ggml_tensor ** tensors) {
    rpc_comm_context * ctx = (rpc_comm_context *)comm_ctx;
    size_t n_backends = ctx->backends.size();

    ggml_backend_t local_backend = nullptr;
    ggml_backend_t rpc_backend = nullptr;
    ggml_tensor * local_tensor = nullptr;
    ggml_tensor * rpc_tensor = nullptr;

    for (size_t i = 0; i < n_backends; i++) {
        if (ggml_backend_is_rpc(ctx->backends[i])) {
            rpc_backend = ctx->backends[i];
            rpc_tensor = tensors[i];
        } else {
            local_backend = ctx->backends[i];
            local_tensor = tensors[i];
        }
    }

    if (!local_backend || !rpc_backend || !local_tensor || !rpc_tensor) {
        return false;
    }

    const size_t bytes = ggml_nbytes(local_tensor);
    if (bytes == 0 || ggml_nbytes(rpc_tensor) != bytes) {
        GGML_LOG_ERROR("rpc allreduce: size mismatch local=%zu rpc=%zu\n",
                       bytes, ggml_nbytes(rpc_tensor));
        return false;
    }

    std::vector<uint8_t> local_data(bytes);
    std::vector<uint8_t> peer_data(bytes);
    ggml_backend_tensor_get(local_tensor, local_data.data(), 0, bytes);

    auto payload = std::make_shared<rpc_msg_get_tensor_req>();
    payload->tensor = serialize_tensor(rpc_tensor);
    payload->offset = 0;
    payload->size = bytes;

    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)rpc_backend->context;
    rpc_ctx->dispatcher->send_xchg(RPC_CMD_ALL_REDUCE, payload, sizeof(*payload),
                                   local_data.data(), peer_data.data(), bytes);

    if (local_tensor->type == GGML_TYPE_F32) {
        allreduce_add_f32((float *)local_data.data(),
                          (const float *)peer_data.data(),
                          bytes / sizeof(float));
    } else if (local_tensor->type == GGML_TYPE_F16) {
        allreduce_add_f16((ggml_fp16_t *)local_data.data(),
                          (const ggml_fp16_t *)peer_data.data(),
                          bytes / sizeof(ggml_fp16_t));
    } else {
        GGML_LOG_ERROR("rpc allreduce: unsupported type %d\n", (int)local_tensor->type);
        return false;
    }

    ggml_backend_tensor_set(local_tensor, local_data.data(), 0, bytes);
    // Remote already added and set its own tensor inside RPC_CMD_ALL_REDUCE.
    // Do not SET the sum back over RPC.

    static std::atomic<int> n_logged{0};
    if (n_logged.fetch_add(1) < 3) {
        GGML_LOG_INFO("rpc allreduce: tbs_xchg %zu bytes type=%d (no extra SET)\n",
                      bytes, (int)local_tensor->type);
    }

    return true;
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_rpc_comm_init;
    }
    if (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_rpc_comm_free;
    }
    if (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    rpc_msg_device_count_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    return response.device_count;
}

static void init_rpc_devices(ggml_backend_rpc_reg_context * ctx) {
    if (!ctx) return;
    static std::once_flag init_flag;
    std::call_once(init_flag, [ctx]() {
        const char * servers = getenv("GGML_RPC_SERVERS");
        const char * stream  = getenv("GGML_RPC_STREAM");
        std::vector<std::string> endpoints;
        if (stream && stream[0] != '\0') {
            endpoints.push_back(std::string("dev://") + stream);
        }
        if (servers && servers[0] != '\0') {
            std::stringstream ss(servers);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    endpoints.push_back(item);
                }
            }
        }
        uint32_t dev_id = 0;
        for (const auto & ep : endpoints) {
            uint32_t dev_count = ggml_backend_rpc_get_device_count(ep.c_str());
            for (uint32_t ind = 0; ind < dev_count; ind++) {
                std::string dev_name = "RPC" + std::to_string(dev_id);
                std::string dev_desc = ep;
                ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
                    /* .endpoint    = */    ep,
                    /* .device      = */    ind,
                    /* .name        = */    dev_name,
                    /* .description = */    dev_desc,
                    /* .last_graph_uid = */ 0,
                };

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_rpc_device_i,
                    /* .reg     = */ ggml_backend_rpc_reg(),
                    /* .context = */ dev_ctx,
                };
                ctx->devices.push_back(dev);
                dev_id++;
            }
        }
    });
}

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_rpc_reg_context ctx;
    ctx.name = "RPC";
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ &ctx,
    };
    return &ggml_backend_rpc_reg;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (reg_map.find(endpoint) != reg_map.end()) {
        return reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    if (dev_count == 0) {
        return nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
            /* .last_graph_uid = */ 0,
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    return reg;
}


extern "C" ggml_backend_reg_t ggml_backend_init(void) {
    return ggml_backend_rpc_reg();
}


