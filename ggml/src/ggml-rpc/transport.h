#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

// USB4STREAM Framing Constants & Structures
static constexpr uint32_t STREAM_FRAME_MAGIC = 0x55534234; // 'U','S','B','4'

#pragma pack(push, 1)
struct stream_frame_header {
    uint32_t magic;        // 0x55534234 ('USB4')
    uint32_t channel_id;   // 0 = Data Plane (/dev/tbstream0), 1 = Control Plane (/dev/tbstream1)
    uint64_t payload_len;  // Length of the following payload in bytes
};
#pragma pack(pop)

static_assert(sizeof(stream_frame_header) == 16, "stream_frame_header must be 16 bytes");

enum rpc_channel_id : uint32_t {
    RPC_CHANNEL_DATA    = 0, // Bulk tensor memory / weights (/dev/tbstream0)
    RPC_CHANNEL_CONTROL = 1, // Control opcodes, sync barriers, graph requests (/dev/tbstream1)
    RPC_CHANNEL_COUNT   = 2,
};

enum class rpc_transport_kind {
    TCP,
    STREAM,
};

struct rpc_endpoint_info {
    rpc_transport_kind kind = rpc_transport_kind::TCP;
    std::string host = "127.0.0.1";
    int port = 50052;
    std::string data_path = "/dev/tbstream0";
    std::string ctrl_path = "/dev/tbstream1";
    std::string raw_endpoint;
};

// Abstract transport interface
struct rpc_transport {
    virtual ~rpc_transport() = default;

    virtual bool send_exact(const void * data, size_t size) = 0;
    virtual bool recv_exact(void * data, size_t size) = 0;

    virtual bool send_exact_channel(uint32_t channel_id, const void * data, size_t size) {
        (void)channel_id;
        return send_exact(data, size);
    }

    virtual bool recv_exact_channel(uint32_t channel_id, void * data, size_t size) {
        (void)channel_id;
        return recv_exact(data, size);
    }

    virtual bool recv_cmd(uint8_t * cmd, uint32_t * out_channel = nullptr) {
        if (out_channel) *out_channel = RPC_CHANNEL_CONTROL;
        return recv_exact(cmd, 1);
    }

    virtual bool flush() { return true; }
    virtual void close() = 0;

    virtual std::shared_ptr<rpc_transport> accept() { return nullptr; }

    virtual void get_caps(uint8_t * local_caps) { memset(local_caps, 0, RPC_CONN_CAPS_SIZE); }
    virtual void update_caps(const uint8_t * remote_caps) { (void)remote_caps; }

    virtual bool is_stream() const { return false; }
};

using rpc_transport_ptr = std::shared_ptr<rpc_transport>;

// Socket / Transport Wrapper providing uniform API for TCP and USB4 streaming character devices
struct socket_t {
    explicit socket_t(rpc_transport_ptr transport);
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);

    bool send_data_channel(uint32_t channel_id, const void * data, size_t size);
    bool recv_data_channel(uint32_t channel_id, void * data, size_t size);

    bool recv_cmd(uint8_t * cmd, uint32_t * out_channel = nullptr);

    bool flush();

    std::shared_ptr<socket_t> accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    bool is_stream() const;
    rpc_transport_ptr get_transport() const { return transport; }

    static std::shared_ptr<socket_t> create_server(const char * host, int port);
    static std::shared_ptr<socket_t> connect(const char * host, int port);

    static std::shared_ptr<socket_t> create_server_endpoint(const char * endpoint);
    static std::shared_ptr<socket_t> connect_endpoint(const char * endpoint);

    static bool parse_endpoint(const std::string & endpoint, rpc_endpoint_info & out);

private:
    rpc_transport_ptr transport;
};

typedef std::shared_ptr<socket_t> socket_ptr;

bool rpc_transport_init();
void rpc_transport_shutdown();
