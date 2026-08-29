#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

// USB4STREAM transport framing constants
static constexpr uint32_t USB4STREAM_MAGIC       = 0x55534234u; // "USB4" in ASCII
static constexpr size_t   USB4STREAM_CHUNK_SIZE  = 256 * 1024;  // 256 KiB DMA chunk size
static constexpr size_t   USB4STREAM_MAX_MSG     = 1024ull * 1024ull * 1024ull; // 1 GiB

struct usb4stream_frame_hdr {
    uint32_t magic;    // USB4STREAM_MAGIC
    uint32_t length;   // Length of payload following header
    uint32_t flags;    // Transport flags (0x1 = handshake, 0x2 = flush, etc.)
    uint32_t sequence; // Monotonic frame sequence number
};

struct usb4stream_transport {
    static bool is_supported();
    static bool is_usb4_path(const char * path);
    static std::string normalize_device_path(const char * path);

    static std::unique_ptr<usb4stream_transport> create_server(const char * dev_path);
    static std::unique_ptr<usb4stream_transport> connect(const char * dev_path);

    // Creates an in-memory / loopback pair for cross-platform testing and emulation
    static std::pair<std::unique_ptr<usb4stream_transport>, std::unique_ptr<usb4stream_transport>> create_loopback_pair();

    virtual ~usb4stream_transport() = default;

    virtual bool send_data(const void * data, size_t size) = 0;
    virtual bool recv_data(void * data, size_t size) = 0;
    virtual bool flush() = 0;
    virtual bool is_connected() const = 0;
    virtual const std::string & get_device_path() const = 0;
    virtual void close() = 0;
};
