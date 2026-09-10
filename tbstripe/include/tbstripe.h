#ifndef TBSTRIPE_H
#define TBSTRIPE_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Dual-USB4STREAM Striped Pipe for Strix Halo (Linux 7.3)
 * Provides 2-node tensor-parallel All-Reduce and tensor exchange over dual Thunderbolt character devices.
 */

typedef struct tbs_pipe tbs_pipe;

typedef struct tbs_config {
    const char *dev_a;      /* Primary device path, e.g. /dev/tbstream0 */
    const char *dev_b;      /* Secondary device path, e.g. /dev/tbstream1 (NULL => single stream fallback) */
    size_t stripe;          /* Stripe chunk size in bytes (default 2048) */
    int busy_spin;          /* 1 to use non-blocking spin on Linux 7.3 (required with busy_poll=1) */
    int cpu_affinity_a;     /* CPU core affinity for Device A worker/poller (-1 for none) */
    int cpu_affinity_b;     /* CPU core affinity for Device B worker/poller (-1 for none) */
    /*
     * IMP-10 simplex is env-gated so this struct stays ABI-compatible with
     * already-running libggml-rpc (do not add runtime fields here).
     *
     * TBSTRIPE_SIMPLEX=master  this node TX on dev_a, RX on dev_b (bosgame1)
     * TBSTRIPE_SIMPLEX=worker  this node RX on dev_a, TX on dev_b (bosgame2)
     * unset / 0 / duplex       current bidirectional striped pipe
     */
} tbs_config;

/* Framing: one USB4STREAM DATA packet is TB_MAX_FRAME_SIZE (4096).
 * Each tbstripe chunk is exactly one 4096-byte packet so the kernel never
 * coalesces or splits a logical chunk. Even chunk_idx -> dev_a, odd -> dev_b.
 */
#define TBS_MAGIC_VAL           0x5442535452495045ULL /* ASCII 'TBSTRIPE' */
#define TBS_FRAME_SIZE          4096
#define TBS_FRAME_HDR_SIZE      32
#define TBS_FRAME_PAYLOAD       (TBS_FRAME_SIZE - TBS_FRAME_HDR_SIZE) /* 4064 */
#define TBS_DEFAULT_STRIPE_SZ   TBS_FRAME_SIZE
#define TBS_ALIGNMENT           4096
#define TBS_FRAME_BATCH         1

enum tbs_msg_type {
    TBS_MSG_DATA      = 0,
    TBS_MSG_BARRIER   = 1,
    TBS_MSG_ALLREDUCE = 2,
    TBS_MSG_HEARTBEAT = 3,
};

enum tbs_msg_flags {
    TBS_FLAG_NONE     = 0,
    TBS_FLAG_STRIPED  = (1 << 0),
    TBS_FLAG_SINGLE   = (1 << 1),
    TBS_FLAG_BOUNCE   = (1 << 2),
};

#pragma pack(push, 1)
typedef struct tbs_frame_hdr {
    uint64_t magic;         /* TBS_MAGIC_VAL */
    uint64_t total_len;     /* logical payload bytes in this message */
    uint64_t seq;           /* per-message sequence */
    uint32_t chunk_idx;     /* 0 .. total_chunks-1 */
    uint32_t chunk_bytes;   /* payload bytes in this frame, <= TBS_FRAME_PAYLOAD */
} tbs_frame_hdr;
#pragma pack(pop)

static_assert(sizeof(tbs_frame_hdr) == TBS_FRAME_HDR_SIZE, "tbs_frame_hdr must be 32 bytes");

/* Alias kept for existing callers */
typedef tbs_frame_hdr tbs_msg_hdr;

/* Core C API */
tbs_pipe *tbs_open(const tbs_config *cfg);
void tbs_close(tbs_pipe *pipe);

/* Basic transfer operations */
int tbs_send(tbs_pipe *pipe, const void *buf, size_t n);
int tbs_recv(tbs_pipe *pipe, void *buf, size_t n);
/* Like tbs_recv but return -2 on timeout_ms. timeout_ms < 0 waits forever. */
int tbs_recv_timeout(tbs_pipe *pipe, void *buf, size_t n, int timeout_ms);
/* Read one framed message. Caller frees *buf. n=0 still returns a non-NULL buf. */
int tbs_recv_malloc(tbs_pipe *pipe, void **buf, size_t *n);

/*
 * Simultaneous bidirectional exchange: sends 'out' while receiving 'in'.
 * Both directions occur concurrently across both striped links to prevent All-Reduce deadlock.
 */
int tbs_xchg(tbs_pipe *pipe, const void *out, void *in, size_t n);

/* Query and status */
int tbs_is_striped(const tbs_pipe *pipe);
size_t tbs_get_stripe_size(const tbs_pipe *pipe);
int tbs_get_fd_a(const tbs_pipe *pipe);
int tbs_get_fd_b(const tbs_pipe *pipe);

/* Sysfs helper functions */
int tbs_sysfs_check_busy_poll(const char *xdomain, const char *stream);
int tbs_sysfs_set_busy_poll(const char *xdomain, const char *stream, int enable);
int tbs_sysfs_discover_xdomains(char *xdomain_a, size_t sz_a, char *xdomain_b, size_t sz_b);

#ifdef __cplusplus
}
#endif

#endif /* TBSTRIPE_H */
