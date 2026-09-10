#define _GNU_SOURCE
#include "tbstripe.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#endif

extern ssize_t tbs_writev_spin(int fd, struct iovec *iov, int iovcnt, int busy_spin);
extern ssize_t tbs_readv_spin(int fd, struct iovec *iov, int iovcnt, int busy_spin);

struct tbs_pipe {
    int fd_a;
    int fd_b;
    size_t stripe_sz;
    int busy_spin;
    int is_striped;
    uint64_t tx_seq;
    uint64_t rx_seq;

    int cpu_affinity_a;
    int cpu_affinity_b;

    int simplex;
    int is_master;
    int fd_tx;
    int fd_rx;

    pthread_t tx_worker_b;
    pthread_mutex_t tx_mu;
    pthread_cond_t  tx_cv_in;
    pthread_cond_t  tx_cv_out;
    struct iovec *tx_iov;
    int tx_iovcnt;
    ssize_t tx_res;
    int tx_cmd;
    int tx_running;

    pthread_t rx_worker_b;
    pthread_mutex_t rx_mu;
    pthread_cond_t  rx_cv_in;
    pthread_cond_t  rx_cv_out;
    struct iovec *rx_iov;
    int rx_iovcnt;
    ssize_t rx_res;
    int rx_cmd;
    int rx_running;

    pthread_t xchg_worker;
    pthread_mutex_t xchg_mu;
    pthread_cond_t  xchg_cv_in;
    pthread_cond_t  xchg_cv_out;
    const void *xchg_send_buf;
    size_t xchg_send_size;
    int xchg_send_res;
    int xchg_cmd;
    int xchg_running;

    pthread_mutex_t send_mu;
    pthread_mutex_t recv_mu;

    uint8_t *tx_a;
    uint8_t *tx_b;
    uint8_t *rx_a;
    uint8_t *rx_b;
};

static void *tbs_tx_worker_b_loop(void *arg) {
    tbs_pipe *pipe = (tbs_pipe *)arg;

    if (pipe->cpu_affinity_b >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(pipe->cpu_affinity_b, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }

    pthread_mutex_lock(&pipe->tx_mu);
    while (pipe->tx_running) {
        while (pipe->tx_cmd == 0 && pipe->tx_running) {
            pthread_cond_wait(&pipe->tx_cv_in, &pipe->tx_mu);
        }
        if (!pipe->tx_running || pipe->tx_cmd == 2) {
            break;
        }

        struct iovec *iov = pipe->tx_iov;
        int iovcnt = pipe->tx_iovcnt;
        pthread_mutex_unlock(&pipe->tx_mu);

        ssize_t res = tbs_writev_spin(pipe->fd_b, iov, iovcnt, pipe->busy_spin);

        pthread_mutex_lock(&pipe->tx_mu);
        pipe->tx_res = res;
        pipe->tx_cmd = 0;
        pthread_cond_signal(&pipe->tx_cv_out);
    }
    pthread_mutex_unlock(&pipe->tx_mu);
    return NULL;
}

static void *tbs_rx_worker_b_loop(void *arg) {
    tbs_pipe *pipe = (tbs_pipe *)arg;

    if (pipe->cpu_affinity_b >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(pipe->cpu_affinity_b + 1, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }

    pthread_mutex_lock(&pipe->rx_mu);
    while (pipe->rx_running) {
        while (pipe->rx_cmd == 0 && pipe->rx_running) {
            pthread_cond_wait(&pipe->rx_cv_in, &pipe->rx_mu);
        }
        if (!pipe->rx_running || pipe->rx_cmd == 2) {
            break;
        }

        struct iovec *iov = pipe->rx_iov;
        int iovcnt = pipe->rx_iovcnt;
        pthread_mutex_unlock(&pipe->rx_mu);

        ssize_t res = tbs_readv_spin(pipe->fd_b, iov, iovcnt, pipe->busy_spin);

        pthread_mutex_lock(&pipe->rx_mu);
        pipe->rx_res = res;
        pipe->rx_cmd = 0;
        pthread_cond_signal(&pipe->rx_cv_out);
    }
    pthread_mutex_unlock(&pipe->rx_mu);
    return NULL;
}

static void *tbs_xchg_worker_loop(void *arg) {
    tbs_pipe *pipe = (tbs_pipe *)arg;

    if (pipe->cpu_affinity_a >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(pipe->cpu_affinity_a, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }

    pthread_mutex_lock(&pipe->xchg_mu);
    while (pipe->xchg_running) {
        while (pipe->xchg_cmd == 0 && pipe->xchg_running) {
            pthread_cond_wait(&pipe->xchg_cv_in, &pipe->xchg_mu);
        }
        if (!pipe->xchg_running || pipe->xchg_cmd == 2) {
            break;
        }

        const void *buf = pipe->xchg_send_buf;
        size_t sz = pipe->xchg_send_size;
        pthread_mutex_unlock(&pipe->xchg_mu);

        int res = tbs_send(pipe, buf, sz);

        pthread_mutex_lock(&pipe->xchg_mu);
        pipe->xchg_send_res = res;
        pipe->xchg_cmd = 0;
        pthread_cond_signal(&pipe->xchg_cv_out);
    }
    pthread_mutex_unlock(&pipe->xchg_mu);
    return NULL;
}

static inline void tbs_pause(void) {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#else
    sched_yield();
#endif
}

/* IMP-10: TBSTRIPE_SIMPLEX=master|worker|1. Unset/0/duplex keeps striped duplex. */
static int tbs_env_simplex(int *is_master) {
    const char *sx = getenv("TBSTRIPE_SIMPLEX");
    if (!sx || !sx[0]) {
        return 0;
    }
    if (sx[0] == '0' ||
        strcasecmp(sx, "off") == 0 ||
        strcasecmp(sx, "false") == 0 ||
        strcasecmp(sx, "no") == 0 ||
        strcasecmp(sx, "duplex") == 0) {
        return 0;
    }
    *is_master = 1;
    if (strcasecmp(sx, "worker") == 0 ||
        strcasecmp(sx, "peer") == 0 ||
        strcasecmp(sx, "slave") == 0) {
        *is_master = 0;
        return 1;
    }
    const char *role = getenv("TBSTRIPE_ROLE");
    if (role && (strcasecmp(role, "worker") == 0 ||
                 strcasecmp(role, "peer") == 0)) {
        *is_master = 0;
    }
    return 1;
}

static int open_device_dir(const char *path, int wr) {
    if (!path || !path[0]) {
        return -1;
    }
    /* O_NONBLOCK: USB4STREAM blocking read can return 0 (EOF) while idle. */
    int flags = O_NONBLOCK | (wr ? O_WRONLY : O_RDONLY);
    int fd = open(path, flags);
    if (fd >= 0) {
        return fd;
    }
    int saved = errno;
    fd = open(path, O_RDWR | O_NONBLOCK);
    if (fd >= 0) {
        fprintf(stderr, "[TBSTRIPE] %s: %s open failed (%s); using O_RDWR unidirectional\n",
                path, wr ? "O_WRONLY" : "O_RDONLY", strerror(saved));
        return fd;
    }
    fprintf(stderr, "[TBSTRIPE] %s: directional (%s) and RDWR (%s) open failed\n",
            path, strerror(saved), strerror(errno));
    return -1;
}

static int open_device_nonblock(const char *path) {
    if (!path || !path[0]) return -1;
    int fd = open(path, O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        return -1;
    }
    return fd;
}

static void tbs_pipe_fail_open(tbs_pipe *pipe) {
    if (!pipe) {
        return;
    }
    if (pipe->fd_a >= 0) {
        close(pipe->fd_a);
    }
    if (pipe->fd_b >= 0 && pipe->fd_b != pipe->fd_a) {
        close(pipe->fd_b);
    }
    free(pipe->tx_a);
    free(pipe->rx_a);
    free(pipe->tx_b);
    free(pipe->rx_b);
    free(pipe);
}

tbs_pipe *tbs_open(const tbs_config *cfg) {
    if (!cfg || !cfg->dev_a) {
        return NULL;
    }

    tbs_pipe *pipe = (tbs_pipe *)calloc(1, sizeof(tbs_pipe));
    if (!pipe) return NULL;

    pipe->stripe_sz = cfg->stripe ? cfg->stripe : TBS_DEFAULT_STRIPE_SZ;
    pipe->busy_spin = cfg->busy_spin;
    pipe->cpu_affinity_a = cfg->cpu_affinity_a;
    pipe->cpu_affinity_b = cfg->cpu_affinity_b;
    pipe->fd_a = -1;
    pipe->fd_b = -1;
    pipe->fd_tx = -1;
    pipe->fd_rx = -1;

    {
        int is_master = 1;
        if (tbs_env_simplex(&is_master)) {
            if (!cfg->dev_b || !cfg->dev_b[0]) {
                fprintf(stderr, "[TBSTRIPE] TBSTRIPE_SIMPLEX needs both dev_a and dev_b\n");
                free(pipe);
                return NULL;
            }
            pipe->simplex = 1;
            pipe->is_master = is_master;
            pipe->is_striped = 0;
            if (is_master) {
                pipe->fd_tx = open_device_dir(cfg->dev_a, 1);
                pipe->fd_rx = open_device_dir(cfg->dev_b, 0);
            } else {
                pipe->fd_rx = open_device_dir(cfg->dev_a, 0);
                pipe->fd_tx = open_device_dir(cfg->dev_b, 1);
            }
            pipe->fd_a = is_master ? pipe->fd_tx : pipe->fd_rx;
            pipe->fd_b = is_master ? pipe->fd_rx : pipe->fd_tx;
            if (pipe->fd_tx < 0 || pipe->fd_rx < 0) {
                tbs_pipe_fail_open(pipe);
                return NULL;
            }
            {
                const size_t batch_bytes = (size_t)TBS_FRAME_BATCH * TBS_FRAME_SIZE;
                if (posix_memalign((void **)&pipe->tx_a, TBS_ALIGNMENT, batch_bytes) != 0 ||
                    posix_memalign((void **)&pipe->rx_a, TBS_ALIGNMENT, batch_bytes) != 0) {
                    tbs_pipe_fail_open(pipe);
                    return NULL;
                }
            }
            pthread_mutex_init(&pipe->send_mu, NULL);
            pthread_mutex_init(&pipe->recv_mu, NULL);
            fprintf(stderr, "[TBSTRIPE] simplex %s TX=%s RX=%s (no helper threads)\n",
                    is_master ? "master" : "worker",
                    is_master ? cfg->dev_a : cfg->dev_b,
                    is_master ? cfg->dev_b : cfg->dev_a);
            return pipe;
        }
    }

    pipe->fd_a = open_device_nonblock(cfg->dev_a);
    if (pipe->fd_a < 0) {
        free(pipe);
        return NULL;
    }

    pipe->fd_b = -1;
    if (cfg->dev_b && cfg->dev_b[0]) {
        pipe->fd_b = open_device_nonblock(cfg->dev_b);
        if (pipe->fd_b >= 0) {
            pipe->is_striped = 1;
        } else {
            fprintf(stderr, "[TBSTRIPE] Error: could not open dev_b '%s' (%s). Dual-stream is required.\n",
                    cfg->dev_b, strerror(errno));
            close(pipe->fd_a);
            free(pipe);
            return NULL;
        }
    } else {
        pipe->is_striped = 0;
    }

    {
        const size_t batch_bytes = (size_t)TBS_FRAME_BATCH * TBS_FRAME_SIZE;
        if (posix_memalign((void **)&pipe->tx_a, TBS_ALIGNMENT, batch_bytes) != 0 ||
            posix_memalign((void **)&pipe->rx_a, TBS_ALIGNMENT, batch_bytes) != 0) {
            close(pipe->fd_a);
            if (pipe->fd_b >= 0) {
                close(pipe->fd_b);
            }
            free(pipe->tx_a);
            free(pipe->rx_a);
            free(pipe);
            return NULL;
        }
        if (pipe->is_striped) {
            if (posix_memalign((void **)&pipe->tx_b, TBS_ALIGNMENT, batch_bytes) != 0 ||
                posix_memalign((void **)&pipe->rx_b, TBS_ALIGNMENT, batch_bytes) != 0) {
                close(pipe->fd_a);
                close(pipe->fd_b);
                free(pipe->tx_a);
                free(pipe->rx_a);
                free(pipe->tx_b);
                free(pipe->rx_b);
                free(pipe);
                return NULL;
            }
        }
    }

    pthread_mutex_init(&pipe->send_mu, NULL);
    pthread_mutex_init(&pipe->recv_mu, NULL);

    if (pipe->is_striped) {
        pthread_mutex_init(&pipe->tx_mu, NULL);
        pthread_cond_init(&pipe->tx_cv_in, NULL);
        pthread_cond_init(&pipe->tx_cv_out, NULL);
        pipe->tx_running = 1;
        pipe->tx_cmd = 0;
        pthread_create(&pipe->tx_worker_b, NULL, tbs_tx_worker_b_loop, pipe);

        pthread_mutex_init(&pipe->rx_mu, NULL);
        pthread_cond_init(&pipe->rx_cv_in, NULL);
        pthread_cond_init(&pipe->rx_cv_out, NULL);
        pipe->rx_running = 1;
        pipe->rx_cmd = 0;
        pthread_create(&pipe->rx_worker_b, NULL, tbs_rx_worker_b_loop, pipe);
    }

    pthread_mutex_init(&pipe->xchg_mu, NULL);
    pthread_cond_init(&pipe->xchg_cv_in, NULL);
    pthread_cond_init(&pipe->xchg_cv_out, NULL);
    pipe->xchg_running = 1;
    pipe->xchg_cmd = 0;
    pthread_create(&pipe->xchg_worker, NULL, tbs_xchg_worker_loop, pipe);

    return pipe;
}

void tbs_close(tbs_pipe *pipe) {
    if (!pipe) return;

    if (pipe->xchg_running) {
        pthread_mutex_lock(&pipe->xchg_mu);
        pipe->xchg_running = 0;
        pipe->xchg_cmd = 2;
        pthread_cond_broadcast(&pipe->xchg_cv_in);
        pthread_mutex_unlock(&pipe->xchg_mu);
        pthread_join(pipe->xchg_worker, NULL);
        pthread_mutex_destroy(&pipe->xchg_mu);
        pthread_cond_destroy(&pipe->xchg_cv_in);
        pthread_cond_destroy(&pipe->xchg_cv_out);
    }

    if (pipe->is_striped) {
        if (pipe->tx_running) {
            pthread_mutex_lock(&pipe->tx_mu);
            pipe->tx_running = 0;
            pipe->tx_cmd = 2;
            pthread_cond_broadcast(&pipe->tx_cv_in);
            pthread_mutex_unlock(&pipe->tx_mu);
            pthread_join(pipe->tx_worker_b, NULL);
            pthread_mutex_destroy(&pipe->tx_mu);
            pthread_cond_destroy(&pipe->tx_cv_in);
            pthread_cond_destroy(&pipe->tx_cv_out);
        }

        if (pipe->rx_running) {
            pthread_mutex_lock(&pipe->rx_mu);
            pipe->rx_running = 0;
            pipe->rx_cmd = 2;
            pthread_cond_broadcast(&pipe->rx_cv_in);
            pthread_mutex_unlock(&pipe->rx_mu);
            pthread_join(pipe->rx_worker_b, NULL);
            pthread_mutex_destroy(&pipe->rx_mu);
            pthread_cond_destroy(&pipe->rx_cv_in);
            pthread_cond_destroy(&pipe->rx_cv_out);
        }
    }

    if (pipe->fd_b >= 0) {
        close(pipe->fd_b);
        pipe->fd_b = -1;
    }
    if (pipe->fd_a >= 0) {
        close(pipe->fd_a);
        pipe->fd_a = -1;
    }

    free(pipe->tx_a);
    free(pipe->tx_b);
    free(pipe->rx_a);
    free(pipe->rx_b);

    pthread_mutex_destroy(&pipe->send_mu);
    pthread_mutex_destroy(&pipe->recv_mu);
    free(pipe);
}

int tbs_is_striped(const tbs_pipe *pipe) {
    return pipe ? pipe->is_striped : 0;
}

size_t tbs_get_stripe_size(const tbs_pipe *pipe) {
    return pipe ? pipe->stripe_sz : TBS_DEFAULT_STRIPE_SZ;
}

int tbs_get_fd_a(const tbs_pipe *pipe) {
    return pipe ? pipe->fd_a : -1;
}

int tbs_get_fd_b(const tbs_pipe *pipe) {
    return pipe ? pipe->fd_b : -1;
}

static uint32_t tbs_total_chunks(size_t n) {
    if (n == 0) {
        return 1;
    }
    return (uint32_t)((n + (size_t)TBS_FRAME_PAYLOAD - 1) / (size_t)TBS_FRAME_PAYLOAD);
}

static uint32_t tbs_chunk_payload(size_t n, uint32_t idx) {
    size_t off = (size_t)idx * (size_t)TBS_FRAME_PAYLOAD;
    if (off >= n) {
        return 0;
    }
    size_t rem = n - off;
    return rem > (size_t)TBS_FRAME_PAYLOAD ? TBS_FRAME_PAYLOAD : (uint32_t)rem;
}

static void tbs_pack_frame(uint8_t *frame, uint64_t seq, size_t n, uint32_t idx, const uint8_t *src) {
    tbs_frame_hdr *h = (tbs_frame_hdr *)frame;
    uint32_t payload = tbs_chunk_payload(n, idx);
    size_t off = (size_t)idx * (size_t)TBS_FRAME_PAYLOAD;
    h->magic = TBS_MAGIC_VAL;
    h->total_len = (uint64_t)n;
    h->seq = seq;
    h->chunk_idx = idx;
    h->chunk_bytes = payload;
    if (payload) {
        memcpy(frame + TBS_FRAME_HDR_SIZE, src + off, payload);
    }
    if (payload < TBS_FRAME_PAYLOAD) {
        memset(frame + TBS_FRAME_HDR_SIZE + payload, 0,
               (size_t)TBS_FRAME_SIZE - TBS_FRAME_HDR_SIZE - payload);
    }
}

static int tbs_unpack_frame(const uint8_t *frame, uint64_t seq, size_t n, uint32_t expect_idx, uint8_t *dst) {
    const tbs_frame_hdr *h = (const tbs_frame_hdr *)frame;
    uint32_t expect = tbs_chunk_payload(n, expect_idx);
    if (h->magic != TBS_MAGIC_VAL) {
        fprintf(stderr, "[TBSTRIPE] magic mismatch chunk %u: got 0x%016lx\n",
                expect_idx, (unsigned long)h->magic);
        return -1;
    }
    if (h->total_len != (uint64_t)n || h->seq != seq || h->chunk_idx != expect_idx ||
        h->chunk_bytes != expect) {
        fprintf(stderr, "[TBSTRIPE] hdr mismatch chunk %u (len=%lu seq=%lu idx=%u bytes=%u)\n",
                expect_idx, (unsigned long)h->total_len, (unsigned long)h->seq,
                h->chunk_idx, h->chunk_bytes);
        return -1;
    }
    if (expect) {
        memcpy(dst + (size_t)expect_idx * TBS_FRAME_PAYLOAD, frame + TBS_FRAME_HDR_SIZE, expect);
    }
    return 0;
}

#define TBS_SIMPLEX_BATCH 8

static int tbs_write_frame(int fd, const uint8_t *frame, int busy_spin) {
    for (;;) {
        ssize_t nw = write(fd, frame, TBS_FRAME_SIZE);
        if (nw == (ssize_t)TBS_FRAME_SIZE) {
            return 0;
        }
        if (nw > 0 && nw != (ssize_t)TBS_FRAME_SIZE) {
            /* USB4STREAM is a 4KiB packet device. Do not emit a remainder. */
            continue;
        }
        if (nw == 0) {
            nw = -1;
            errno = EAGAIN;
        }
        if (nw < 0 && errno == EINTR) {
            continue;
        }
        if (nw < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (busy_spin) {
                tbs_pause();
            } else {
                sched_yield();
            }
            continue;
        }
        fprintf(stderr, "[TBSTRIPE] write frame: %s\n", strerror(errno));
        return -1;
    }
}

static int tbs_read_frame(int fd, uint8_t *frame, int busy_spin, int timeout_ms) {
    struct timespec t0;
    if (timeout_ms >= 0) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }
    for (;;) {
        ssize_t nr = read(fd, frame, TBS_FRAME_SIZE);
        if (nr == (ssize_t)TBS_FRAME_SIZE) {
            return 0;
        }
        if (nr > 0 && nr != (ssize_t)TBS_FRAME_SIZE) {
            /* Discard short USB4 packets so we do not assemble a misaligned frame. */
            continue;
        }
        if (nr == 0) {
            nr = -1;
            errno = EAGAIN;
        }
        if (errno == EINTR) {
            continue;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            if (timeout_ms >= 0) {
                struct timespec t1;
                clock_gettime(CLOCK_MONOTONIC, &t1);
                long ms = (t1.tv_sec - t0.tv_sec) * 1000L +
                          (t1.tv_nsec - t0.tv_nsec) / 1000000L;
                if (ms >= timeout_ms) {
                    return -2;
                }
            }
            if (busy_spin) {
                tbs_pause();
            } else {
                sched_yield();
            }
            continue;
        }
        fprintf(stderr, "[TBSTRIPE] read frame: %s\n", strerror(errno));
        return -1;
    }
}

static int tbs_send_simplex(tbs_pipe *pipe, const void *buf, size_t n) {
    if (pipe->fd_tx < 0) {
        return -1;
    }
    if (n > 0 && buf == NULL) {
        return -1;
    }

    pthread_mutex_lock(&pipe->send_mu);

    const uint8_t *src = (const uint8_t *)buf;
    uint64_t seq = ++pipe->tx_seq;
    uint32_t total_chunks = tbs_total_chunks(n);
    int ret = 0;
    uint32_t idx = 0;

    while (idx < total_chunks && ret == 0) {
        tbs_pack_frame(pipe->tx_a, seq, n, idx, src);
        if (tbs_write_frame(pipe->fd_tx, pipe->tx_a, pipe->busy_spin) != 0) {
            ret = -1;
        }
        idx++;
    }

    pthread_mutex_unlock(&pipe->send_mu);
    return ret;
}

static int tbs_recv_simplex(tbs_pipe *pipe, void *buf, size_t n, int timeout_ms) {
    if (pipe->fd_rx < 0) {
        return -1;
    }
    if (n > 0 && buf == NULL) {
        return -1;
    }

    pthread_mutex_lock(&pipe->recv_mu);

    uint8_t *dst = (uint8_t *)buf;
    uint32_t total_chunks = tbs_total_chunks(n);
    uint64_t seq = pipe->rx_seq + 1;
    int ret = 0;
    uint32_t idx = 0;

    while (idx < total_chunks && ret == 0) {
        int rf = tbs_read_frame(pipe->fd_rx, pipe->rx_a, pipe->busy_spin, timeout_ms);
        if (rf != 0) {
            ret = rf;
            break;
        }
        if (tbs_unpack_frame(pipe->rx_a, seq, n, idx, dst) != 0) {
            ret = -1;
        }
        idx++;
    }

    if (ret == 0) {
        pipe->rx_seq = seq;
    }

    pthread_mutex_unlock(&pipe->recv_mu);
    return ret;
}

/*
 * Sequential simplex xchg. Same-process concurrent TX+RX deadlocks on
 * USB4STREAM (driver cannot write one stream while another thread reads
 * the pair). WINDOW=total of a large payload also deadlocks (fills the
 * RX ring before any RX).
 *
 * FFN AllReduce is ~10 KiB = 3 frames. Burst-TX then RX when the whole
 * message fits in 8 frames (32 KiB) is one cable RTT and fits in the
 * 2048-frame ring. Larger xchg stays 1:1. TBS_XCHG_WINDOW=1 forces 1:1.
 */
#define TBS_XCHG_SMALL_FRAMES 8

static uint32_t tbs_xchg_window_env(uint32_t total) {
    const char *we = getenv("TBS_XCHG_WINDOW");
    if (we != NULL && we[0] != '\0') {
        int w = atoi(we);
        if (w == 1) {
            return 1;
        }
        if (w > 0) {
            uint32_t window = (uint32_t) w;
            if (window > TBS_XCHG_SMALL_FRAMES) {
                window = TBS_XCHG_SMALL_FRAMES;
            }
            return window;
        }
    }
    if (total <= TBS_XCHG_SMALL_FRAMES) {
        return total;
    }
    return 1;
}

static int tbs_xchg_simplex(tbs_pipe *pipe, const void *out, void *in, size_t n) {
    if (pipe->fd_tx < 0 || pipe->fd_rx < 0) {
        return -1;
    }
    if (n > 0 && (out == NULL || in == NULL)) {
        return -1;
    }
    if (n == 0) {
        return 0;
    }

    pthread_mutex_lock(&pipe->send_mu);
    pthread_mutex_lock(&pipe->recv_mu);

    const uint8_t *src = (const uint8_t *)out;
    uint8_t *dst = (uint8_t *)in;
    uint32_t total = tbs_total_chunks(n);
    uint64_t tx_seq = ++pipe->tx_seq;
    uint64_t rx_seq = pipe->rx_seq + 1;
    uint32_t window = tbs_xchg_window_env(total);
    uint32_t tx_idx = 0;
    uint32_t rx_idx = 0;
    int ret = 0;

    if (window < 1) {
        window = 1;
    }
    while ((tx_idx < total || rx_idx < total) && ret == 0) {
        while (tx_idx < total && (tx_idx - rx_idx) < window && ret == 0) {
            tbs_pack_frame(pipe->tx_a, tx_seq, n, tx_idx, src);
            if (tbs_write_frame(pipe->fd_tx, pipe->tx_a, pipe->busy_spin) != 0) {
                ret = -1;
                break;
            }
            tx_idx++;
        }
        if (ret != 0) {
            break;
        }
        if (rx_idx < total) {
            if (tbs_read_frame(pipe->fd_rx, pipe->rx_a, pipe->busy_spin, -1) != 0) {
                ret = -1;
                break;
            }
            if (tbs_unpack_frame(pipe->rx_a, rx_seq, n, rx_idx, dst) != 0) {
                ret = -1;
                break;
            }
            rx_idx++;
        }
    }

    if (ret == 0) {
        pipe->rx_seq = rx_seq;
        static int nlog;
        if (nlog < 2) {
            nlog++;
            fprintf(stderr, "[TBSTRIPE] xchg window=%u chunks=%u bytes=%zu\n",
                    window, total, n);
        }
    }

    pthread_mutex_unlock(&pipe->recv_mu);
    pthread_mutex_unlock(&pipe->send_mu);
    return ret;
}

int tbs_send(tbs_pipe *pipe, const void *buf, size_t n) {
    if (!pipe || pipe->fd_a < 0) {
        return -1;
    }
    if (pipe->simplex) {
        return tbs_send_simplex(pipe, buf, n);
    }
    if (n > 0 && buf == NULL) {
        return -1;
    }

    pthread_mutex_lock(&pipe->send_mu);

    const uint8_t *src = (const uint8_t *)buf;
    uint64_t seq = ++pipe->tx_seq;
    uint32_t total_chunks = tbs_total_chunks(n);
    const int striped = pipe->is_striped && pipe->fd_b >= 0;

    struct iovec iov_a;
    struct iovec iov_b;

    int ret = 0;
    uint32_t idx = 0;
    while (idx < total_chunks && ret == 0) {
        int na = 0;
        int nb = 0;

        while (idx < total_chunks && na < TBS_FRAME_BATCH && (!striped || nb < TBS_FRAME_BATCH)) {
            if (!striped || (idx % 2u) == 0u) {
                if (na >= TBS_FRAME_BATCH) {
                    break;
                }
                tbs_pack_frame(pipe->tx_a + (size_t)na * TBS_FRAME_SIZE, seq, n, idx, src);
                na++;
            } else {
                if (nb >= TBS_FRAME_BATCH) {
                    break;
                }
                tbs_pack_frame(pipe->tx_b + (size_t)nb * TBS_FRAME_SIZE, seq, n, idx, src);
                nb++;
            }
            idx++;
        }

        ssize_t want_a = (ssize_t)na * TBS_FRAME_SIZE;
        ssize_t want_b = (ssize_t)nb * TBS_FRAME_SIZE;
        iov_a.iov_base = pipe->tx_a;
        iov_a.iov_len = (size_t)want_a;
        iov_b.iov_base = pipe->tx_b;
        iov_b.iov_len = (size_t)want_b;

        if (striped && nb > 0) {
            if (total_chunks <= 16) {
                ssize_t res_a = tbs_writev_spin(pipe->fd_a, &iov_a, 1, pipe->busy_spin);
                ssize_t res_b = tbs_writev_spin(pipe->fd_b, &iov_b, 1, pipe->busy_spin);
                if (res_a != want_a || res_b != want_b) {
                    ret = -1;
                }
            } else {
                pthread_mutex_lock(&pipe->tx_mu);
                pipe->tx_iov = &iov_b;
                pipe->tx_iovcnt = 1;
                pipe->tx_cmd = 1;
                pthread_cond_signal(&pipe->tx_cv_in);
                pthread_mutex_unlock(&pipe->tx_mu);

                ssize_t res_a = tbs_writev_spin(pipe->fd_a, &iov_a, 1, pipe->busy_spin);

                pthread_mutex_lock(&pipe->tx_mu);
                while (pipe->tx_cmd != 0) {
                    pthread_cond_wait(&pipe->tx_cv_out, &pipe->tx_mu);
                }
                ssize_t res_b = pipe->tx_res;
                pthread_mutex_unlock(&pipe->tx_mu);

                if (res_a != want_a || res_b != want_b) {
                    ret = -1;
                }
            }
        } else {
            ssize_t res_a = tbs_writev_spin(pipe->fd_a, &iov_a, 1, pipe->busy_spin);
            if (res_a != want_a) {
                ret = -1;
            }
        }
    }

    pthread_mutex_unlock(&pipe->send_mu);
    return ret;
}

int tbs_recv(tbs_pipe *pipe, void *buf, size_t n) {
    if (!pipe || pipe->fd_a < 0) {
        return -1;
    }
    if (pipe->simplex) {
        return tbs_recv_simplex(pipe, buf, n, -1);
    }
    if (n > 0 && buf == NULL) {
        return -1;
    }

    pthread_mutex_lock(&pipe->recv_mu);

    uint8_t *dst = (uint8_t *)buf;
    uint32_t total_chunks = tbs_total_chunks(n);
    const int striped = pipe->is_striped && pipe->fd_b >= 0;
    uint64_t seq = pipe->rx_seq + 1;

    struct iovec iov_a;
    struct iovec iov_b;
    uint32_t a_idx[TBS_FRAME_BATCH];
    uint32_t b_idx[TBS_FRAME_BATCH];

    int ret = 0;
    uint32_t idx = 0;
    while (idx < total_chunks && ret == 0) {
        int na = 0;
        int nb = 0;

        while (idx < total_chunks && na < TBS_FRAME_BATCH && (!striped || nb < TBS_FRAME_BATCH)) {
            if (!striped || (idx % 2u) == 0u) {
                if (na >= TBS_FRAME_BATCH) {
                    break;
                }
                a_idx[na] = idx;
                na++;
            } else {
                if (nb >= TBS_FRAME_BATCH) {
                    break;
                }
                b_idx[nb] = idx;
                nb++;
            }
            idx++;
        }

        ssize_t want_a = (ssize_t)na * TBS_FRAME_SIZE;
        ssize_t want_b = (ssize_t)nb * TBS_FRAME_SIZE;
        iov_a.iov_base = pipe->rx_a;
        iov_a.iov_len = (size_t)want_a;
        iov_b.iov_base = pipe->rx_b;
        iov_b.iov_len = (size_t)want_b;

        if (striped && nb > 0) {
            if (total_chunks <= 16) {
                ssize_t res_a = tbs_readv_spin(pipe->fd_a, &iov_a, 1, pipe->busy_spin);
                ssize_t res_b = tbs_readv_spin(pipe->fd_b, &iov_b, 1, pipe->busy_spin);
                if (res_a != want_a || res_b != want_b) {
                    ret = -1;
                }
            } else {
                pthread_mutex_lock(&pipe->rx_mu);
                pipe->rx_iov = &iov_b;
                pipe->rx_iovcnt = 1;
                pipe->rx_cmd = 1;
                pthread_cond_signal(&pipe->rx_cv_in);
                pthread_mutex_unlock(&pipe->rx_mu);

                ssize_t res_a = tbs_readv_spin(pipe->fd_a, &iov_a, 1, pipe->busy_spin);

                pthread_mutex_lock(&pipe->rx_mu);
                while (pipe->rx_cmd != 0) {
                    pthread_cond_wait(&pipe->rx_cv_out, &pipe->rx_mu);
                }
                ssize_t res_b = pipe->rx_res;
                pthread_mutex_unlock(&pipe->rx_mu);

                if (res_a != want_a || res_b != want_b) {
                    ret = -1;
                }
            }
        } else {
            ssize_t res_a = tbs_readv_spin(pipe->fd_a, &iov_a, 1, pipe->busy_spin);
            if (res_a != want_a) {
                ret = -1;
            }
        }

        for (int i = 0; i < na && ret == 0; i++) {
            if (tbs_unpack_frame(pipe->rx_a + (size_t)i * TBS_FRAME_SIZE, seq, n, a_idx[i], dst) != 0) {
                ret = -1;
            }
        }
        for (int i = 0; i < nb && ret == 0; i++) {
            if (tbs_unpack_frame(pipe->rx_b + (size_t)i * TBS_FRAME_SIZE, seq, n, b_idx[i], dst) != 0) {
                ret = -1;
            }
        }
    }

    if (ret == 0) {
        pipe->rx_seq = seq;
    }

    pthread_mutex_unlock(&pipe->recv_mu);
    return ret;
}

int tbs_recv_timeout(tbs_pipe *pipe, void *buf, size_t n, int timeout_ms) {
    if (!pipe || pipe->fd_a < 0) {
        return -1;
    }
    if (pipe->simplex) {
        return tbs_recv_simplex(pipe, buf, n, timeout_ms);
    }
    return tbs_recv(pipe, buf, n);
}

int tbs_recv_malloc(tbs_pipe *pipe, void **buf, size_t *n) {
    if (!pipe || !buf || !n) {
        return -1;
    }
    *buf = NULL;
    *n = 0;

    int fd = pipe->simplex ? pipe->fd_rx : pipe->fd_a;
    if (fd < 0) {
        return -1;
    }

    pthread_mutex_lock(&pipe->recv_mu);

    unsigned skipped = 0;
    size_t total = 0;
    uint64_t seq = 0;
    uint8_t *dst = NULL;
    uint32_t chunks = 0;
    int ret = 0;
    uint32_t idx = 0;
    const tbs_frame_hdr *h = NULL;
find_start:
    if (dst) {
        free(dst);
        dst = NULL;
    }
    if (tbs_read_frame(fd, pipe->rx_a, pipe->busy_spin, -1) != 0) {
        pthread_mutex_unlock(&pipe->recv_mu);
        return -1;
    }

    h = (const tbs_frame_hdr *)pipe->rx_a;
    if (h->magic != TBS_MAGIC_VAL || h->chunk_idx != 0) {
        skipped++;
        if (skipped == 1 || (skipped % 1024u) == 0u) {
            fprintf(stderr, "[TBSTRIPE] skip frame magic=0x%016lx idx=%u (skipped=%u)\n",
                    (unsigned long)h->magic, h->chunk_idx, skipped);
        }
        goto find_start;
    }

    total = (size_t)h->total_len;
    seq = h->seq;
    dst = (uint8_t *)malloc(total > 0 ? total : 1);
    if (!dst) {
        pthread_mutex_unlock(&pipe->recv_mu);
        return -1;
    }

    if (tbs_unpack_frame(pipe->rx_a, seq, total, 0, dst) != 0) {
        goto find_start;
    }

    chunks = tbs_total_chunks(total);
    ret = 0;
    for (idx = 1; idx < chunks && ret == 0; idx++) {
        if (tbs_read_frame(fd, pipe->rx_a, pipe->busy_spin, -1) != 0) {
            ret = -1;
            break;
        }
        if (tbs_unpack_frame(pipe->rx_a, seq, total, idx, dst) != 0) {
            ret = -1;
        }
    }

    if (ret != 0) {
        goto find_start;
    }

    pipe->rx_seq = seq;
    pthread_mutex_unlock(&pipe->recv_mu);
    *buf = dst;
    *n = total;
    return 0;
}

int tbs_xchg(tbs_pipe *pipe, const void *out, void *in, size_t n) {
    if (!pipe) return -1;
    if (pipe->simplex) {
        return tbs_xchg_simplex(pipe, out, in, n);
    }

    pthread_mutex_lock(&pipe->xchg_mu);
    pipe->xchg_send_buf = out;
    pipe->xchg_send_size = n;
    pipe->xchg_cmd = 1;
    pthread_cond_signal(&pipe->xchg_cv_in);
    pthread_mutex_unlock(&pipe->xchg_mu);

    int r_recv = tbs_recv(pipe, in, n);

    pthread_mutex_lock(&pipe->xchg_mu);
    while (pipe->xchg_cmd != 0) {
        pthread_cond_wait(&pipe->xchg_cv_out, &pipe->xchg_mu);
    }
    int r_send = pipe->xchg_send_res;
    pthread_mutex_unlock(&pipe->xchg_mu);

    return (r_send == 0 && r_recv == 0) ? 0 : -1;
}
