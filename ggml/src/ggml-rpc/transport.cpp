#include "transport.h"
#include "ggml-impl.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#else
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#  include <poll.h>
#endif
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <algorithm>
#include <vector>
#include <thread>
#include <chrono>
#include <future>
#include <condition_variable>
#include <functional>
#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#endif

#ifdef GGML_RPC_RDMA
#  include <infiniband/verbs.h>
#  include <array>
#  include <time.h>
#  ifdef GGML_RPC_RDMA_APPLE
#    include "transport-apple.h"
#  endif
#endif // GGML_RPC_RDMA

#ifdef GGML_TBSTRIPE
#  include "tbstripe.h"
#endif

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

#ifdef GGML_RPC_RDMA
static constexpr size_t RDMA_GID_SIZE = 16;
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#if defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
static constexpr size_t RDMA_CHUNK    = 256 * 1024;
static constexpr int    RDMA_RX_DEPTH = 24;

struct rdma_conn {
    struct ibv_context * ctx = nullptr;
    struct ibv_pd * pd  = nullptr;
    struct ibv_cq * scq = nullptr;
    struct ibv_cq * rcq = nullptr;
    struct ibv_qp * qp  = nullptr;

    void          * tx_buf = nullptr;
    struct ibv_mr * tx_mr  = nullptr;

    void          * rx_buf = nullptr;
    struct ibv_mr * rx_mr  = nullptr;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(int i) const {
        return static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    bool post_rx(int i) {
        struct ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        struct ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        return ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        if (tx_mr) ibv_dereg_mr(tx_mr);
        if (rx_mr) ibv_dereg_mr(rx_mr);
        free(tx_buf);
        free(rx_buf);
        if (qp)  ibv_destroy_qp(qp);
        if (scq) ibv_destroy_cq(scq);
        if (rcq) ibv_destroy_cq(rcq);
        if (pd)  ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }
};

struct rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    int      gid_idx = 0;
    enum ibv_mtu path_mtu = IBV_MTU_1024;
};

struct rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(sizeof(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

static bool is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    return sockfd != INVALID_SOCKET;
#else
    return sockfd >= 0;
#endif
}

static bool set_no_delay(sockfd_t sockfd) {
    int flag = 1;
    int ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
    return ret == 0;
}

static bool set_reuse_addr(sockfd_t sockfd) {
    int flag = 1;
    int ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&flag, sizeof(int));
    return ret == 0;
}

class tcp_rpc_transport : public rpc_transport {
public:
    sockfd_t fd;
    bool     use_rdma;

#ifdef GGML_RPC_RDMA
#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  else
    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
    bool rdma_probe();
    bool rdma_send(const void * data, size_t size);
    bool rdma_recv(void * data, size_t size);
    bool tcp_peer_closed();
    bool rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, const uint8_t * remote_gid);
    bool rdma_poll(struct ibv_cq * cq, struct ibv_wc * wc);
#  endif
    std::optional<rdma_gid_t> rdma_build_target_gid();
#endif

    explicit tcp_rpc_transport(sockfd_t fd) : fd(fd), use_rdma(false) {}
    ~tcp_rpc_transport() override;

    bool send_exact(const void * data, size_t size) override;
    bool recv_exact(void * data, size_t size) override;
    bool flush() override;
    void close() override;
    std::shared_ptr<rpc_transport> accept() override;
    void get_caps(uint8_t * local_caps) override;
    void update_caps(const uint8_t * remote_caps) override;
    bool is_stream() const override { return false; }
};

tcp_rpc_transport::~tcp_rpc_transport() {
    close();
}

void tcp_rpc_transport::close() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif
    if (is_valid_fd(fd)) {
        LOG_DBG("[%s] closing socket %d\n", __func__, (int)this->fd);
#ifdef _WIN32
        closesocket(this->fd);
        this->fd = INVALID_SOCKET;
#else
        ::close(this->fd);
        this->fd = -1;
#endif
    }
}

#ifdef GGML_RPC_RDMA
std::optional<rdma_gid_t> tcp_rpc_transport::rdma_build_target_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        return std::nullopt;
    }
    rdma_gid_t target = {};
    if (addr.ss_family == AF_INET) {
        const auto * a = reinterpret_cast<const sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        return target;
    }
    if (addr.ss_family == AF_INET6) {
        const auto * a = reinterpret_cast<const sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        return target;
    }
    return std::nullopt;
}

#ifndef GGML_RPC_RDMA_APPLE
bool tcp_rpc_transport::tcp_peer_closed() {
    if (fd < 0) return false;
#ifndef _WIN32
    struct pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    int r = poll(&pfd, 1, 0);
    return r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#else
    return false;
#endif
}

bool tcp_rpc_transport::rdma_probe() {
    const char * dev_env = std::getenv("GGML_RDMA_DEV");
    const char * gid_env = std::getenv("GGML_RDMA_GID");
    auto target_gid = rdma_build_target_gid();
    if (!target_gid) return false;
    const uint8_t ib_port = 1;
    int num_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&num_devs);
    if (!devs || num_devs == 0) return false;
    ibv_context * ibctx = nullptr;
    const char * matched_dev = nullptr;
    int gid_idx = gid_env ? atoi(gid_env) : -1;
    int gid_version = IBV_GID_TYPE_IB;
    for (int d = 0; d < num_devs; d++) {
        const char * dn = ibv_get_device_name(devs[d]);
        if (dev_env && strcmp(dev_env, dn) != 0) continue;
        ibv_context * ctx = ibv_open_device(devs[d]);
        if (!ctx) continue;
        ibv_port_attr pa;
        if (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); continue; }
        int found_gid = gid_idx;
        int found_version = IBV_GID_TYPE_IB;
        if (found_gid < 0) {
            int v2_idx = -1;
            int v1_idx = -1;
            for (int i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                if (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) continue;
                if (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) == 0) {
                    if (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) v2_idx = i;
                    else if (v1_idx < 0) v1_idx = i;
                }
            }
            if (v2_idx >= 0) { found_gid = v2_idx; found_version = IBV_GID_TYPE_ROCE_V2; }
            else if (v1_idx >= 0) { found_gid = v1_idx; found_version = 0; }
        }
        if (found_gid >= 0) { ibctx = ctx; matched_dev = dn; gid_idx = found_gid; found_version = found_version; break; }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    if (!ibctx) { LOG_DBG("RDMA probe: no RoCE device matched local TCP address\n"); return false; }
    auto conn = std::make_unique<rdma_conn>();
    conn->ctx = ibctx;
    conn->pd = ibv_alloc_pd(conn->ctx);
    if (!conn->pd) return false;
    conn->scq = ibv_create_cq(conn->ctx, 16, nullptr, nullptr, 0);
    conn->rcq = ibv_create_cq(conn->ctx, RDMA_RX_DEPTH, nullptr, nullptr, 0);
    if (!conn->scq || !conn->rcq) return false;
    struct ibv_device_attr dev_attr = {};
    ibv_query_device(conn->ctx, &dev_attr);
    uint32_t want_inline = (dev_attr.max_inline_data >= 128) ? 128 : dev_attr.max_inline_data;
    struct ibv_qp_init_attr qpia = {};
    qpia.send_cq = conn->scq;
    qpia.recv_cq = conn->rcq;
    qpia.cap.max_send_wr = 16;
    qpia.cap.max_recv_wr = RDMA_RX_DEPTH;
    qpia.cap.max_send_sge = 1;
    qpia.cap.max_recv_sge = 1;
    qpia.cap.max_inline_data = want_inline;
    qpia.qp_type = IBV_QPT_RC;
    qpia.sq_sig_all = 1;
    conn->qp = ibv_create_qp(conn->pd, &qpia);
    if (!conn->qp) return false;
    conn->max_inline = qpia.cap.max_inline_data;
    conn->tx_buf = malloc(RDMA_CHUNK);
    conn->rx_buf = malloc((size_t)RDMA_RX_DEPTH * RDMA_CHUNK);
    if (!conn->tx_buf || !conn->rx_buf) return false;
    int mr_access = IBV_ACCESS_LOCAL_WRITE;
    conn->tx_mr = ibv_reg_mr(conn->pd, conn->tx_buf, RDMA_CHUNK, mr_access);
    conn->rx_mr = ibv_reg_mr(conn->pd, conn->rx_buf, (size_t)RDMA_RX_DEPTH * RDMA_CHUNK, mr_access);
    if (!conn->tx_mr || !conn->rx_mr) return false;
    for (int i = 0; i < RDMA_RX_DEPTH; i++) if (!conn->post_rx(i)) return false;
    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = ib_port;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE;
    if (ibv_modify_qp(conn->qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) return false;
    union ibv_gid local_gid;
    if (ibv_query_gid(conn->ctx, ib_port, gid_idx, &local_gid) != 0) return false;
    srand((unsigned)time(nullptr));
    rdma_local.qpn = conn->qp->qp_num;
    rdma_local.psn = rand() & 0xffffff;
    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;
    memcpy(rdma_local.gid, local_gid.raw, RDMA_GID_SIZE);
    LOG_DBG("RDMA probe ok: dev=%s port=%d gid_idx=%d (%s) qpn=0x%x psn=0x%x inline=%u\n", matched_dev, ib_port, gid_idx, gid_version == IBV_GID_TYPE_ROCE_V2 ? "RoCEv2" : "RoCEv1/IB", rdma_local.qpn, rdma_local.psn, conn->max_inline);
    rdma = std::move(conn);
    return true;
}

bool tcp_rpc_transport::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, const uint8_t * remote_gid) {
    if (!rdma || !rdma->qp) return false;
    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = rdma_local.path_mtu;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = remote_psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = rdma_local.ib_port;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.grh.sgid_index = (uint8_t)rdma_local.gid_idx;
    memcpy(attr.ah_attr.grh.dgid.raw, remote_gid, RDMA_GID_SIZE);
    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    if (ibv_modify_qp(rdma->qp, &attr, flags) != 0) { LOG_DBG("RDMA activate: modify_qp -> RTR failed\n"); return false; }
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = rdma_local.psn;
    attr.max_rd_atomic = 1;
    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    if (ibv_modify_qp(rdma->qp, &attr, flags) != 0) { LOG_DBG("RDMA activate: modify_qp -> RTS failed\n"); return false; }
    LOG_DBG("RDMA connected: local QPN 0x%x -> remote QPN 0x%x\n", rdma->qp->qp_num, remote_qpn);
    return true;
}

bool tcp_rpc_transport::rdma_poll(struct ibv_cq * cq, struct ibv_wc * wc) {
    int n;
    while ((n = ibv_poll_cq(cq, 1, wc)) == 0) { if (tcp_peer_closed()) { LOG_DBG("RDMA poll: peer closed TCP socket while polling CQ\n"); return false; } }
    if (n < 0 || wc->status != IBV_WC_SUCCESS) { LOG_DBG("RDMA CQ error: status=%d (%s)\n", wc->status, ibv_wc_status_str(wc->status)); return false; }
    return true;
}

bool tcp_rpc_transport::rdma_send(const void * data, size_t size) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    size_t remaining = size;
    while (remaining > 0) {
        size_t chunk = std::min(remaining, RDMA_CHUNK);
        bool use_inline = (chunk <= rdma->max_inline);
        struct ibv_sge sge = {};
        sge.length = (uint32_t)chunk;
        if (use_inline) { sge.addr = (uintptr_t)p; sge.lkey = 0; }
        else { memcpy(rdma->tx_buf, p, chunk); sge.addr = (uintptr_t)rdma->tx_buf; sge.lkey = rdma->tx_mr->lkey; }
        struct ibv_send_wr wr = {}, * bad = nullptr;
        wr.wr_id = 1;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED | (use_inline ? IBV_SEND_INLINE : 0);
        if (ibv_post_send(rdma->qp, &wr, &bad) != 0) { LOG_DBG("ibv_post_send failed\n"); return false; }
        struct ibv_wc wc = {};
        if (!rdma_poll(rdma->scq, &wc)) return false;
        p += chunk; remaining -= chunk;
    }
    return true;
}

bool tcp_rpc_transport::rdma_recv(void * data, size_t size) {
    uint8_t * p = static_cast<uint8_t *>(data);
    size_t remaining = size;
    while (remaining > 0) {
        struct ibv_wc wc = {};
        if (!rdma_poll(rdma->rcq, &wc)) return false;
        int slot = (int)wc.wr_id;
        size_t got = wc.byte_len;
        if (got > remaining) { LOG_DBG("RDMA recv: received %zu bytes but only %zu expected\n", got, remaining); return false; }
        memcpy(p, rdma->rx_slot(slot), got);
        p += got; remaining -= got;
        if (!rdma->post_rx(slot)) { LOG_DBG("RDMA recv: failed to repost recv buffer\n"); return false; }
    }
    return true;
}
#endif
#endif

bool tcp_rpc_transport::send_exact(const void * data, size_t size) {
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) return rdma->send(data, size);
#elif defined(GGML_RPC_RDMA)
    if (use_rdma) return rdma_send(data, size);
#endif
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, (size_t)1024*1024);
        ssize_t n = send(fd, (const char *)data + bytes_sent, size_to_send, 0);
        if (n < 0) return false;
        bytes_sent += (size_t)n;
    }
    return true;
}

bool tcp_rpc_transport::recv_exact(void * data, size_t size) {
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) return rdma->recv(data, size);
#elif defined(GGML_RPC_RDMA)
    if (use_rdma) return rdma_recv(data, size);
#endif
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, (size_t)1024*1024);
        ssize_t n = recv(fd, (char *)data + bytes_recv, size_to_recv, 0);
        if (n <= 0) return false;
        bytes_recv += (size_t)n;
    }
    return true;
}

void tcp_rpc_transport::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    if (std::getenv("GGML_RPC_NO_RDMA")) return;
#  ifdef GGML_RPC_RDMA_APPLE
    auto target_gid = rdma_build_target_gid();
    if (target_gid) rdma = apple_rdma::probe(fd, target_gid->data(), local_caps);
#  else
    if (rdma_probe()) {
        rdma_caps rc = {rdma_local.qpn, rdma_local.psn};
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, sizeof(rc));
    }
#  endif
#endif
}

void tcp_rpc_transport::update_caps(const uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    bool remote_rdma = false;
    for (size_t i = 0; i < RPC_CONN_CAPS_SIZE; i++) remote_rdma |= remote_caps[i] != 0;
    if (!rdma || !remote_rdma) { rdma.reset(); return; }
#  ifdef GGML_RPC_RDMA_APPLE
    if (rdma->activate(remote_caps)) use_rdma = true;
#  else
    rdma_caps rc; memcpy(&rc, remote_caps, sizeof(rc));
    if (rdma_activate(rc.qpn, rc.psn, rc.gid)) use_rdma = true;
#  endif
    else rdma.reset();
#endif
}

bool tcp_rpc_transport::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) return rdma->flush();
#endif
    return true;
}

std::shared_ptr<rpc_transport> tcp_rpc_transport::accept() {
    auto client_socket_fd = ::accept(fd, NULL, NULL);
    if (!is_valid_fd(client_socket_fd)) return nullptr;
    set_no_delay(client_socket_fd);
    return std::make_shared<tcp_rpc_transport>(client_socket_fd);
}

#ifndef _WIN32
#include <fcntl.h>
#include <poll.h>

static bool stream_write_exact(int fd, const void * buf, size_t size) {
    const uint8_t * p = static_cast<const uint8_t *>(buf);
    size_t total = 0;
    alignas(4096) static thread_local uint8_t io_tx_page_buf[65536];
    int retries = 0;

    while (total < size) {
        size_t to_write = std::min(size - total, sizeof(io_tx_page_buf));
        memcpy(io_tx_page_buf, p + total, to_write);
        errno = 0;
        ssize_t n = ::write(fd, io_tx_page_buf, to_write);
        if (n > 0) {
            total += (size_t)n;
            retries = 0;
            continue;
        }
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n == 0 || (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == ENOBUFS || errno == ENOMEM))) {
            if (++retries < 100) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(5));
            }
            continue;
        }
        GGML_LOG_ERROR("stream_write_exact failed: fd=%d, n=%zd, total=%zu, size=%zu, errno=%d (%s)\n",
                       fd, n, total, size, errno, strerror(errno));
        return false;
    }
    return true;
}

static bool stream_read_exact(int fd, void * buf, size_t size) {
    uint8_t * p = static_cast<uint8_t *>(buf);
    size_t total = 0;
    while (total < size) {
        errno = 0;
        ssize_t n = ::read(fd, p + total, size - total);
        if (n > 0) {
            total += (size_t)n;
            continue;
        }
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n == 0) {
            std::this_thread::yield();
            continue;
        }
        GGML_LOG_ERROR("stream_read_exact failed: fd=%d, n=%zd, total=%zu, size=%zu, errno=%d (%s)\n",
                       fd, n, total, size, errno, strerror(errno));
        return false;
    }
    return true;
}
#endif

class stream_rpc_transport : public rpc_transport, public std::enable_shared_from_this<stream_rpc_transport> {
private:
    int data_fd = -1;
    int ctrl_fd = -1;
    bool is_listener = false;
    std::string data_path;
    std::string ctrl_path;
    std::mutex data_send_mu;
    std::mutex data_recv_mu;
    std::mutex ctrl_send_mu;
    std::mutex ctrl_recv_mu;

    std::thread stripe_worker;
    std::mutex stripe_mu;
    std::condition_variable stripe_cv_in;
    std::condition_variable stripe_cv_out;
    std::function<bool()> stripe_fn;
    bool stripe_has_work = false;
    bool stripe_done = false;
    bool stripe_res = false;
    bool stripe_stop = false;

#ifdef GGML_TBSTRIPE
    tbs_pipe * tbs_pipe_handle = nullptr;
#endif

    void start_stripe_worker() {
#ifndef _WIN32
        if (data_fd != ctrl_fd && data_fd >= 0 && ctrl_fd >= 0 && !stripe_worker.joinable()) {
            stripe_stop = false;
            stripe_has_work = false;
            stripe_done = false;
            stripe_worker = std::thread([this]() {
                std::unique_lock<std::mutex> lock(stripe_mu);
                while (!stripe_stop) {
                    stripe_cv_in.wait(lock, [this]() { return stripe_has_work || stripe_stop; });
                    if (stripe_stop) break;
                    stripe_has_work = false;
                    lock.unlock();

                    bool r = stripe_fn ? stripe_fn() : false;

                    lock.lock();
                    stripe_res = r;
                    stripe_done = true;
                    stripe_cv_out.notify_one();
                }
            });
        }
#endif
    }

    void stop_stripe_worker() {
#ifndef _WIN32
        {
            std::lock_guard<std::mutex> lock(stripe_mu);
            stripe_stop = true;
            stripe_has_work = false;
            stripe_cv_in.notify_all();
        }
        if (stripe_worker.joinable()) {
            stripe_worker.join();
        }
#endif
    }

    bool run_striped_op(std::function<bool()> fn0, std::function<bool()> fn1) {
#ifndef _WIN32
        if (!stripe_worker.joinable()) {
            return fn0() && fn1();
        }
        {
            std::lock_guard<std::mutex> lock(stripe_mu);
            stripe_fn = std::move(fn0);
            stripe_done = false;
            stripe_has_work = true;
            stripe_cv_in.notify_one();
        }
        bool r1 = fn1();
        {
            std::unique_lock<std::mutex> lock(stripe_mu);
            stripe_cv_out.wait(lock, [this]() { return stripe_done; });
            return stripe_res && r1;
        }
#else
        return fn0() && fn1();
#endif
    }

    struct rx_stream_buffer {
        std::vector<uint8_t> raw_buf;
        size_t raw_head = 0;
        size_t raw_tail = 0;

        std::vector<uint8_t> msg_buf;
        size_t msg_offset = 0;

        rx_stream_buffer() : raw_buf(524288) {}
        void clear() {
            msg_buf.clear();
            msg_offset = 0;
            raw_head = raw_tail = 0;
        }
    };
    rx_stream_buffer data_rx_buf;
    rx_stream_buffer ctrl_rx_buf;

    rx_stream_buffer & get_rx_buf(uint32_t channel_id) {
        if (data_fd == ctrl_fd) return data_rx_buf;
        return (channel_id == RPC_CHANNEL_DATA) ? data_rx_buf : ctrl_rx_buf;
    }

    static inline bool tb_debug_enabled() {
        static int enabled = -1;
        if (enabled == -1) {
            const char * e = std::getenv("GGML_TB_DEBUG");
            enabled = (e && strcmp(e, "0") != 0 && strcmp(e, "") != 0) ? 1 : 0;
        }
        return enabled == 1;
    }

    static inline void tb_delay_us(int us) {
        auto start = std::chrono::steady_clock::now();
        int64_t target_ns = (int64_t)us * 1000;
        while (std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - start).count() < target_ns) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
    }

    static bool buffered_stream_read_exact(int fd, rx_stream_buffer & sbuf, void * dst, size_t size) {
        uint8_t * out = static_cast<uint8_t *>(dst);
        size_t needed = size;
        if (tb_debug_enabled()) {
            fprintf(stderr, "[RX START fd=%d needed=%zu msg_avail=%zu raw_avail=%zu]\n",
                    fd, needed, sbuf.msg_buf.size() - sbuf.msg_offset, sbuf.raw_tail - sbuf.raw_head);
        }

        while (needed > 0) {
            // 1. Deliver from msg_buf if available
            size_t available = sbuf.msg_buf.size() - sbuf.msg_offset;
            if (available > 0) {
                size_t to_copy = std::min(needed, available);
                memcpy(out, sbuf.msg_buf.data() + sbuf.msg_offset, to_copy);
                sbuf.msg_offset += to_copy;
                out += to_copy;
                needed -= to_copy;
                if (sbuf.msg_offset == sbuf.msg_buf.size()) {
                    sbuf.msg_buf.clear();
                    sbuf.msg_offset = 0;
                }
                if (needed == 0) return true;
            }

            // 2. Unpack complete frames directly from raw_buf to out or msg_buf
            while (sbuf.raw_tail - sbuf.raw_head >= 2048) {
                uint32_t chunk_len = 0;
                memcpy(&chunk_len, sbuf.raw_buf.data() + sbuf.raw_head, 4);
                if (chunk_len > 2044) {
                    GGML_LOG_ERROR("corrupted frame header: %u > 2044\n", chunk_len);
                    return false;
                }
                const uint8_t * payload = sbuf.raw_buf.data() + sbuf.raw_head + 4;
                if (needed > 0 && chunk_len > 0) {
                    size_t to_out = std::min(needed, (size_t)chunk_len);
                    memcpy(out, payload, to_out);
                    out += to_out;
                    needed -= to_out;
                    payload += to_out;
                    chunk_len -= (uint32_t)to_out;
                }
                if (chunk_len > 0) {
                    size_t old_sz = sbuf.msg_buf.size();
                    sbuf.msg_buf.resize(old_sz + chunk_len);
                    memcpy(sbuf.msg_buf.data() + old_sz, payload, chunk_len);
                }
                sbuf.raw_head += 2048;
                if (sbuf.raw_head == sbuf.raw_tail) {
                    sbuf.raw_head = sbuf.raw_tail = 0;
                }
            }

            if (needed == 0) return true;

            // 3. Compact raw_buf before reading more
            if (sbuf.raw_head > 0) {
                size_t rem = sbuf.raw_tail - sbuf.raw_head;
                if (rem > 0) {
                    memmove(sbuf.raw_buf.data(), sbuf.raw_buf.data() + sbuf.raw_head, rem);
                }
                sbuf.raw_head = 0;
                sbuf.raw_tail = rem;
            }

            // Read from kernel using page-aligned buffer
            alignas(4096) static thread_local uint8_t io_page_buf[65536];
            errno = 0;
            ssize_t n = ::read(fd, io_page_buf, sizeof(io_page_buf));
            if (n > 0) {
                if (tb_debug_enabled()) {
                    fprintf(stderr, "  [RX READ fd=%d n=%zd raw_tail_was=%zu needed=%zu]\n",
                            fd, n, sbuf.raw_tail, needed);
                }
                if (sbuf.raw_buf.size() < sbuf.raw_tail + (size_t)n) {
                    sbuf.raw_buf.resize(sbuf.raw_tail + (size_t)n + 262144);
                }
                memcpy(sbuf.raw_buf.data() + sbuf.raw_tail, io_page_buf, (size_t)n);
                sbuf.raw_tail += (size_t)n;
                continue;
            }
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                std::this_thread::yield();
                continue;
            }
            if (n == 0) {
                GGML_LOG_ERROR("buffered_stream_read_exact: EOF from peer (fd=%d)\n", fd);
                return false;
            }
            GGML_LOG_ERROR("buffered_stream_read_exact read failed: fd=%d, errno=%d (%s)\n", fd, errno, strerror(errno));
            return false;
        }
        if (tb_debug_enabled()) {
            fprintf(stderr, "[RX DONE fd=%d size=%zu rem_msg=%zu rem_raw=%zu]\n",
                    fd, size, sbuf.msg_buf.size() - sbuf.msg_offset, sbuf.raw_tail - sbuf.raw_head);
        }
        return true;
    }

public:
    bool init_tbstripe() {
#ifdef GGML_TBSTRIPE
        if (data_path.empty() || data_path == ctrl_path) {
            return false;
        }
        tbs_config cfg = {};
        cfg.dev_a = data_path.c_str();
        cfg.dev_b = ctrl_path.c_str();
        cfg.stripe = TBS_FRAME_SIZE;
        cfg.busy_spin = 1;
        cfg.cpu_affinity_a = 8;
        cfg.cpu_affinity_b = 9;
        tbs_pipe_handle = tbs_open(&cfg);
        if (!tbs_pipe_handle) {
            GGML_LOG_ERROR("stream_rpc_transport: tbs_open failed for %s,%s\n",
                           data_path.c_str(), ctrl_path.c_str());
            return false;
        }
        GGML_LOG_INFO("stream_rpc_transport: tbstripe pipe on %s + %s (striped=%d)\n",
                      data_path.c_str(), ctrl_path.c_str(), tbs_is_striped(tbs_pipe_handle));
        return true;
#else
        return false;
#endif
    }

    void release_raw_fds() {
#ifndef _WIN32
        if (ctrl_fd >= 0 && ctrl_fd != data_fd) {
            ::close(ctrl_fd);
        }
        if (data_fd >= 0) {
            ::close(data_fd);
        }
        data_fd = -1;
        ctrl_fd = -1;
#endif
    }

    stream_rpc_transport(int dfd, int cfd, const std::string & dpath, const std::string & cpath, bool listener)
        : data_fd(dfd), ctrl_fd(cfd), is_listener(listener), data_path(dpath), ctrl_path(cpath) {
        // Drop constructor FDs first so tbstripe is the only opener.
        release_raw_fds();
        if (init_tbstripe()) {
            return;
        }
        data_fd = open_stream_device(data_path);
        if (data_path == ctrl_path) {
            ctrl_fd = data_fd;
        } else if (data_fd >= 0) {
            ctrl_fd = open_stream_device(ctrl_path);
        }
        start_stripe_worker();
    }

    ~stream_rpc_transport() override {
#ifdef GGML_TBSTRIPE
        if (tbs_pipe_handle) {
            tbs_close(tbs_pipe_handle);
            tbs_pipe_handle = nullptr;
        }
#endif
        stop_stripe_worker();
        if (!is_listener) {
            close();
        }
    }

    int get_fd_for_channel(uint32_t channel_id) const { return (channel_id == RPC_CHANNEL_DATA) ? data_fd : ctrl_fd; }
    std::mutex & get_send_mutex(uint32_t channel_id) {
        if (data_fd == ctrl_fd) return data_send_mu;
        return (channel_id == RPC_CHANNEL_DATA) ? data_send_mu : ctrl_send_mu;
    }
    std::mutex & get_recv_mutex(uint32_t channel_id) {
        if (data_fd == ctrl_fd) return data_recv_mu;
        return (channel_id == RPC_CHANNEL_DATA) ? data_recv_mu : ctrl_recv_mu;
    }

    bool send_exact(const void * data, size_t size) override {
#ifdef GGML_TBSTRIPE
        if (tbs_pipe_handle) {
            if (size == 0 || data == nullptr) return true;
            return tbs_send(tbs_pipe_handle, data, size) == 0;
        }
#endif
        return send_exact_channel(RPC_CHANNEL_CONTROL, data, size);
    }
    bool recv_exact(void * data, size_t size) override {
#ifdef GGML_TBSTRIPE
        if (tbs_pipe_handle) {
            if (size == 0 || data == nullptr) return true;
            return tbs_recv(tbs_pipe_handle, data, size) == 0;
        }
#endif
        return recv_exact_channel(RPC_CHANNEL_CONTROL, data, size);
    }

    static int open_stream_device(const std::string & path, int max_wait_sec = 10) {
        auto start = std::chrono::steady_clock::now();
        while (true) {
            int fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK);
            if (fd >= 0) {
                int flags = fcntl(fd, F_GETFL, 0);
                if (flags != -1) {
                    fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
                }
                return fd;
            }
            if (errno != ENXIO && errno != ENOENT && errno != EAGAIN && errno != EBUSY) {
                GGML_LOG_ERROR("Failed to open stream device '%s': %s\n", path.c_str(), strerror(errno));
                return -1;
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= max_wait_sec) {
                GGML_LOG_ERROR("Timeout waiting for stream device '%s' (waited %d s): %s\n",
                               path.c_str(), max_wait_sec, strerror(errno));
                return -1;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    bool send_exact_channel(uint32_t channel_id, const void * data, size_t size) override {
#ifndef _WIN32
        if (size == 0 || data == nullptr) return true;
#ifdef GGML_TBSTRIPE
        if (tbs_pipe_handle) {
            (void)channel_id;
            return tbs_send(tbs_pipe_handle, data, size) == 0;
        }
#endif
        if (channel_id == RPC_CHANNEL_STRIPED) {
            if (data_fd == ctrl_fd || size < 2048) {
                return send_exact_channel(RPC_CHANNEL_CONTROL, data, size);
            }
            size_t half0 = size / 2;
            size_t half1 = size - half0;
            return run_striped_op(
                [this, data, half0]() {
                    return send_exact_channel(RPC_CHANNEL_DATA, data, half0);
                },
                [this, data, half0, half1]() {
                    return send_exact_channel(RPC_CHANNEL_CONTROL, (const uint8_t*)data + half0, half1);
                }
            );
        }
        int fd = get_fd_for_channel(channel_id);
        std::lock_guard<std::mutex> lock(get_send_mutex(channel_id));

        const uint8_t * p = static_cast<const uint8_t *>(data);
        size_t total = 0;

        if (tb_debug_enabled()) {
            fprintf(stderr, "[TX START ch=%u fd=%d size=%zu]\n", channel_id, fd, size);
        }

        // Batch buffer: up to 32 frames of 2048 bytes = 65536 bytes
        alignas(4096) uint8_t batch_buf[65536];

        while (total < size) {
            size_t batch_bytes = 0;
            while (total < size && batch_bytes + 2048 <= sizeof(batch_buf)) {
                size_t chunk_len = std::min(size - total, (size_t)2044);
                uint32_t header = (uint32_t)chunk_len;

                uint8_t * frame = batch_buf + batch_bytes;
                memset(frame, 0, 2048);
                memcpy(frame, &header, 4);
                memcpy(frame + 4, p + total, chunk_len);

                total += chunk_len;
                batch_bytes += 2048;
            }

            if (batch_bytes > 0) {
                if (!stream_write_exact(fd, batch_buf, batch_bytes)) {
                    return false;
                }
            }
        }

        // Append 2048-byte flush trailer frame (header = 0) to flush trailing DMA descriptors
        {
            alignas(4096) uint8_t flush_frame[2048];
            memset(flush_frame, 0, sizeof(flush_frame));
            if (!stream_write_exact(fd, flush_frame, sizeof(flush_frame))) {
                return false;
            }
        }

        if (tb_debug_enabled()) {
            fprintf(stderr, "[TX DONE ch=%u fd=%d size=%zu]\n", channel_id, fd, size);
        }
        return true;
#else
        (void)channel_id; (void)data; (void)size;
        return false;
#endif
    }
    bool recv_exact_channel(uint32_t channel_id, void * data, size_t size) override {
#ifndef _WIN32
        if (size == 0 || data == nullptr) return true;
#ifdef GGML_TBSTRIPE
        if (tbs_pipe_handle) {
            (void)channel_id;
            return tbs_recv(tbs_pipe_handle, data, size) == 0;
        }
#endif
        if (channel_id == RPC_CHANNEL_STRIPED) {
            if (data_fd == ctrl_fd || size < 2048) {
                return recv_exact_channel(RPC_CHANNEL_CONTROL, data, size);
            }
            size_t half0 = size / 2;
            size_t half1 = size - half0;
            return run_striped_op(
                [this, data, half0]() {
                    return recv_exact_channel(RPC_CHANNEL_DATA, data, half0);
                },
                [this, data, half0, half1]() {
                    return recv_exact_channel(RPC_CHANNEL_CONTROL, (uint8_t*)data + half0, half1);
                }
            );
        }
        int fd = get_fd_for_channel(channel_id);
        std::lock_guard<std::mutex> lock(get_recv_mutex(channel_id));
        return buffered_stream_read_exact(fd, get_rx_buf(channel_id), data, size);
#else
        (void)channel_id; (void)data; (void)size;
        return false;
#endif
    }
    bool recv_cmd(uint8_t * cmd, uint32_t * out_channel) override {
#ifndef _WIN32
        if (out_channel) *out_channel = RPC_CHANNEL_CONTROL;
#ifdef GGML_TBSTRIPE
        if (tbs_pipe_handle) {
            return recv_exact(cmd, 1);
        }
#endif
        return recv_exact_channel(RPC_CHANNEL_CONTROL, cmd, 1);
#else
        (void)cmd; (void)out_channel;
        return false;
#endif
    }
    bool flush() override {
        return true;
    }
    void close() override {
#ifndef _WIN32
        stop_stripe_worker();
        if (ctrl_fd >= 0 && ctrl_fd != data_fd) ::close(ctrl_fd);
        if (data_fd >= 0) ::close(data_fd);
        ctrl_fd = data_fd = -1;
#endif
    }
    bool is_first_accept = true;
    std::shared_ptr<rpc_transport> accept() override {
        if (!is_listener) return nullptr;
#ifndef _WIN32
        // USB4STREAM is a persistent pipe, not a TCP listen socket. Re-opening
        // sends CLOSE/EOF and desynchronizes the peer. Keep the same tbstripe
        // handle for the life of the process.
#ifdef GGML_TBSTRIPE
        if (tbs_pipe_handle) {
            if (!is_first_accept) {
                // Previous client closed the USB4STREAM (CLOSE/EOF). Reopen
                // once, with a short pause so we do not spin at 100% CPU.
                tbs_close(tbs_pipe_handle);
                tbs_pipe_handle = nullptr;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (!init_tbstripe()) {
                    return nullptr;
                }
            }
            is_first_accept = false;
            return shared_from_this();
        }
#endif
        if (!is_first_accept) {
            stop_stripe_worker();
            release_raw_fds();
            data_rx_buf.clear();
            ctrl_rx_buf.clear();
            if (init_tbstripe()) {
                is_first_accept = false;
                return shared_from_this();
            }
            data_fd = open_stream_device(data_path);
            if (data_fd < 0) {
                GGML_LOG_ERROR("Failed to re-open data stream device '%s': %s\n", data_path.c_str(), strerror(errno));
                return nullptr;
            }
            if (data_path == ctrl_path) {
                ctrl_fd = data_fd;
            } else {
                ctrl_fd = open_stream_device(ctrl_path);
                if (ctrl_fd < 0) {
                    GGML_LOG_ERROR("Failed to re-open control stream device '%s': %s\n", ctrl_path.c_str(), strerror(errno));
                    ::close(data_fd);
                    data_fd = -1;
                    return nullptr;
                }
            }
            start_stripe_worker();
        }
        is_first_accept = false;
        return shared_from_this();
#else
        return nullptr;
#endif
    }
    bool is_stream() const override { return true; }
    bool xchg(const void * out, void * in, size_t n) override {
#ifdef GGML_TBSTRIPE
        if (tbs_pipe_handle) {
            if (n == 0) {
                return true;
            }
            if (out == nullptr || in == nullptr) {
                return false;
            }
            return tbs_xchg(tbs_pipe_handle, out, in, n) == 0;
        }
#else
        (void)out;
        (void)in;
        (void)n;
#endif
        return false;
    }
    static std::shared_ptr<stream_rpc_transport> open_stream(const std::string & data_path, const std::string & ctrl_path, bool is_server = false) {
#ifndef _WIN32
        // Do not open the char devices here. Closing a USB4STREAM fd sends a
        // CLOSE/EOF frame to the peer. tbstripe (or the constructor fallback)
        // must be the only opener for the session.
        return std::make_shared<stream_rpc_transport>(-1, -1, data_path, ctrl_path, is_server);
#else
        (void)data_path; (void)ctrl_path; (void)is_server;
        return nullptr;
#endif
    }
};

socket_t::socket_t(rpc_transport_ptr transport) : transport(std::move(transport)) {}
socket_t::~socket_t() = default;
bool socket_t::send_data(const void * data, size_t size) { return transport && transport->send_exact(data, size); }
bool socket_t::recv_data(void * data, size_t size) { return transport && transport->recv_exact(data, size); }
bool socket_t::send_data_channel(uint32_t channel_id, const void * data, size_t size) { return transport && transport->send_exact_channel(channel_id, data, size); }
bool socket_t::recv_data_channel(uint32_t channel_id, void * data, size_t size) { return transport && transport->recv_exact_channel(channel_id, data, size); }
bool socket_t::recv_cmd(uint8_t * cmd, uint32_t * out_channel) { return transport && transport->recv_cmd(cmd, out_channel); }
bool socket_t::flush() { return transport && transport->flush(); }
bool socket_t::xchg(const void * out, void * in, size_t n) { return transport && transport->xchg(out, in, n); }
std::shared_ptr<socket_t> socket_t::accept() {
    auto child = transport ? transport->accept() : nullptr;
    return child ? std::make_shared<socket_t>(child) : nullptr;
}
void socket_t::get_caps(uint8_t * local_caps) { if (transport) transport->get_caps(local_caps); else memset(local_caps, 0, RPC_CONN_CAPS_SIZE); }
void socket_t::update_caps(const uint8_t * remote_caps) { if (transport) transport->update_caps(remote_caps); }
bool socket_t::is_stream() const { return transport && transport->is_stream(); }

socket_ptr socket_t::create_server(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid_fd(sockfd)) return nullptr;
    set_reuse_addr(sockfd);
    struct sockaddr_in serv = {AF_INET, htons(port), {inet_addr(host)}};
    if (bind(sockfd, (struct sockaddr *)&serv, sizeof(serv)) < 0 || listen(sockfd, 1) < 0) return nullptr;
    return std::make_shared<socket_t>(std::make_shared<tcp_rpc_transport>(sockfd));
}

socket_ptr socket_t::connect(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid_fd(sockfd)) return nullptr;
    set_no_delay(sockfd);
    struct hostent * h = gethostbyname(host);
    if (!h) return nullptr;
    struct sockaddr_in addr = {AF_INET, htons(port)};
    memcpy(&addr.sin_addr, h->h_addr, h->h_length);
    if (::connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) return nullptr;
    return std::make_shared<socket_t>(std::make_shared<tcp_rpc_transport>(sockfd));
}

bool socket_t::parse_endpoint(const std::string & endpoint, rpc_endpoint_info & out) {
    out.raw_endpoint = endpoint;
    std::string ep = endpoint;
    if (ep.rfind("dev://", 0) == 0 || ep.rfind("stream://", 0) == 0 || ep.find("/dev/") == 0) {
        out.kind = rpc_transport_kind::STREAM;
        if (ep.find("dev://") == 0) ep = ep.substr(6);
        else if (ep.find("stream://") == 0) ep = ep.substr(9);
        size_t comma = ep.find(',');
        out.data_path = (comma != std::string::npos) ? ep.substr(0, comma) : ep;
        out.ctrl_path = (comma != std::string::npos) ? ep.substr(comma + 1) : ep;
        return true;
    }
    out.kind = rpc_transport_kind::TCP;
    size_t pos = ep.rfind(':');
    if (pos == std::string::npos) return false;
    out.host = ep.substr(0, pos);
    try { out.port = std::stoi(ep.substr(pos + 1)); } catch (...) { return false; }
    return out.port > 0 && out.port <= 65535;
}

socket_ptr socket_t::create_server_endpoint(const char * endpoint) {
    rpc_endpoint_info info;
    if (!parse_endpoint(endpoint, info)) return nullptr;
    if (info.kind == rpc_transport_kind::STREAM) {
        auto tp = stream_rpc_transport::open_stream(info.data_path, info.ctrl_path, true);
        return tp ? std::make_shared<socket_t>(tp) : nullptr;
    }
    return create_server(info.host.c_str(), info.port);
}

socket_ptr socket_t::connect_endpoint(const char * endpoint) {
    rpc_endpoint_info info;
    if (!parse_endpoint(endpoint, info)) return nullptr;
    if (info.kind == rpc_transport_kind::STREAM) {
        auto tp = stream_rpc_transport::open_stream(info.data_path, info.ctrl_path, false);
        return tp ? std::make_shared<socket_t>(tp) : nullptr;
    }
    return connect(info.host.c_str(), info.port);
}

#ifdef _WIN32
static std::mutex g_rpc_transport_mu;
static bool g_rpc_transport_wsa_started = false;
#endif

bool rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    if (g_rpc_transport_wsa_started) return true;
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) return false;
    g_rpc_transport_wsa_started = true;
#endif
    return true;
}

void rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    if (g_rpc_transport_wsa_started) { WSACleanup(); g_rpc_transport_wsa_started = false; }
#endif
}
