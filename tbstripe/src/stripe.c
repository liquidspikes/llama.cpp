#define _GNU_SOURCE
#include "tbstripe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/uio.h>
#include <pthread.h>
#include <sched.h>
#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#endif

#define UIO_BATCH_MAX 1024

static inline void tbs_spin_pause(void) {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#else
    sched_yield();
#endif
}

/*
 * Spin-wait writev: handles non-blocking busy_poll fds on Linux 7.3.
 * Continuously writes until all vectors are exhausted or a fatal error occurs.
 */
ssize_t tbs_writev_spin(int fd, struct iovec *iov, int iovcnt, int busy_spin) {
    if (iovcnt <= 0) return 0;
    
    size_t total_bytes = 0;
    for (int i = 0; i < iovcnt; i++) {
        total_bytes += iov[i].iov_len;
    }
    if (total_bytes == 0) return 0;

    int cur_idx = 0;
    size_t written_total = 0;

    while (cur_idx < iovcnt) {
        int batch_cnt = iovcnt - cur_idx;
        if (batch_cnt > UIO_BATCH_MAX) {
            batch_cnt = UIO_BATCH_MAX;
        }

        ssize_t nw = writev(fd, &iov[cur_idx], batch_cnt);
        if (nw > 0) {
            written_total += nw;
            size_t rem = (size_t)nw;
            while (rem > 0 && cur_idx < iovcnt) {
                if (rem >= iov[cur_idx].iov_len) {
                    rem -= iov[cur_idx].iov_len;
                    iov[cur_idx].iov_len = 0;
                    cur_idx++;
                } else {
                    iov[cur_idx].iov_base = (char *)iov[cur_idx].iov_base + rem;
                    iov[cur_idx].iov_len -= rem;
                    rem = 0;
                }
            }
            continue;
        }

        if (nw < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                if (busy_spin) {
                    tbs_spin_pause();
                } else {
                    sched_yield();
                }
                continue;
            }
            return -1;
        }

        /* nw == 0: peer or stream device returned 0 */
        tbs_spin_pause();
    }

    return (ssize_t)written_total;
}

/*
 * Spin-wait readv: handles non-blocking busy_poll fds on Linux 7.3.
 * Continuously reads until all vectors are exhausted or a fatal error occurs.
 */
ssize_t tbs_readv_spin(int fd, struct iovec *iov, int iovcnt, int busy_spin) {
    if (iovcnt <= 0) return 0;

    size_t total_bytes = 0;
    for (int i = 0; i < iovcnt; i++) {
        total_bytes += iov[i].iov_len;
    }
    if (total_bytes == 0) return 0;

    int cur_idx = 0;
    size_t read_total = 0;

    while (cur_idx < iovcnt) {
        int batch_cnt = iovcnt - cur_idx;
        if (batch_cnt > UIO_BATCH_MAX) {
            batch_cnt = UIO_BATCH_MAX;
        }

        ssize_t nr = readv(fd, &iov[cur_idx], batch_cnt);
        if (nr > 0) {
            read_total += nr;
            size_t rem = (size_t)nr;
            while (rem > 0 && cur_idx < iovcnt) {
                if (rem >= iov[cur_idx].iov_len) {
                    rem -= iov[cur_idx].iov_len;
                    iov[cur_idx].iov_len = 0;
                    cur_idx++;
                } else {
                    iov[cur_idx].iov_base = (char *)iov[cur_idx].iov_base + rem;
                    iov[cur_idx].iov_len -= rem;
                    rem = 0;
                }
            }
            continue;
        }

        if (nr < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                if (busy_spin) {
                    tbs_spin_pause();
                } else {
                    sched_yield();
                }
                continue;
            }
            return -1;
        }

        if (nr == 0) {
            /* EOF from peer */
            return 0;
        }
    }

    return (ssize_t)read_total;
}
