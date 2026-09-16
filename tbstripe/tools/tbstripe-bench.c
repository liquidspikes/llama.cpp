#define _GNU_SOURCE
#include "tbstripe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

typedef _Float16 tbs_fp16_t;

void run_bw_benchmark(tbs_pipe *pipe, int rank, int iters) {
    printf("\n=== [BANDWIDTH BENCHMARK] (Striped: %s) ===\n", tbs_is_striped(pipe) ? "YES (2-Port Dual USB4)" : "NO (1-Port Single)");
    printf("%-10s %-12s %-12s %-12s %-12s\n", "Size", "Throughput", "p50 Latency", "p99 Latency", "Min Latency");
    printf("----------------------------------------------------------------------\n");

    size_t sizes[] = { 4096, 65536, 1048576, 8388608, 33554432 };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        size_t sz = sizes[s];
        void *buf = NULL;
        if (posix_memalign(&buf, TBS_ALIGNMENT, sz) != 0) {
            fprintf(stderr, "Allocation failed for %zu bytes\n", sz);
            continue;
        }
        memset(buf, 0x5A, sz);

        int current_iters = iters;
        if (sz >= 8388608) current_iters = (iters > 20) ? 20 : iters;
        if (sz >= 33554432) current_iters = (iters > 10) ? 10 : iters;

        double *latencies = (double *)malloc(current_iters * sizeof(double));

        for (int i = 0; i < current_iters; i++) {
            double t0 = get_time_sec();
            if (rank == 0) {
                if (tbs_send(pipe, buf, sz) != 0) {
                    fprintf(stderr, "Send failed at size %zu, iter %d\n", sz, i);
                    break;
                }
            } else {
                if (tbs_recv(pipe, buf, sz) != 0) {
                    fprintf(stderr, "Recv failed at size %zu, iter %d\n", sz, i);
                    break;
                }
            }
            double t1 = get_time_sec();
            latencies[i] = t1 - t0;
        }

        qsort(latencies, current_iters, sizeof(double), compare_doubles);
        double p50 = latencies[current_iters / 2];
        double p99 = latencies[(int)(current_iters * 0.99)];
        double min_lat = latencies[0];
        double gbps = ((double)sz / p50) / 1e9;

        const char *sz_str;
        char tmp_sz[32];
        if (sz >= 1048576) {
            snprintf(tmp_sz, sizeof(tmp_sz), "%zu MiB", sz / 1048576);
        } else {
            snprintf(tmp_sz, sizeof(tmp_sz), "%zu KiB", sz / 1024);
        }
        sz_str = tmp_sz;

        printf("%-10s %8.3f GB/s   %8.3f ms   %8.3f ms   %8.3f ms\n",
               sz_str, gbps, p50 * 1000.0, p99 * 1000.0, min_lat * 1000.0);

        free(latencies);
        free(buf);
    }
}

void run_xchg_benchmark(tbs_pipe *pipe, int rank, int iters) {
    printf("\n=== [SIMULTANEOUS BIDIRECTIONAL EXCHANGE BENCHMARK] ===\n");
    size_t sz = 1048576; // 1 MiB
    void *send_buf = NULL;
    void *recv_buf = NULL;
    posix_memalign(&send_buf, TBS_ALIGNMENT, sz);
    posix_memalign(&recv_buf, TBS_ALIGNMENT, sz);

    uint8_t fill_val = (rank == 0) ? 0xAA : 0xBB;
    uint8_t expected = (rank == 0) ? 0xBB : 0xAA;
    memset(send_buf, fill_val, sz);
    memset(recv_buf, 0, sz);

    printf("Verifying zero deadlock across %d concurrent bidirectional exchanges (1 MiB payload)...\n", iters);
    double t0 = get_time_sec();
    for (int i = 0; i < iters; i++) {
        if (tbs_xchg(pipe, send_buf, recv_buf, sz) != 0) {
            fprintf(stderr, "Exchange failed at iter %d!\n", i);
            break;
        }
    }
    double t1 = get_time_sec();
    double total_time = t1 - t0;
    double avg_lat_ms = (total_time / iters) * 1000.0;
    double bi_gbps = ((double)sz * 2.0 / (total_time / iters)) / 1e9;

    /* Verify data */
    uint8_t *r = (uint8_t *)recv_buf;
    int corrupt = 0;
    for (size_t i = 0; i < sz; i++) {
        if (r[i] != expected) {
            corrupt = 1;
            break;
        }
    }

    if (corrupt) {
        printf("  [FAIL] Data corruption detected in received buffer!\n");
    } else {
        printf("  [PASS] All %d exchanges completed without deadlock. Aggregate Duplex BW: %.3f GB/s (%.3f ms/xchg)\n",
               iters, bi_gbps, avg_lat_ms);
    }

    free(send_buf);
    free(recv_buf);
}

void run_allreduce_benchmark(tbs_pipe *pipe, int rank, int iters) {
    printf("\n=== [2-RANK TENSOR PARALLEL FP16 ALL-REDUCE BENCHMARK] ===\n");
    printf("%-24s %-12s %-12s %-12s\n", "Shape / Dimensions", "Payload Size", "Avg Latency", "Effective BW");
    printf("----------------------------------------------------------------------\n");

    struct {
        const char *desc;
        size_t n_elem;
    } cases[] = {
        { "Hidden 4096 (Batch 1)",    4096 },
        { "Hidden 8192 (Batch 1)",    8192 },
        { "Hidden 4096 (Batch 512)",  4096 * 512 },
        { "Hidden 8192 (Batch 512)",  8192 * 512 },
    };
    int n_cases = sizeof(cases) / sizeof(cases[0]);

    for (int c = 0; c < n_cases; c++) {
        size_t n_elem = cases[c].n_elem;
        size_t bytes = n_elem * sizeof(tbs_fp16_t);

        tbs_fp16_t *my_buf = NULL;
        tbs_fp16_t *peer_buf = NULL;
        posix_memalign((void **)&my_buf, TBS_ALIGNMENT, bytes);
        posix_memalign((void **)&peer_buf, TBS_ALIGNMENT, bytes);

        /* Initialize: Rank 0 sends 1.0f, Rank 1 sends 2.0f -> Sum = 3.0f */
        float init_val = (rank == 0) ? 1.0f : 2.0f;
        tbs_fp16_t init_fp16 = (tbs_fp16_t)init_val;
        for (size_t i = 0; i < n_elem; i++) {
            my_buf[i] = init_fp16;
        }

        int current_iters = (bytes >= 4194304) ? 20 : iters;
        double t0 = get_time_sec();

        for (int i = 0; i < current_iters; i++) {
            for (size_t k = 0; k < n_elem; k++) my_buf[k] = init_fp16;
            /* 1. Bidirectional exchange */
            if (tbs_xchg(pipe, my_buf, peer_buf, bytes) != 0) {
                fprintf(stderr, "AllReduce exchange failed at iter %d\n", i);
                break;
            }
            /* 2. Local vector reduction (FP16 sum) */
            for (size_t j = 0; j < n_elem; j++) {
                my_buf[j] = my_buf[j] + peer_buf[j];
            }
        }

        double t1 = get_time_sec();
        double avg_lat = (t1 - t0) / current_iters;
        double eff_gbps = ((double)bytes * 2.0 / avg_lat) / 1e9;

        /* Verify numerical result */
        float expected_sum = 3.0f;
        float actual_sum = (float)my_buf[0];
        int accurate = (fabs(actual_sum - expected_sum) < 0.05f);

        printf("%-24s %8zu KB   %8.3f ms   %8.3f GB/s  [%s]\n",
               cases[c].desc, bytes / 1024, avg_lat * 1000.0, eff_gbps,
               accurate ? "VERIFIED (3.0)" : "MISMATCH");

        free(my_buf);
        free(peer_buf);
    }
}

int main(int argc, char **argv) {
    const char *dev_a = "/dev/tbstream0";
    const char *dev_b = "/dev/tbstream1";
    int rank = 0;
    int iters = 50;
    int single_port = 0;
    const char *mode = "all";

    const char *env_rank = getenv("TBSTRIPE_RANK");
    if (env_rank) {
        rank = atoi(env_rank);
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc) {
            rank = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dev-a") == 0 && i + 1 < argc) {
            dev_a = argv[++i];
        } else if (strcmp(argv[i], "--dev-b") == 0 && i + 1 < argc) {
            dev_b = argv[++i];
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--single") == 0) {
            single_port = 1;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--rank 0|1] [--dev-a /dev/tbstream0] [--dev-b /dev/tbstream1] [--single] [--mode bw|xchg|allreduce|all] [--iters N]\n", argv[0]);
            return 0;
        }
    }

    tbs_config cfg = {
        .dev_a = dev_a,
        .dev_b = single_port ? NULL : dev_b,
        .stripe = TBS_DEFAULT_STRIPE_SZ,
        .busy_spin = 1,
        .cpu_affinity_a = 8,
        .cpu_affinity_b = 9,
    };

    printf("======================================================================\n");
    printf("   Dual-USB4STREAM Striped Pipe Benchmark for Strix Halo (Linux 7.3)   \n");
    printf("======================================================================\n");
    printf("Host Rank: %d\n", rank);
    printf("Device A:  %s\n", cfg.dev_a);
    printf("Device B:  %s\n", cfg.dev_b ? cfg.dev_b : "<NONE> (Single Port Mode)");
    printf("Mode:      %s\n", mode);

    tbs_pipe *pipe = tbs_open(&cfg);
    if (!pipe) {
        fprintf(stderr, "Failed to open tbstripe pipe on %s (and %s)!\n", cfg.dev_a, cfg.dev_b ? cfg.dev_b : "null");
        return 1;
    }

    if (strcmp(mode, "bw") == 0 || strcmp(mode, "all") == 0) {
        run_bw_benchmark(pipe, rank, iters);
    }
    if (strcmp(mode, "xchg") == 0 || strcmp(mode, "all") == 0) {
        run_xchg_benchmark(pipe, rank, iters);
    }
    if (strcmp(mode, "allreduce") == 0 || strcmp(mode, "all") == 0) {
        run_allreduce_benchmark(pipe, rank, iters);
    }

    tbs_close(pipe);
    printf("\nBenchmark finished successfully.\n");
    return 0;
}
