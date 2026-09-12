#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Fast-path predicates for USB4 rpc-tensor SEQ decode.
// After the first T=1 graph is planned, walking 7k nodes to re-hash graph_sig
// and dummy-syncing RPC on every llama_get_logits are pure overhead.

static inline bool ggml_meta_seq_sig_reuse(bool plan_valid, int cached_n_nodes, int n_nodes,
        uint64_t cached_fp, uint64_t fp) {
    return plan_valid && cached_n_nodes == n_nodes && n_nodes > 0 && cached_fp == fp;
}

static inline uint64_t ggml_meta_seq_graph_fp(int n_nodes, uint64_t ne0, uint64_t ne1, uint64_t ne2, uint64_t op0) {
    return ((uint64_t) n_nodes * 0x9e3779b97f4a7c15ull) ^ ne0 ^ (ne1 << 20) ^ (ne2 << 40) ^ (op0 << 8);
}

static inline bool ggml_meta_seq_skip_rpc_sync(bool seq_just_completed) {
    return seq_just_completed;
}

// Skip bit stays set across llama_decode/sampler synchronizes until the next
// graph_compute. Tests: a completed SEQ skips; a fresh compute does not.

// Warm T=1 decode: SEQ plan is already filled. Skip lambda/aux/rebuild setup.
static inline bool ggml_meta_seq_can_fast(bool plan_valid, int cached_n_nodes, int n_nodes,
        uint64_t cached_fp, uint64_t fp, size_t n_subgraphs, size_t seq_gs_len) {
    return ggml_meta_seq_sig_reuse(plan_valid, cached_n_nodes, n_nodes, cached_fp, fp)
        && n_subgraphs >= 2 && seq_gs_len == n_subgraphs;
}

// Two SEQ plan slots so a T>1 prompt cannot drop the T=1 HIP/RPC uids.
// Slot 0 is decode (ne2==1); slot 1 is the latest prompt shape.
enum { GGML_META_SEQ_FP_SLOTS = 2 };

static inline int ggml_meta_seq_slot_pick(uint64_t ne2) {
    return ne2 == 1 ? 0 : 1;
}
