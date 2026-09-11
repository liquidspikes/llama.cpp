#include "speculative.h"
#include "../src/llama-ext.h"
#include "../ggml/src/ggml-backend-meta-seq.h"

#include <cstdio>

// USB4 rpc-tensor verify ubatch T must stay fixed so Meta GRAPH_RECOMPUTE hits.
// common_spec_keep_drafting_usb4 is the shipped early-stop gate used by draft-mtp.

int main() {
    int fails = 0;

    auto check = [&](bool cond, const char * msg) {
        if (!cond) {
            fprintf(stderr, "FAIL: %s\n", msg);
            fails++;
        }
    };

    // Meta: pad short verify batches to n_max+1; leave T=1 and already-full alone
    check(common_spec_usb4_verify_n_tokens(1, 4, true) == 1,
          "T=1 decode is not padded");
    check(common_spec_usb4_verify_n_tokens(3, 4, true) == 5,
          "T=3 verify pads to n_max+1=5");
    check(common_spec_usb4_verify_n_tokens(5, 4, true) == 5,
          "already n_max+1 stays");
    check(common_spec_usb4_verify_n_tokens(3, 4, false) == 3,
          "non-meta does not pad");
    check(common_spec_usb4_verify_n_tokens(2, 2, true) == 3,
          "n_max=2 pads T=2 to 3");

    check(llama_model_has_meta_device(nullptr) == false,
          "nullptr model is not Meta");
    check(common_spec_target_has_meta(nullptr) == false,
          "nullptr ctx is not Meta");

    const uint64_t fp_t1 = ggml_meta_seq_graph_fp(7287, 2560, 4, 1, 1);
    const uint64_t fp_t8 = ggml_meta_seq_graph_fp(7287, 2560, 4, 8, 1);
    check(fp_t1 != fp_t8, "T=1 and T=8 fingerprints differ");
    check(ggml_meta_seq_sig_reuse(true, 7287, 7287, fp_t1, fp_t1) == true,
          "SEQ plan + same n_nodes + same T reuses graph_sig");
    check(ggml_meta_seq_sig_reuse(true, 7287, 7287, fp_t1, fp_t8) == false,
          "same n_nodes but T=8 vs T=1 forces rehash");
    check(ggml_meta_seq_sig_reuse(true, 7287, 8000, fp_t1, fp_t1) == false,
          "n_nodes change forces rehash");
    check(ggml_meta_seq_sig_reuse(false, 7287, 7287, fp_t1, fp_t1) == false,
          "no SEQ plan forces rehash");
    check(ggml_meta_seq_skip_rpc_sync(true) == true,
          "SEQ success skips dummy RPC sync");
    check(ggml_meta_seq_skip_rpc_sync(false) == false,
          "without SEQ still RPC-syncs");

    if (fails) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return 1;
    }
    fprintf(stderr, "ok: usb4 spec draft gate\n");
    return 0;
}
