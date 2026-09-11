#include "speculative.h"
#include "../src/llama-ext.h"

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

    if (fails) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return 1;
    }
    fprintf(stderr, "ok: usb4 spec draft gate\n");
    return 0;
}
