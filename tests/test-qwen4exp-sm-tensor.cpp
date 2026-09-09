#include "../src/llama-arch.h"

#include <cstdio>

// USB4STREAM tensor-parallel Flash-Next requires qwen4exp to accept
// LLAMA_SPLIT_MODE_TENSOR / RPC_TENSOR. llama_model_create throws otherwise.
int main() {
    if (!llm_arch_supports_sm_tensor(LLM_ARCH_QWEN4EXP)) {
        fprintf(stderr, "FAIL: LLM_ARCH_QWEN4EXP must support split-mode tensor\n");
        return 1;
    }
    fprintf(stderr, "ok: qwen4exp supports split-mode tensor\n");
    return 0;
}
