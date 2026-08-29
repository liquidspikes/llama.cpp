#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <cinttypes>

namespace fs = std::filesystem;

static const size_t HASH_THRESHOLD = 0;

static uint64_t fnv_hash(const uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> <output_cache_dir>\n";
        return 1;
    }

    const char * model_path = argv[1];
    const char * cache_dir = argv[2];

    fs::create_directories(cache_dir);

    struct ggml_context * ctx_ggml = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &ctx_ggml,
    };

    struct gguf_context * ctx_gguf = gguf_init_from_file(model_path, params);
    if (!ctx_gguf) {
        std::cerr << "Failed to load GGUF from " << model_path << "\n";
        return 1;
    }

    int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    std::cout << "Loaded " << n_tensors << " tensors from " << model_path << "\n";

    size_t cached_count = 0;
    size_t total_bytes = 0;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * tensor = ggml_get_tensor(ctx_ggml, name);
        if (!tensor || !tensor->data) {
            continue;
        }

        size_t size = ggml_nbytes(tensor);
        if (size > HASH_THRESHOLD) {
            uint64_t hash = fnv_hash((const uint8_t *)tensor->data, size);
            char hash_str[17];
            snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
            fs::path cache_file = fs::path(cache_dir) / hash_str;

            if (!fs::exists(cache_file)) {
                std::ofstream ofs(cache_file, std::ios::binary);
                ofs.write((const char *)tensor->data, size);
                ofs.close();
            }

            cached_count++;
            total_bytes += size;
        }
    }

    std::cout << "Cached " << cached_count << " total tensors (" << total_bytes / (1024 * 1024)
              << " MB) to " << cache_dir << "\n";

    gguf_free(ctx_gguf);
    if (ctx_ggml) {
        ggml_free(ctx_ggml);
    }
    return 0;
}
