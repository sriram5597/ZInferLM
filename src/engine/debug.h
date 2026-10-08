#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <ggml.h>

struct DebugCallbackData {
    bool debug_mode;
    std::vector<uint8_t> buffer;
};

float debug_get_float_value(const uint8_t *data, ggml_type type,
                            const size_t *nb, size_t i0, size_t i1,
                            size_t i2, size_t i3);

void debug_print_tensor_values(const uint8_t *data, ggml_type type,
                               const int64_t *ne, const size_t *nb, int n);

bool debug_eval_callback(ggml_tensor *tensor, bool ask, void *user_data);
