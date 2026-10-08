#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <ggml.h>
#include <ggml-backend.h>

#include "debug.h"

#define INDENT "    "

float debug_get_float_value(const uint8_t *data, ggml_type type,
                             const size_t *nb, size_t i0, size_t i1,
                             size_t i2, size_t i3) {
  size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
  float v;
  if (type == GGML_TYPE_F16) {
    v = ggml_fp16_to_fp32(*(const ggml_fp16_t *)&data[i]);
  } else if (type == GGML_TYPE_F32) {
    v = *(const float *)&data[i];
  } else if (type == GGML_TYPE_I64) {
    v = (float)*(const int64_t *)&data[i];
  } else if (type == GGML_TYPE_I32) {
    v = (float)*(const int32_t *)&data[i];
  } else if (type == GGML_TYPE_I16) {
    v = (float)*(const int16_t *)&data[i];
  } else if (type == GGML_TYPE_I8) {
    v = (float)*(const int8_t *)&data[i];
  } else if (type == GGML_TYPE_BF16) {
    v = ggml_bf16_to_fp32(*(const ggml_bf16_t *)&data[i]);
  } else {
    GGML_ABORT("fatal error");
  }
  return v;
}

void debug_print_tensor_values(const uint8_t *data, ggml_type type,
                                const int64_t *ne, const size_t *nb, int n) {
  GGML_ASSERT(n > 0);
  float sum = 0;
  for (int64_t i3 = 0; i3 < ne[3]; i3++) {
    for (int64_t i2 = 0; i2 < ne[2]; i2++) {
      for (int64_t i1 = 0; i1 < ne[1]; i1++) {
        for (int64_t i0 = 0; i0 < ne[0]; i0++) {
          const float v =
              debug_get_float_value(data, type, nb, i0, i1, i2, i3);
          sum += v;
        }
      }
    }
  }
  for (int64_t i3 = 0; i3 < ne[3]; i3++) {
    std::printf(INDENT "[\n");
    for (int64_t i2 = 0; i2 < ne[2]; i2++) {
      if (i2 == n && ne[2] > 2 * n) {
        std::printf(INDENT INDENT "..., \n");
        i2 = ne[2] - n;
      }
      std::printf(INDENT INDENT "[\n");
      for (int64_t i1 = 0; i1 < ne[1]; i1++) {
        if (i1 == n && ne[1] > 2 * n) {
          std::printf(INDENT INDENT INDENT "..., \n");
          i1 = ne[1] - n;
        }
        std::printf(INDENT INDENT INDENT "[");
        for (int64_t i0 = 0; i0 < ne[0]; i0++) {
          if (i0 == n && ne[0] > 2 * n) {
            std::printf("   ..., ");
            i0 = ne[0] - n;
          }
          const float v =
              debug_get_float_value(data, type, nb, i0, i1, i2, i3);
          std::printf("%12.4f", v);
          if (i0 < ne[0] - 1) {
            std::printf(", ");
          }
        }
        std::printf("  ],\n");
      }
      std::printf(INDENT INDENT "],\n");
    }
    std::printf(INDENT "]\n");
    std::printf(INDENT "sum = %f\n", sum);
  }
}

bool debug_eval_callback(ggml_tensor *tensor, bool ask, void *user_data) {
  auto *cb_data = static_cast<DebugCallbackData *>(user_data);
  bool is_layer_output = true;
  const char *name = ggml_get_name(tensor);

  if (ask) {
    return cb_data->debug_mode && (is_layer_output);
  }

  if (!cb_data->debug_mode || (!is_layer_output)) {
    return true;
  }

  if (ggml_is_quantized(tensor->type)) {
    return true;
  }

  const char *op = ggml_op_desc(tensor);

  std::cout << "[DEBUG] " << name << " = " << op
            << " | type: " << ggml_type_name(tensor->type) << " | shape: ["
            << tensor->ne[0];
  for (int i = 1; i < GGML_MAX_DIMS; i++) {
    if (tensor->ne[i] > 1) {
      std::cout << ", " << tensor->ne[i];
    }
  }
  std::cout << "]" << std::endl;

  size_t nbytes = ggml_nbytes(tensor);
  cb_data->buffer.resize(nbytes);

  bool is_host = ggml_backend_buffer_is_host(tensor->buffer);
  uint8_t *data_ptr;

  if (is_host) {
    data_ptr = static_cast<uint8_t *>(tensor->data);
  } else {
    ggml_backend_tensor_get(tensor, cb_data->buffer.data(), 0, nbytes);
    data_ptr = cb_data->buffer.data();
  }

  std::cout << "values:\n";
  debug_print_tensor_values(data_ptr, tensor->type, tensor->ne, tensor->nb, 3);
  std::cout << std::endl;

  return true;
}
