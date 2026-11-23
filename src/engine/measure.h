#pragma once

#include <cstdint>

#include <ggml.h>

namespace zinferlm {
namespace internal {

// Sentinel address used to mark tensors whose storage lives outside the
// compute buffers (weights, KV pool). ggml's allocator treats a tensor with
// non-null `data` as externally allocated and skips it when sizing compute
// buffers, which is exactly the state of a real run.
//
// Measurement only: these tensors are never read or written.
inline constexpr uintptr_t kExternalTensorMarker = 0x10;

inline void mark_external_tensor(ggml_tensor *tensor) {
  if (tensor != nullptr) {
    tensor->data = reinterpret_cast<void *>(kExternalTensorMarker);
  }
}

} // namespace internal
} // namespace zinferlm
