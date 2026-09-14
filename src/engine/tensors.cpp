#include <cstdint>
#include <iostream>
#include <string>
#include <ggml.h>
#include <vector>

#include "tensors.h"

ggml_tensor* create_tensor(ggml_context* ctx, std::string name, ggml_type type, void* data, std::vector<uint64_t> dims) {
  ggml_tensor* t = ggml_new_tensor(ctx, type, dims.size(), reinterpret_cast<int64_t *>(dims.data()));
  t->data = data;
  ggml_set_name(t, name.c_str());

  std::cout << "[Weight] " << name << " | type: " << ggml_type_name(type)
            << " | dims: [";
  for (size_t i = 0; i < dims.size(); i++) {
    std::cout << dims[i];
    if (i < dims.size() - 1) std::cout << ", ";
  }
  std::cout << "] | data_ptr: " << data;

  if (type == GGML_TYPE_F32 && data != nullptr) {
    float* fdata = reinterpret_cast<float*>(data);
    std::cout << " | first 5 values: [";
    for (int i = 0; i < 5; i++) {
      std::cout << fdata[i];
      if (i < 4) std::cout << ", ";
    }
    std::cout << "]";
  }
  std::cout << std::endl;

  return t;
}
