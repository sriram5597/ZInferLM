#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <zinferlm/models.h>

namespace zinferlm {
namespace internal {

// Internal loader interface. Implementations (e.g. GGUFLoader) read model
// metadata/tensors from disk. Not part of the public API.
class ModelLoader {
public:
  virtual ~ModelLoader() = default;

  virtual zinferlm::model_info_t info() const = 0;
  virtual zinferlm::tokenizer_info_t tokenizer_info() const = 0;
  virtual std::vector<zinferlm::tensor_info_t> tensor_info() const = 0;
  virtual void *get_tensor_ptr(uint64_t offset) const = 0;
  virtual uint64_t get_tensor_count() const = 0;
  virtual zinferlm::tensor_info_t tensor_info(std::string id) const = 0;
  virtual zinferlm::model_config_t model_config() const = 0;
};

} // namespace internal
} // namespace zinferlm
