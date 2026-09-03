#include <iostream>
#include <functional>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <unistd.h>
#include <vector>
#include <map>
#include <cstring>
#include <unordered_map>
#include <format>
#include <sys/mman.h>
#include <memory>

#include <zinferlm/models.h>
#include "metadata.h"
#include "tensors.h"
#include "gguf.h"

bool is_gguf_file(int *fd)
{
  char magic[4];
  read(*fd, magic, sizeof(decltype(gguf_header_t::magic)));
  lseek(*fd, 0, SEEK_SET);
  return std::memcmp(magic, "GGUF", 4) == 0;
}

std::unordered_map<std::string, std::unique_ptr<Metadata>> gguf_metadata(const char *&ptr, int kv_count)
{
  std::unordered_map<std::string, std::unique_ptr<Metadata>> metadata;
  for (int i = 0; i < kv_count; i++)
  {
    Metadata met = Metadata::from_ptr(ptr);
    ptr += met.size();
    metadata.insert({met.key_string(), std::make_unique<Metadata>(std::move(met))});
  }
  return metadata;
}

std::unordered_map<std::string, std::unique_ptr<TensorInfo>> gguf_tensors_info(const char *&ptr, uint64_t tensors_count)
{
  std::unordered_map<std::string, std::unique_ptr<TensorInfo>> tensors_map;
  for (int i = 0; i < tensors_count; i++)
  {
    TensorInfo t = TensorInfo::from_ptr(ptr);
    ptr += t.size();
    tensors_map.insert({t.name->to_string(), std::make_unique<TensorInfo>(std::move(t))});
  }
  return tensors_map;
}

std::unique_ptr<GGUFLoader> load_gguf_file(int *fd, size_t file_size)
{
  void *raw_map = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (raw_map == MAP_FAILED)
  {
    std::cerr << "Error loading gguf file" << std::endl;
    return nullptr;
  }
  const char *base_ptr = static_cast<const char *>(raw_map);

  const gguf_header_t *header = reinterpret_cast<const gguf_header_t *>(base_ptr);

  GGUFLoader gguf;
  gguf.header = header;

  const char *cur = base_ptr + sizeof(gguf_header_t);
  gguf.metadata = gguf_metadata(cur, header->metadata_kv_count);
  gguf.tensors = gguf_tensors_info(cur, header->tensor_count);
  gguf.set_tensor_base_ptr(cur);

  auto unmap_deleter = [file_size](const char *ptr)
  {
    munmap(const_cast<char *>(ptr), file_size);
  };
  gguf.set_mapped_memory(std::unique_ptr<const char, std::function<void(const char *)>>(
      base_ptr, std::function<void(const char *)>(unmap_deleter)));

  return std::make_unique<GGUFLoader>(std::move(gguf));
}

zinferlm::model_info_t GGUFLoader::info() const
{
  zinferlm::model_info_t model_info;

  model_info.architecture = get_metadata("general.architecture")->val_string();
  model_info.version = get_metadata("general.version")->val_string();
  model_info.name = get_metadata("general.name")->val_string();
  uint32_t file_type = get_metadata("general.file_type")->value<uint32_t>();
  model_info.file_type = gguf_file_type_name(static_cast<gguf_file_type>(file_type));
  return model_info;
}

zinferlm::model_config_t GGUFLoader::model_config() const {
  zinferlm::model_config_t config;
  config.nheads = get_metadata("qwen2.attention.head_count")->value<uint32_t>();
  config.nkv = get_metadata("qwen2.attention.head_count_kv")->value<uint32_t>();
  config.rope_freq_base = get_metadata("qwen2.rope.freq_base")->value<float>();
  config.embedding_dim = get_metadata("qwen2.embedding_length")->value<uint32_t>();
  config.n_blocks = get_metadata("qwen2.block_count")->value<uint32_t>();
  return config;
}

uint64_t GGUFLoader::align_address(uint64_t ptr) const
{
  uint32_t alignment = get_metadata("general.alignment")->value<uint8_t>();
  if (!alignment)
  {
    alignment = 32;
  }
  return ptr + (alignment - ptr % alignment) % alignment;
}

void GGUFLoader::set_tensor_base_ptr(const char *ptr)
{
  uint64_t address = reinterpret_cast<uint64_t>(ptr);
  tensor_base_ptr_ = reinterpret_cast<void *>(align_address(address));
}

Metadata *GGUFLoader::get_metadata(std::string key) const
{
  const auto &it = metadata.find(key);
  if (it != metadata.end())
  {
    std::cout << "Metadata: " << it->second->key_string() << " " << it->second->dtype() << std::endl;
    return it->second.get();
  }
  std::cout << "Metadata not found: " << key << std::endl;
  return const_cast<Metadata *>(&default_metadata_);
}

zinferlm::tokenizer_info_t GGUFLoader::tokenizer_info() const
{
  zinferlm::tokenizer_info_t t_info;

  t_info.model = get_metadata("tokenizer.ggml.model")->val_string();
  t_info.pre = get_metadata("tokenizer.ggml.pre")->val_string();
  t_info.tokens = get_metadata("tokenizer.ggml.tokens")->val_array<std::string>();
  t_info.merges = get_metadata("tokenizer.ggml.merges")->val_array<std::string>();
  t_info.token_type = get_metadata("tokenizer.ggml.token_type")->val_array<int32_t>();
  t_info.bos_token_id = get_metadata("tokenizer.ggml.bos_token_id")->value<uint32_t>();
  t_info.eos_token_id = get_metadata("tokenizer.ggml.eos_token_id")->value<uint32_t>();
  t_info.padding_token_id = get_metadata("tokenizer.ggml.padding_token_id")->value<uint32_t>();
  t_info.add_bos_token = get_metadata("tokenizer.ggml.add_bos_token")->value<bool>();

  return t_info;
}

std::vector<zinferlm::tensor_info_t> GGUFLoader::tensor_info() const
{
  std::vector<zinferlm::tensor_info_t> tinfo_list;
  for (auto &&ele : tensors)
  {
    auto&& t = ele.second;
    tinfo_list.push_back(zinferlm::tensor_info_t{
        .name = t->name->to_string(),
        .type_name = ggml_type_name(*t->dtype),
        .type_id = static_cast<uint32_t>(*t->dtype),
        .n_dim = *t->n_dimensions,
        .dimensions = t->dimensions});
  }
  return tinfo_list;
}

zinferlm::tensor_info_t GGUFLoader::tensor_info(std::string name) const {
  const auto &it = tensors.find(name);
  if (it != tensors.end())
  {
    TensorInfo* t = it->second.get();
    return zinferlm::tensor_info_t{
        .name = t->name->to_string(),
        .type_name = ggml_type_name(*t->dtype),
        .type_id = static_cast<uint32_t>(*t->dtype),
        .n_dim = *t->n_dimensions,
        .dimensions = t->dimensions};
  }
  return zinferlm::tensor_info_t{};
}

uint64_t GGUFLoader::get_tensor_count() const
{
  return header->tensor_count;
}

void *GGUFLoader::get_tensor_ptr(uint64_t offset) const
{
  return tensor_base_ptr_ + offset;
}
