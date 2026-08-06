#include <iostream>
#include <functional>
#include <cstddef>
#include <cstdint>
#include <unistd.h>
#include <vector>
#include <map>
#include <cstring>
#include <map>
#include <format>
#include <sys/mman.h>
#include <memory>

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

std::map<std::string, std::unique_ptr<Metadata>> gguf_metadata(const char *&ptr, int kv_count)
{
  std::map<std::string, std::unique_ptr<Metadata>> metadata;
  for (int i = 0; i < kv_count; i++)
  {
    Metadata met = Metadata::from_ptr(ptr);
    ptr += met.size();
    metadata.insert({met.key_string(), std::make_unique<Metadata>(std::move(met))});
  }
  return metadata;
}

std::vector<std::unique_ptr<TensorInfo>> gguf_tensors_info(const char *ptr, uint64_t tensors_count)
{
  std::vector<std::unique_ptr<TensorInfo>> tensors;
  for (int i = 0; i < tensors_count; i++)
  {
    TensorInfo t = TensorInfo::from_ptr(ptr);
    ptr += t.size();
    tensors.push_back(std::make_unique<TensorInfo>(std::move(t)));
  }
  return tensors;
}

void GGUFFile::print()
{
  // Header
  char magic[5] = {};
  std::cout << "\n=== GGUF File Header ===\n";
  std::cout << std::format("  Magic:          {}\n", header->magic);
  std::cout << std::format("  Version:        {}\n", header->version);
  std::cout << std::format("  Tensor Count:   {}\n", header->tensor_count);
  std::cout << std::format("  Metadata Count: {}\n", header->metadata_kv_count);

  // Compute dynamic column widths
  std::size_t max_key = 3, max_type = 4;
  for (const auto &[key, val] : metadata)
  {
    max_key = std::max(max_key, val->key_string().size());
    max_type = std::max(max_type, metadata_dtype_name(val->dtype()).size());
  }

  // Metadata table
  std::cout << "\n=== Metadata (Key/Value) ===\n";
  std::cout << std::format("  {:<{}}  {:<{}}  Value\n",
                           "Key", max_key, "Type", max_type);
  std::cout << "  " << std::string(max_key, '-') << "  "
            << std::string(max_type, '-') << "  --------\n";
  for (const auto &[k, m] : metadata)
  {
    std::cout << std::format("  {:<{}}  {:<{}}  {}\n",
                             m->key_string(), max_key,
                             metadata_dtype_name(m->dtype()), max_type,
                             m->value_string());
  }

  // Tensor info table
  std::size_t max_tname = 4, max_ttype = 4;
  for (const auto &t : tensors)
  {
    max_tname = std::max(max_tname, t->name_string().size());
    max_ttype = std::max(max_ttype, ggml_type_name(t->type()).size());
  }

  std::cout << "\n=== Tensor Info ===\n";
  std::cout << std::format("  {:<{}}  {:<{}}  NDims  Dims                 Offset\n",
                           "Name", max_tname, "Type", max_ttype);
  std::cout << "  " << std::string(max_tname, '-') << "  "
            << std::string(max_ttype, '-') << "  -----  ------------------  --------\n";
  for (const auto &t : tensors)
  {
    std::cout << std::format("  {:<{}}  {:<{}}  {:<5}  {:<18}  {}\n",
                             t->name_string(), max_tname,
                             ggml_type_name(t->type()), max_ttype,
                             t->n_dim_sting(),
                             t->dims_string(),
                             t->offset_string());
  }
}

std::unique_ptr<GGUFFile> load_gguf_file(int *fd, size_t file_size)
{
  void *raw_map = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (raw_map == MAP_FAILED)
  {
    std::cerr << "Error loading gguf file" << std::endl;
    return nullptr;
  }
  const char *base_ptr = static_cast<const char *>(raw_map);

  const gguf_header_t *header = reinterpret_cast<const gguf_header_t *>(base_ptr);

  GGUFFile gguf;
  gguf.header = header;

  const char *cur = base_ptr + sizeof(gguf_header_t);
  gguf.metadata = gguf_metadata(cur, header->metadata_kv_count);
  gguf.tensors = gguf_tensors_info(cur, header->tensor_count);

  auto unmap_deleter = [file_size](const char *ptr)
  {
    munmap(const_cast<char *>(ptr), file_size);
  };
  gguf.set_mapped_memory(std::unique_ptr<const char, std::function<void(const char *)>>(
      base_ptr, std::function<void(const char *)>(unmap_deleter)));

  return std::make_unique<GGUFFile>(std::move(gguf));
}
