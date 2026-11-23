#include "engine/memplan.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ggml-backend.h>
#include <ggml.h>

#include "inference/inference.h"
#include "kv_cache/cache.h"
#include "logging/log.h"
#include "model_loader/gguf/gguf.h"
#include "models/model_impl.h"
#include "models/qwen/model.h"

namespace zinferlm {
namespace internal {

namespace {

std::string human_bytes(size_t bytes) {
  static const char *units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double value = static_cast<double>(bytes);
  int unit = 0;
  while (value >= 1024.0 && unit < 4) {
    value /= 1024.0;
    ++unit;
  }
  char buffer[64];
  snprintf(buffer, sizeof(buffer), "%.1f %s", value, units[unit]);
  return std::string(buffer);
}

} // namespace

memory_plan_t plan_memory(const char *model_path, const config_t &cfg,
                          bool cpu_only) {
  memory_plan_t plan;

  int fd = open(model_path, O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error("unable to open model file: " +
                             std::string(model_path));
  }
  struct stat model_stats;
  if (fstat(fd, &model_stats) == -1) {
    close(fd);
    throw std::runtime_error("unable to stat model file: " +
                             std::string(model_path));
  }
  plan.file_size = static_cast<size_t>(model_stats.st_size);

  if (!is_gguf_file(&fd)) {
    close(fd);
    throw std::runtime_error("not a GGUF file: " + std::string(model_path));
  }
  ModelLoader *loader = load_gguf_file(&fd, plan.file_size);
  close(fd);
  if (loader == nullptr) {
    throw std::runtime_error("unable to parse GGUF file: " +
                             std::string(model_path));
  }

  // Metadata-only model: tensor structs exist, but no data is allocated or
  // copied. The model owns the loader from here on (and unmaps on destruction).
  QwenModel model(loader, cpu_only);
  model.create_tensor_structs();
  model.mark_tensors_external();

  const zinferlm::model_info_t info = model.info();
  const zinferlm::model_config_t mcfg = model.config();
  const size_t d_head = mcfg.embedding_dim / mcfg.nheads;

  plan.model_name = info.name;
  plan.file_type = info.file_type;
  plan.tensor_count = static_cast<size_t>(loader->get_tensor_count());
  plan.weights = model.size();
  plan.tensor_metadata = model.tensor_metadata_size();

  KVCache cache(model.get_backend(), static_cast<int>(cfg.max_ctx),
                static_cast<int>(mcfg.n_blocks), static_cast<int>(mcfg.nkv),
                static_cast<int>(d_head), GGML_TYPE_F16,
                /*measure_only=*/true);
  plan.kv_cache = cache.size();

  Inference inference(&cache, model, static_cast<int>(cfg.max_batches),
                      static_cast<uint32_t>(cfg.prefill_chunk_size),
                      /*measure_only=*/true);
  plan.compute = inference.size();
  plan.graph_metadata = inference.metadata_size();

  plan.host_scratch = 2 * cfg.max_tokens * cfg.max_ctx + // causal mask (f16)
                      16 * cfg.max_tokens +              // input, positions, write rows
                      4 * cfg.max_ctx +                  // lookup rows
                      4 * cfg.max_batches;               // output rows

  plan.blocks.push_back({"weights", "", "sum(ggml_nbytes)", plan.weights});
  plan.blocks.push_back({"tensor metadata", "",
                         "tensor_count * ggml_tensor_overhead",
                         plan.tensor_metadata});
  plan.blocks.push_back({"kv cache", "",
                         "2 * layers * n_kv * d_head * ctx * 2B * 4",
                         plan.kv_cache});

  std::vector<ggml_backend *> backends = model.backends();
  const std::vector<size_t> &backend_sizes = inference.backend_sizes();
  for (size_t i = 0; i < backends.size() && i < backend_sizes.size(); i++) {
    // Measure-only mode sizes the whole graph against the primary backend's
    // buffer type; secondary backends report zero unless a split allocated.
    if (backend_sizes[i] == 0) {
      continue;
    }
    plan.blocks.push_back({"compute buffers", ggml_backend_name(backends[i]),
                           "activation peak (gallocr)", backend_sizes[i]});
  }

  plan.blocks.push_back({"graph metadata", "", "ggml_used_mem(measure ctx)",
                         plan.graph_metadata});
  plan.blocks.push_back({"host step scratch", "",
                         "mask + row buffers", plan.host_scratch});

  for (const auto &block : plan.blocks) {
    plan.total += block.bytes;
  }

  const uint64_t step_tokens =
      static_cast<uint64_t>(cfg.max_batches) * cfg.prefill_chunk_size;
  if (step_tokens > cfg.max_tokens) {
    plan.warnings.push_back(
        "prefill step can reach " + std::to_string(step_tokens) +
        " tokens (max_batches * prefill_chunk_size) but the reserve graph is "
        "built for max_tokens=" + std::to_string(cfg.max_tokens) +
        "; the compute reservation can be too small");
  }

  const uint64_t full_contexts = 4; // blocks_count() * block_size / max_ctx
  if (cfg.max_batches > full_contexts) {
    plan.warnings.push_back(
        "kv cache pool holds " + std::to_string(full_contexts) +
        " full contexts but max_batches=" + std::to_string(cfg.max_batches) +
        "; long concurrent sequences will exhaust kv blocks");
  }

  if (plan.host_scratch > (size_t)1024 * 1024 * 1024) {
    plan.warnings.push_back(
        "host step scratch is " + human_bytes(plan.host_scratch) +
        " (causal mask is n_tokens * total_filled); reduce max_tokens or "
        "prefill_chunk_size");
  }

  LOG_DBG("memplan: %s file=%s weights=%zu kv=%zu compute=%zu total=%zu\n",
          plan.model_name.c_str(), human_bytes(plan.file_size).c_str(),
          plan.weights, plan.kv_cache, plan.compute, plan.total);

  return plan;
}

} // namespace internal
} // namespace zinferlm
