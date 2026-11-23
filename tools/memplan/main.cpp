#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <zinferlm/config.hpp>

#include "../log_flags.h"
#include "engine/memplan.h"
#include "logging/log.h"

namespace {

constexpr const char *kUsage =
    "Usage: %s <model.gguf> [--ctx N] [--prefill N] [--batches N]\n"
    "                       [--max-tokens N] [--cpu] [--json] [-v N]\n"
    "                       [--log-file PATH] [--log-colors auto|on|off]\n"
    "\n"
    "Prints the memory a run would need: weights, tensor metadata, KV cache,\n"
    "compute buffers and host step scratch. Nothing is allocated: the model is\n"
    "parsed from metadata and compute sizes come from ggml's no-alloc planner.\n";

std::string human_bytes(size_t bytes) {
  static const char *units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double value = static_cast<double>(bytes);
  int unit = 0;
  while (value >= 1024.0 && unit < 4) {
    value /= 1024.0;
    ++unit;
  }
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.1f %s", value, units[unit]);
  return std::string(buffer);
}

bool parse_size(const char *value, size_t &out) {
  if (value == nullptr || *value == '\0') {
    return false;
  }
  errno = 0;
  char *end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0') {
    return false;
  }
  out = static_cast<size_t>(parsed);
  return true;
}

std::string json_escape(const std::string &text) {
  std::string out;
  out.reserve(text.size() + 2);
  for (char ch : text) {
    switch (ch) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    default:
      out += ch;
      break;
    }
  }
  return out;
}

void print_table(const zinferlm::internal::memory_plan_t &plan,
                 const zinferlm::config_t &cfg) {
  std::printf("model        %s\n", plan.model_name.c_str());
  std::printf("file         %s (%s, %zu tensors)\n", human_bytes(plan.file_size).c_str(),
              plan.file_type.c_str(), plan.tensor_count);
  std::printf("config        max_ctx=%zu max_tokens=%zu prefill=%zu batches=%zu\n",
              cfg.max_ctx, cfg.max_tokens, cfg.prefill_chunk_size,
              cfg.max_batches);
  std::printf("\n");

  std::printf("%-22s %-18s %14s %12s %7s\n", "component", "backend", "bytes",
              "size", "%");
  std::printf("%-22s %-18s %14s %12s %7s\n", "----------------------",
              "------------------", "--------------", "------------",
              "-------");
  for (const auto &block : plan.blocks) {
    const double pct = plan.total == 0 ? 0.0
                                       : 100.0 * (double)block.bytes /
                                             (double)plan.total;
    std::printf("%-22s %-18s %14zu %12s %6.1f%%\n", block.name.c_str(),
                block.backend.empty() ? "-" : block.backend.c_str(),
                block.bytes, human_bytes(block.bytes).c_str(), pct);
  }
  std::printf("%-22s %-18s %14s %12s %7s\n", "----------------------",
              "------------------", "--------------", "------------",
              "-------");
  std::printf("%-22s %-18s %14zu %12s %6.1f%%\n", "total", "-", plan.total,
              human_bytes(plan.total).c_str(), 100.0);

  if (!plan.warnings.empty()) {
    std::printf("\n");
    for (const auto &warning : plan.warnings) {
      std::printf("warning: %s\n", warning.c_str());
    }
  }
}

void print_json(const zinferlm::internal::memory_plan_t &plan,
                const zinferlm::config_t &cfg) {
  std::printf("{\n");
  std::printf("  \"model\": \"%s\",\n", json_escape(plan.model_name).c_str());
  std::printf("  \"file_type\": \"%s\",\n", json_escape(plan.file_type).c_str());
  std::printf("  \"file_size\": %zu,\n", plan.file_size);
  std::printf("  \"tensor_count\": %zu,\n", plan.tensor_count);
  std::printf("  \"config\": {\"max_ctx\": %zu, \"max_tokens\": %zu, "
              "\"prefill_chunk_size\": %zu, \"max_batches\": %zu},\n",
              cfg.max_ctx, cfg.max_tokens, cfg.prefill_chunk_size,
              cfg.max_batches);
  std::printf("  \"blocks\": [\n");
  for (size_t i = 0; i < plan.blocks.size(); i++) {
    const auto &block = plan.blocks[i];
    std::printf("    {\"name\": \"%s\", \"backend\": \"%s\", "
                "\"formula\": \"%s\", \"bytes\": %zu}%s\n",
                json_escape(block.name).c_str(),
                json_escape(block.backend).c_str(),
                json_escape(block.formula).c_str(), block.bytes,
                i + 1 < plan.blocks.size() ? "," : "");
  }
  std::printf("  ],\n");
  std::printf("  \"summary\": {\"weights\": %zu, \"tensor_metadata\": %zu, "
              "\"kv_cache\": %zu, \"compute\": %zu, \"graph_metadata\": %zu, "
              "\"host_scratch\": %zu, \"total\": %zu},\n",
              plan.weights, plan.tensor_metadata, plan.kv_cache, plan.compute,
              plan.graph_metadata, plan.host_scratch, plan.total);
  std::printf("  \"warnings\": [");
  for (size_t i = 0; i < plan.warnings.size(); i++) {
    std::printf("%s\"%s\"", i == 0 ? "" : ", ",
                json_escape(plan.warnings[i]).c_str());
  }
  std::printf("]\n}\n");
}

} // namespace

int main(int argc, char *argv[]) {
  std::vector<std::string> args = zinferlm_tools::parse_log_flags(argc, argv);

  std::string model_path;
  zinferlm::config_t cfg;
  bool cpu_only = false;
  bool json = false;

  for (size_t i = 0; i < args.size(); i++) {
    const std::string &arg = args[i];
    auto take_value = [&args, &i](const char *flag) -> const char * {
      if (i + 1 >= args.size()) {
        std::fprintf(stderr, "missing value for %s\n", flag);
        std::exit(2);
      }
      return args[++i].c_str();
    };

    if (arg == "-h" || arg == "--help") {
      std::printf(kUsage, argv[0]);
      return 0;
    } else if (arg == "--ctx") {
      if (!parse_size(take_value("--ctx"), cfg.max_ctx)) {
        std::fprintf(stderr, "invalid value for --ctx\n");
        return 2;
      }
    } else if (arg == "--prefill") {
      if (!parse_size(take_value("--prefill"), cfg.prefill_chunk_size)) {
        std::fprintf(stderr, "invalid value for --prefill\n");
        return 2;
      }
    } else if (arg == "--batches") {
      if (!parse_size(take_value("--batches"), cfg.max_batches)) {
        std::fprintf(stderr, "invalid value for --batches\n");
        return 2;
      }
    } else if (arg == "--max-tokens") {
      if (!parse_size(take_value("--max-tokens"), cfg.max_tokens)) {
        std::fprintf(stderr, "invalid value for --max-tokens\n");
        return 2;
      }
    } else if (arg == "--cpu") {
      cpu_only = true;
    } else if (arg == "--json") {
      json = true;
    } else if (!arg.empty() && arg[0] == '-') {
      std::fprintf(stderr, "unknown flag: %s\n", arg.c_str());
      std::fprintf(stderr, kUsage, argv[0]);
      return 2;
    } else if (model_path.empty()) {
      model_path = arg;
    } else {
      std::fprintf(stderr, "unexpected argument: %s\n", arg.c_str());
      return 2;
    }
  }

  if (model_path.empty()) {
    std::fprintf(stderr, kUsage, argv[0]);
    return 1;
  }

  if (cfg.prefill_chunk_size == 0 || cfg.max_batches == 0 ||
      cfg.max_ctx == 0 || cfg.max_tokens == 0) {
    std::fprintf(stderr, "config values must be greater than zero\n");
    return 2;
  }

  zinferlm::ConfigManager::init(cfg);

  try {
    zinferlm::internal::memory_plan_t plan =
        zinferlm::internal::plan_memory(model_path.c_str(), cfg, cpu_only);
    if (json) {
      print_json(plan, cfg);
    } else {
      print_table(plan, cfg);
    }
  } catch (const std::exception &e) {
    LOG_ERR("memplan failed: %s\n", e.what());
    return 1;
  }

  return 0;
}
