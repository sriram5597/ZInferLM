#pragma once

// Shared logging flags for ZInferLM tools.
//
// Supported flags (anywhere in argv):
//   -v, --verbosity N        log verbosity threshold (0..5, default 3 = info)
//   --log-file PATH          also write all logs (incl. debug) to PATH
//   --log-colors MODE        auto | on | off

#include <cstdlib>
#include <string>
#include <vector>

#include "logging/log.h"

namespace zinferlm_tools {

// Applies logging flags found in argv and returns the remaining positional
// arguments (argv[0] excluded).
inline std::vector<std::string> parse_log_flags(int argc, char *argv[]) {
  std::vector<std::string> rest;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-v" || arg == "--verbosity") && i + 1 < argc) {
      zinferlm::internal::log_set_verbosity_thold(std::atoi(argv[++i]));
    } else if (arg == "--log-file" && i + 1 < argc) {
      zinferlm::internal::log_set_file(zinferlm::internal::log_main(),
                                       argv[++i]);
    } else if (arg == "--log-colors" && i + 1 < argc) {
      std::string mode = argv[++i];
      auto colors = mode == "on"    ? zinferlm::internal::LOG_COLORS_ENABLED
                    : mode == "off" ? zinferlm::internal::LOG_COLORS_DISABLED
                                    : zinferlm::internal::LOG_COLORS_AUTO;
      zinferlm::internal::log_set_colors(zinferlm::internal::log_main(),
                                         colors);
    } else {
      rest.push_back(arg);
    }
  }
  return rest;
}

inline const char *log_flags_usage() {
  return "[-v N] [--log-file PATH] [--log-colors auto|on|off]";
}

} // namespace zinferlm_tools
