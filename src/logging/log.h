#pragma once

// Adapted from llama.cpp common/log.h
// Copyright (c) 2023-2024 The llama.cpp authors
// SPDX-License-Identifier: MIT
//
// Ported to ZInferLM: lives in zinferlm::internal, JSONL support removed,
// no dependency on llama.cpp's common.h / json.h.

#include <ggml.h> // for ggml_log_level

#include <zinferlm/export.h>

#define LOG_CLR_TO_EOL  "\033[K\r"
#define LOG_COL_DEFAULT "\033[0m"
#define LOG_COL_BOLD    "\033[1m"
#define LOG_COL_RED     "\033[31m"
#define LOG_COL_GREEN   "\033[32m"
#define LOG_COL_YELLOW  "\033[33m"
#define LOG_COL_BLUE    "\033[34m"
#define LOG_COL_MAGENTA "\033[35m"
#define LOG_COL_CYAN    "\033[36m"
#define LOG_COL_WHITE   "\033[37m"

#ifndef __GNUC__
#    define LOG_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__) && !defined(__clang__)
#    define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#    define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#define LOG_LEVEL_DEBUG  5
#define LOG_LEVEL_TRACE  4
#define LOG_LEVEL_INFO   3
#define LOG_LEVEL_WARN   2
#define LOG_LEVEL_ERROR  1
#define LOG_LEVEL_OUTPUT 0 // output data from tools

#define LOG_DEFAULT_DEBUG LOG_LEVEL_DEBUG
#define LOG_DEFAULT_INFO  LOG_LEVEL_INFO

namespace zinferlm {
namespace internal {

enum log_colors {
    LOG_COLORS_AUTO     = -1,
    LOG_COLORS_DISABLED = 0,
    LOG_COLORS_ENABLED  = 1,
};

// needed by the LOG_TMPL macro to avoid computing log arguments if the verbosity lower
// set via log_set_verbosity_thold()
ZINFERLM_API int  log_get_verbosity_thold(void);
ZINFERLM_API void log_set_verbosity_thold(int verbosity); // not thread-safe

ZINFERLM_API int log_get_verbosity(enum ggml_log_level level);

// ggml_log_set callback: routes ggml logs into the zinferlm logger
ZINFERLM_API void log_default_callback(enum ggml_log_level level, const char * text, void * user_data);

// the logger uses an internal worker thread to print/write log messages
// when the worker thread is paused, incoming log messages are discarded
struct Log;

ZINFERLM_API struct Log * log_init();

// Singleton, intentionally leaked to avoid Windows teardown hangs.
// Call log_flush() before exit if you want to ensure all logs are flushed.
ZINFERLM_API struct Log * log_main();

ZINFERLM_API void log_pause (struct Log * log); // pause the worker thread, not thread-safe
ZINFERLM_API void log_resume(struct Log * log); // resume the worker thread, not thread-safe
ZINFERLM_API void log_free  (struct Log * log);

LOG_ATTRIBUTE_FORMAT(3, 4)
ZINFERLM_API void log_add(struct Log * log, enum ggml_log_level level, const char * fmt, ...);

// defaults: file = NULL, colors = auto, prefix = false, timestamps = false
//
// regular log output:
//
//   llm_load_tensors: ggml ctx size =    0.27 MiB
//   llm_load_tensors: offloading 32 repeating layers to GPU
//
// with prefix = true, timestamps = true, the log output will look like this:
//
//   0.00.035.060 D ggml_backend_metal_log_allocated_size: allocated buffer, size = 6695.84 MiB
//   0.00.035.064 I llm_load_tensors: ggml ctx size =    0.27 MiB
//
// D - debug   (stderr)
// I - info    (stderr)
// W - warning (stderr)
// E - error   (stderr)
// O - output  (stdout)
//

ZINFERLM_API void log_set_file      (struct Log * log, const char * file); // not thread-safe
ZINFERLM_API void log_set_stdo      (struct Log * log, bool enabled);      // whether to output to stdout/stderr
ZINFERLM_API void log_set_colors    (struct Log * log, log_colors colors); // not thread-safe
ZINFERLM_API bool log_get_colors    (struct Log * log);                    // whether colors are enabled
ZINFERLM_API void log_set_prefix    (struct Log * log, bool prefix);       // whether to output prefix to each log
ZINFERLM_API void log_set_timestamps(struct Log * log, bool timestamps);   // whether to output timestamps in the prefix
ZINFERLM_API void log_flush         (struct Log * log);                    // flush all pending log messages

} // namespace internal
} // namespace zinferlm

// helper macros for logging
// use these to avoid computing log arguments if the verbosity of the log is higher than the threshold
//
// for example:
//
//   LOG_DBG("this is a debug message: %d\n", expensive_function());
//
// this will avoid calling expensive_function() if LOG_LEVEL_DEBUG > log_get_verbosity_thold()
//

#define LOG_TMPL(level, verbosity, ...) \
    do { \
        if ((verbosity) <= zinferlm::internal::log_get_verbosity_thold()) { \
            zinferlm::internal::log_add(zinferlm::internal::log_main(), (level), __VA_ARGS__); \
        } \
    } while (0)

#define LOG(...)             LOG_TMPL(GGML_LOG_LEVEL_NONE, LOG_LEVEL_OUTPUT, __VA_ARGS__)
#define LOGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_NONE, verbosity,        __VA_ARGS__)

#define LOG_DBG(...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_TRC(...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  LOG_LEVEL_TRACE, __VA_ARGS__)
#define LOG_INF(...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, LOG_LEVEL_ERROR, __VA_ARGS__)
#define LOG_CNT(...) LOG_TMPL(GGML_LOG_LEVEL_CONT,  LOG_LEVEL_INFO,  __VA_ARGS__) // same as INFO

#define LOG_DBGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, verbosity, __VA_ARGS__)
#define LOG_TRCV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  verbosity, __VA_ARGS__)
#define LOG_INFV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  verbosity, __VA_ARGS__)
#define LOG_WRNV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  verbosity, __VA_ARGS__)
#define LOG_ERRV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, verbosity, __VA_ARGS__)
#define LOG_CNTV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_CONT,  verbosity, __VA_ARGS__)
