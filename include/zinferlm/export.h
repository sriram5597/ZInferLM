#pragma once

// Visibility control for the zinferlm shared library.
//
// The library is built with -fvisibility=hidden, so only symbols annotated with
// ZINFERLM_API (or explicitly marked default) are exported from the .so.

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(ZINFERLM_STATIC)
    #define ZINFERLM_API
  #elif defined(zinferlm_EXPORTS)
    #define ZINFERLM_API __declspec(dllexport)
  #else
    #define ZINFERLM_API __declspec(dllimport)
  #endif
#else
  #define ZINFERLM_API __attribute__((visibility("default")))
#endif
