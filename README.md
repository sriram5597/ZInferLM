# ZInferLM

**A zero-copy LLM inference runtime in C++20 — GGUF-native model loading, quantized weight awareness, and a custom BPE tokenizer with PCRE2 JIT-compiled regex pretokenization.**

ZInferLM is a foundational inference engine for large language models. It loads GGUF-format models directly over memory-mapped files (zero-copy, no heap deserialization), parses their full metadata and tensor layout, and tokenizes text using a GPT-2–compatible BPE tokenizer with configurable regex pretokenization patterns. The library is designed as the critical bottom layer of a high-performance LLM serving stack — model ingestion and text preprocessing — with the forward-pass engine as the natural next stage.

---

## Architecture

```
┌──────────────────────────────────────────┐
│  CLI  (inspect / tokenize)                │
└────────────┬─────────────────────────────┘
             │
┌────────────▼─────────────────────────────┐
│  Public API  (Model, Tokenizer)           │
│  include/zinferlm/                         │
└────┬──────────────────────┬──────────────┘
     │                      │
┌────▼──────────┐  ┌────────▼─────────────┐
│  Model Loader  │  │  Tokenizer            │
│  src/model_    │  │  src/tokenizer/       │
│  loader/       │  │                       │
│                │  │  • PCRE2 JIT regex    │
│  • GGUF v3     │  │  • Byte→Unicode map   │
│    parser      │  │  • BPE merge ranking  │
│  • mmap file   │  │  • Token→ID lookup    │
│    access      │  │                       │
│  • Metadata KV │  │                       │
│  • Tensor index│  │                       │
└───────────────┘  └───────────────────────┘
```

---

## Features

### Model Loading
- **Zero-copy file access** via `mmap()` — model weights and metadata are referenced directly from mapped memory; no deserialization into heap buffers.
- **Full GGUF v3 spec support** — parses magic, version, metadata key-value pairs, and tensor index entries using pointer-arithmetic over the mapped region.
- **Heterogeneous metadata model** — C++20 `std::variant`–backed polymorphic value types (`uint8..uint64`, `int8..int64`, `float32/64`, `bool`, `string`, `array`) with exact byte accounting for sequential streaming parse.
- **Complete tensor type enumeration** — all 40 GGML quantized and unquantized types catalogued, from legacy `Q4_0`/`Q4_1` through `K`-quants, `IQ` importance quants, `TQ` ternary quants, `BF16`, and `MXFP4`.

### Tokenizer
- **GPT-2 BPE algorithm** — faithful implementation of the GPT-2 byte-pair encoding pipeline: regex pretokenization, byte-to-Unicode character remapping (identical to `bytes_to_unicode()`), iterative merge with rank-based greedy pair selection, and vocabulary ID mapping.
- **PCRE2 with JIT compilation** — pretokenization regex patterns are compiled with PCRE2's JIT engine for native-speed matching, avoiding the overhead of interpreted regex at tokenization time.
- **Configurable pretokenizer** — architecture-specific regex patterns (Qwen2, Llama) are pluggable via the `PRE_TOKENIZER_TYPE` enum and associated regex strings.

### Foundation for Inference
- **Quantization-aware** — the tensor index captures the GGML type for every tensor in the file, enabling dispatch to quantized dequantization kernels at inference time.
- **Static library target** — `libzinferlm.a` links directly into an inference engine, with no runtime dependency beyond PCRE2.
- **Singleton model access** — `Model::instance()` ensures a single loaded model at a time, with a factory-style `Model::load()` that auto-detects file format.

---

## Supported Quantization Formats

| Category | Types |
|---|---|
| **Unquantized** | F32, F16, BF16, F64, I8, I16, I32, I64 |
| **Legacy Q** | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 |
| **K-quants** | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K |
| **Importance quants** | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS |
| **Ternary quants** | TQ1_0, TQ2_0 |
| **Microscaling** | MXFP4 |

---

## Quick Start

### Prerequisites
- C++20 compiler (GCC 12+ or Clang 16+)
- CMake ≥ 3.20
- PCRE2 development library (`libpcre2-dev` on Debian/Ubuntu, `pcre2` on macOS)
- Ninja (optional but recommended)

### Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Artifacts:
- `build/lib/libzinferlm.a` — static library
- `build/bin/zinferlm-cli` — CLI tool

### Usage

```bash
# Inspect a GGUF model's metadata
./build/bin/zinferlm-cli inspect models/qwen2.5-1.5b-instruct-q5_k_m.gguf

# View tokenizer configuration
./build/bin/zinferlm-cli tokenize models/qwen2.5-1.5b-instruct-q5_k_m.gguf --info

# Tokenize text from stdin
echo "The capital of France is" | ./build/bin/zinferlm-cli tokenize models/qwen2.5-1.5b-instruct-q5_k_m.gguf --stdin
```

### Linking

```cmake
target_link_libraries(your_inference_engine PRIVATE zinferlm pcre2-8)
```

---

## Technical Details

### GGUF Parsing via In-Place Pointer Arithmetic

The GGUF file is opened via `open()` and mapped with `mmap()`. Parsing operates directly on the mapped region by advancing typed `const char*` pointers — no intermediate deserialization structures, no copying of tensor data. String metadata values are represented as `std::string_view` pointing into the mmap'd region:

```
mmap'd region:  [magic][version][n_tensors][n_kv][kv...][ti...][tensor data...]
                                    ↑
                              pointer advances sequentially,
                              MetadataValue::from_ptr() reads each KV in-place
```

The `MetadataValue` class hierarchy (`PrimitiveValue`, `StringValue`, `ArrayValue`) tracks its own byte size, enabling the parser to advance the pointer by the exact number of bytes consumed. The same approach is used for tensor info entries (`TensorInfo::from_ptr()`).

### BPE Tokenizer Pipeline

1. **Pretokenization** — The input text is split by a PCRE2 JIT-compiled regex pattern. For Qwen2, the pattern handles English contractions (case-insensitive: `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`), Unicode letter sequences, digit groups of 1–3, special characters with optional leading whitespace, and various whitespace/line-break forms.

2. **Byte-to-Unicode** — Each byte 0–255 is mapped to a Unicode character following the GPT-2 convention: printable Latin-1 bytes (33–126, 161–172, 174–255) map directly; control characters and non-printables map to an offset range starting at U+0100.

3. **BPE Merge** — Starting from individual characters, the algorithm repeatedly finds the pair with the lowest rank in the merge table (loaded from `tokenizer.ggml.merges`) and merges them. This is the classic greedy BPE merge identical to GPT-2.

4. **Token ID Lookup** — Final subword tokens are hashed against `token_to_id_map_` (built from `tokenizer.ggml.tokens`) to produce integer token IDs.

### RAII Resource Management

All system resources are wrapped in `std::unique_ptr` with custom deleters:
- File descriptors (`close`)
- `mmap` regions (`munmap`)
- PCRE2 compiled patterns (`pcre2_code_free`)
- PCRE2 match data (`pcre2_match_data_free`)

This guarantees cleanup on scope exit, including during exceptions, without explicit `try`/`catch` blocks.

---

## Project Structure

```
inferlm/
├── include/zinferlm/          # Public API headers
│   ├── model_loader.h         #   Model base class, info structs
│   └── tokenizer.h            #   Tokenizer class, token_t
├── src/
│   ├── model_loader/
│   │   ├── loader.cpp         #   File type detection, factory
│   │   ├── gguf/
│   │   │   ├── gguf.cpp/h     #   GGUF binary parser (mmap, header, tensors)
│   │   │   ├── metadata.cpp/h #   GGUF metadata KV parser
│   │   │   └── tensors.cpp/h  #   Tensor info parser, GGMLType enum
│   │   └── models/
│   │       └── gguf.cpp/h     #   GGUFModel adapter
│   └── tokenizer/
│       ├── tokenizer.cpp      #   BPE tokenizer implementation
│       └── tokenizer.h        #   Pretokenizer regex patterns
├── tools/cli/
│   └── cli.cpp                #   CLI: inspect & tokenize subcommands
├── models/                    #   Bundled test model
├── CMakeLists.txt             #   Top-level build
├── LICENSE                    #   Apache 2.0
└── README.md
```

---

## Roadmap

- [ ] **Forward-pass engine** — transformer block implementation (self-attention, FFN, RMS norm, rotary embeddings)
- [ ] **Quantized dequantization kernels** — dispatch table from GGML type to optimized dequantize routines
- [ ] **KV cache** — with paged attention–style memory management
- [ ] **Sampling strategies** — greedy, temperature, top-k, top-p, min-p
- [ ] **Speculative decoding** — draft model head
- [ ] **Continuous batching** — dynamic batch scheduling
- [ ] **Metal/CUDA backends** — GPU offload for compute-bound layers

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
