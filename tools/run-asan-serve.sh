#!/usr/bin/env bash
set -euo pipefail

# Run zinferlm-serve under AddressSanitizer.
#
# Usage:
#   tools/run-asan-serve.sh [--verify|--smoke] [model_path]
#
#   (no args)            run the server in the foreground (Ctrl-C to stop;
#                        LeakSanitizer reports on graceful shutdown)
#   --smoke              start the server, hit /health and /v1/chat/completions,
#                        then SIGINT it and tail the ASan report
#   --verify             confirm ASan is wired up and working: check linkage,
#                        runtime resolution, run an injected-use-after-free
#                        sanity test, then smoke-test the server and scan its
#                        log for findings (exits non-zero if anything fails)
#   <model_path>         run against a specific .gguf model
#
# The ASan runtime (libasan) is taken from the local copy under .asan-libs/ when
# the system libasan package is not installed; otherwise the system copy is used.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BIN="$ROOT/build-asan/bin/zinferlm-serve"
MODEL="${MODEL:-$ROOT/model_files/qwen2.5-1.5b-instruct-q5_k_m.gguf}"
ASAN_LIB_DIR="$ROOT/.asan-libs/usr/lib64"
LOG="${TMPDIR:-/tmp}/opencode/asan-server.log"

# -- shared library search path --
# build-asan/lib holds libzinferlm.so / libggml*.so; the .asan-libs entry makes
# the locally-downloaded libasan.so.8 resolvable when it isn't installed system-wide.
export LD_LIBRARY_PATH="$ROOT/build-asan/lib"
if [ -f "$ASAN_LIB_DIR/libasan.so.8" ]; then
  export LD_LIBRARY_PATH="$ASAN_LIB_DIR:$LD_LIBRARY_PATH"
fi

# -- linker search path (for compiling the sanity test) --
LINK_LIB_PATH="$ROOT/build-asan/lib"
if [ -f "$ASAN_LIB_DIR/libasan.so.8" ]; then
  LINK_LIB_PATH="$ASAN_LIB_DIR:$LINK_LIB_PATH"
fi

export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1:halt_on_error=0:abort_on_error=0:print_summary=1}"

if [ ! -x "$BIN" ]; then
  echo "error: ASan binary not found: $BIN" >&2
  echo "Build it first:" >&2
  echo "  cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON" >&2
  echo "  LIBRARY_PATH=$ASAN_LIB_DIR cmake --build build-asan --target zinferlm-serve" >&2
  exit 1
fi

if [ ! -f "$MODEL" ]; then
  echo "error: model not found: $MODEL" >&2
  exit 1
fi

# Start the server, exercise it, then SIGINT for a graceful shutdown. The ASan /
# LeakSanitizer report ends up in $LOG.
smoke_run() {
  mkdir -p "$(dirname "$LOG")"
  "$BIN" "$MODEL" >"$LOG" 2>&1 &
  PID=$!
  trap 'kill -INT $PID 2>/dev/null || true; wait $PID 2>/dev/null || true' EXIT

  echo "  server pid=$PID (log: $LOG)"
  for _ in $(seq 1 120); do
    grep -q "HTTP Server Listening" "$LOG" 2>/dev/null && break
    sleep 1
  done

  curl -s -m 5 http://127.0.0.1:8080/health >/dev/null || true
  curl -s -m 12 -N http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hi."}]}' 2>/dev/null | head -c 800 >/dev/null || true

  sleep 3
  kill -INT "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
}

# 1. The binary and shared library must actually link libasan and carry __asan
#    instrumentation symbols.
verify_linkage() {
  echo "=== 1. linkage check ==="
  local f
  for f in "$BIN" "$ROOT/build-asan/lib/libzinferlm.so"; do
    if readelf -d "$f" 2>/dev/null | grep -q 'libasan\.so'; then
      echo "  ok: $(basename "$f") links libasan"
    else
      echo "  FAIL: $(basename "$f") does not link libasan" >&2
      return 1
    fi
  done
  local asan_syms
  asan_syms="$(nm -D "$BIN" 2>/dev/null | grep -c __asan)"
  if [ "${asan_syms:-0}" -gt 0 ]; then
    echo "  ok: __asan symbols present in the binary ($asan_syms)"
  else
    echo "  FAIL: no __asan symbols (binary is not instrumented)" >&2
    return 1
  fi
}

# 2. libasan must resolve at runtime to a real path (catches LD_LIBRARY_PATH bugs).
verify_runtime() {
  echo "=== 2. runtime resolution ==="
  if ldd "$BIN" 2>/dev/null | grep 'libasan' | grep -q '=> /'; then
    echo "  ok: libasan resolves to $(ldd "$BIN" | awk '/libasan\.so/{print $3}')"
  else
    echo "  FAIL: libasan.so.8 not found at runtime (check LD_LIBRARY_PATH)" >&2
    return 1
  fi
}

# 3. Prove ASan catches a known bug: an injected heap-use-after-free.
verify_detection() {
  echo "=== 3. detection sanity test (injected use-after-free) ==="
  local src="$LOG.verify-uaf.cpp" exe="$LOG.verify-uaf" out
  printf 'int main(){int*p=new int[4];delete[]p;return p[2];}\n' > "$src"
  if ! LIBRARY_PATH="$LINK_LIB_PATH" g++ -fsanitize=address "$src" -o "$exe" 2>/dev/null; then
    echo "  FAIL: could not compile the sanity test (g++ -fsanitize=address)" >&2
    return 1
  fi
  out="$("$exe" 2>&1 || true)"
  if echo "$out" | grep -q "ERROR: AddressSanitizer"; then
    echo "  ok: ASan reported the injected use-after-free"
  else
    echo "  FAIL: ASan did NOT report the injected use-after-free" >&2
    echo "$out" | head -n 5 >&2
    return 1
  fi
}

# 5. Scan the server log for ASan / LeakSanitizer findings.
scan_report() {
  echo "=== 5. report scan ==="
  local findings
  findings="$(grep -E "ERROR: AddressSanitizer|ERROR: LeakSanitizer|SUMMARY:" "$LOG" || true)"
  if [ -n "$findings" ]; then
    echo "$findings"
    echo "  verdict: ASan findings DETECTED"
    return 1
  fi
  echo "  no ASan / LeakSanitizer findings in the log"
  echo "  verdict: clean"
  return 0
}

case "${1:-}" in
  --verify)
    rc=0
    verify_linkage   || rc=1
    verify_runtime   || rc=1
    verify_detection || rc=1
    echo "=== 4. server smoke run ==="
    smoke_run
    scan_report || rc=1
    echo
    if [ "$rc" -eq 0 ]; then
      echo "VERIFY PASSED"
    else
      echo "VERIFY FAILED"
    fi
    exit "$rc"
    ;;
  --smoke)
    smoke_run
    echo "=== server log (tail) ==="
    tail -n 80 "$LOG"
    ;;
  *)
    exec "$BIN" "${1:-$MODEL}"
    ;;
esac
