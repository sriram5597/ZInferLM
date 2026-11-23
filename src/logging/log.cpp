// Adapted from llama.cpp common/log.cpp
// Copyright (c) 2023-2024 The llama.cpp authors
// SPDX-License-Identifier: MIT
//
// Ported to ZInferLM: lives in zinferlm::internal, JSONL support removed,
// TTY color detection inlined (no dependency on llama.cpp's common.h).

#include "log.h"

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace zinferlm {
namespace internal {

static int initial_verbosity_thold() {
    if (const char * env = getenv("ZINFERLM_LOG_VERBOSITY")) {
        return atoi(env);
    }
    return LOG_DEFAULT_INFO;
}

static int log_verbosity_thold = initial_verbosity_thold();

int log_get_verbosity_thold(void) {
    return log_verbosity_thold;
}

void log_set_verbosity_thold(int verbosity) {
    log_verbosity_thold = verbosity;
}

static int64_t t_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static bool tty_can_use_colors() {
#ifdef _WIN32
    return false;
#else
    const char * term = getenv("TERM");
    return isatty(fileno(stderr)) && term && strcmp(term, "dumb") != 0;
#endif
}

// colors
enum log_col : int {
    LOG_COL_IDX_DEFAULT = 0,
    LOG_COL_IDX_BOLD,
    LOG_COL_IDX_RED,
    LOG_COL_IDX_GREEN,
    LOG_COL_IDX_YELLOW,
    LOG_COL_IDX_BLUE,
    LOG_COL_IDX_MAGENTA,
    LOG_COL_IDX_CYAN,
    LOG_COL_IDX_WHITE,
};

// disable colors by default
static const char * g_col[] = {
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
};

struct log_entry {
    enum ggml_log_level level {GGML_LOG_LEVEL_INFO};

    std::vector<char> msg;

    int64_t timestamp { 0 };
    bool is_end       { false }; // signals the worker thread to stop
    bool prefix       { false };

    log_entry(size_t size = 256) : msg(size) { }

    void print(FILE * file = nullptr) const {
        FILE * fcur = file;
        if (!fcur) {
            // stderr displays DBG messages only when their verbosity level is not higher than the threshold
            // these messages will still be logged to a file
            if (level == GGML_LOG_LEVEL_DEBUG && log_verbosity_thold < LOG_DEFAULT_DEBUG) {
                return;
            }

            fcur = stdout;

            if (level != GGML_LOG_LEVEL_NONE) {
                fcur = stderr;
            }
        }

        if (level != GGML_LOG_LEVEL_NONE && level != GGML_LOG_LEVEL_CONT && prefix) {
            if (timestamp) {
                // [M.s.ms.us]
                fprintf(fcur, "%s%d.%02d.%03d.%03d%s ",
                        g_col[LOG_COL_IDX_BLUE],
                        (int) (timestamp / 1000000 / 60),
                        (int) (timestamp / 1000000 % 60),
                        (int) (timestamp / 1000 % 1000),
                        (int) (timestamp % 1000),
                        g_col[LOG_COL_IDX_DEFAULT]);
            }

            switch (level) {
                case GGML_LOG_LEVEL_INFO:  fprintf(fcur, "%sI %s", g_col[LOG_COL_IDX_GREEN],   g_col[LOG_COL_IDX_DEFAULT]); break;
                case GGML_LOG_LEVEL_WARN:  fprintf(fcur, "%sW %s", g_col[LOG_COL_IDX_MAGENTA], ""                          ); break;
                case GGML_LOG_LEVEL_ERROR: fprintf(fcur, "%sE %s", g_col[LOG_COL_IDX_RED],     ""                          ); break;
                case GGML_LOG_LEVEL_DEBUG: fprintf(fcur, "%sD %s", g_col[LOG_COL_IDX_YELLOW],  ""                          ); break;
                default:
                    break;
            }
        }

        // the reset goes before the trailing newlines, so that every line carries its own colors
        const bool reset = level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_DEBUG;

        size_t end = strlen(msg.data());
        while (end > 0 && msg[end - 1] == '\n') {
            end--;
        }

        fprintf(fcur, "%.*s%s%s", (int) end, msg.data(), reset ? g_col[LOG_COL_IDX_DEFAULT] : "", msg.data() + end);

        fflush(fcur);
    }
};

struct Log {
    // default capacity
    Log(size_t capacity = 512) {
        file       = nullptr;
        stdo       = true;
        colors     = false;
        prefix     = false;
        timestamps = false;
        running    = false;
        t_start    = t_us();

        queue.resize(capacity, log_entry(256));
        head = 0;
        tail = 0;

        resume();
    }

    ~Log() {
        pause();
        if (file) {
            fclose(file);
        }
    }

private:
    std::mutex              mtx;
    std::thread             thrd;
    std::condition_variable cv_new;  // new entry
    std::condition_variable cv_full; // wait on full

    FILE * file;

    bool stdo;
    bool colors;
    bool prefix;
    bool timestamps;
    bool running;

    int64_t t_start;

    // queue of entries
    std::vector<log_entry> queue;
    size_t head;
    size_t tail;

    bool print_entry(const log_entry & e) const {
        if (e.is_end) return true;

        if (stdo) {
            e.print();
        }
        if (file) {
            e.print(file);
        }
        return false;
    }

    bool flush_queue(size_t start_head, size_t end_tail, size_t & out_head) const {
        bool stop = false;
        size_t h = start_head;
        while (h != end_tail && !stop) {
            stop = print_entry(queue[h]);
            h = (h + 1) % queue.size();
        }
        out_head = h;
        return stop;
    }

public:
    bool is_full() const {
        return ((tail + 1) % queue.size()) == head;
    }

    bool is_empty() const {
        return head == tail;
    }

    void add(enum ggml_log_level level, const char * fmt, va_list args) {
        std::unique_lock<std::mutex> lock(mtx);

        // block if the queue is full
        cv_full.wait(lock, [this]() { return !running || !is_full(); });

        if (!running) {
            // discard messages while the worker thread is paused
            return;
        }

        auto & entry = queue[tail];

        {
            // cannot use args twice, so make a copy in case we need to expand the buffer
            va_list args_copy;
            va_copy(args_copy, args);

            const size_t n = vsnprintf(entry.msg.data(), entry.msg.size(), fmt, args);
            if (n >= entry.msg.size()) {
                entry.msg.resize(n + 1);
                vsnprintf(entry.msg.data(), entry.msg.size(), fmt, args_copy);
            }

            va_end(args_copy);
        }

        entry.is_end    = false;
        entry.level     = level;
        entry.prefix    = prefix;
        entry.timestamp = 0;
        if (timestamps) {
            entry.timestamp = t_us() - t_start;
        }

        tail = (tail + 1) % queue.size();
        cv_new.notify_one();
    }

    void resume() {
        std::lock_guard<std::mutex> lock(mtx);

        if (running) {
            return;
        }

        running = true;

        thrd = std::thread([this]() {
            while (true) {
                std::unique_lock<std::mutex> lock(mtx);
                cv_new.wait(lock, [this]() { return !is_empty(); });

                size_t cached_head = head;
                size_t cached_tail = tail;

                lock.unlock(); // drop the lock during flush

                size_t next_head;
                bool stop = flush_queue(cached_head, cached_tail, next_head);

                lock.lock();
                head = next_head;
                cv_full.notify_all();

                if (stop) {
                    break;
                }
            }
        });
    }

    void pause() {
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (!running) {
                return;
            }

            running = false;

            // push an entry to signal the worker thread to stop
            auto & entry = queue[tail];
            entry.is_end = true;
            tail = (tail + 1) % queue.size();

            // wakeup everyone
            cv_new.notify_one();
            cv_full.notify_all();
        }

        thrd.join();
    }

    void set_file(const char * path) {
        pause();

        if (file) {
            fclose(file);
        }

        if (path) {
            file = fopen(path, "w");
        } else {
            file = nullptr;
        }

        resume();
    }

    void set_stdo(bool enabled) {
        pause();

        stdo = enabled;

        resume();
    }

    bool get_colors() const {
        return colors;
    }

    void set_colors(bool colors) {
        pause();

        this->colors = colors;

        if (this->colors) {
            g_col[LOG_COL_IDX_DEFAULT] = LOG_COL_DEFAULT;
            g_col[LOG_COL_IDX_BOLD]    = LOG_COL_BOLD;
            g_col[LOG_COL_IDX_RED]     = LOG_COL_RED;
            g_col[LOG_COL_IDX_GREEN]   = LOG_COL_GREEN;
            g_col[LOG_COL_IDX_YELLOW]  = LOG_COL_YELLOW;
            g_col[LOG_COL_IDX_BLUE]    = LOG_COL_BLUE;
            g_col[LOG_COL_IDX_MAGENTA] = LOG_COL_MAGENTA;
            g_col[LOG_COL_IDX_CYAN]    = LOG_COL_CYAN;
            g_col[LOG_COL_IDX_WHITE]   = LOG_COL_WHITE;
        } else {
            for (size_t i = 0; i < sizeof(g_col) / sizeof(g_col[0]); i++) {
                g_col[i] = "";
            }
        }

        resume();
    }

    void set_prefix(bool prefix) {
        std::lock_guard<std::mutex> lock(mtx);

        this->prefix = prefix;
    }

    void set_timestamps(bool timestamps) {
        std::lock_guard<std::mutex> lock(mtx);

        this->timestamps = timestamps;
    }
};

//
// public API
//

struct Log * log_init() {
    return new Log;
}

struct Log * log_main() {
    // We intentionally leak (i.e. do not delete) the logger singleton because
    // a destructor called at DLL teardown phase can cause hangs on Windows.
    // The OS will release resources anyway, so it should not be a significant
    // issue, though this design may cause logs to be lost if not flushed
    // before the program exits.
    // Refer to https://github.com/ggml-org/llama.cpp/issues/22142 for details.
    static struct Log * log;
    static std::once_flag init_flag;
    std::call_once(init_flag, [&]() {
        log = new Log;
        // Set default to auto-detect colors
        log->set_colors(tty_can_use_colors());
    });

    return log;
}

void log_pause(struct Log * log) {
    log->pause();
}

void log_resume(struct Log * log) {
    log->resume();
}

void log_free(struct Log * log) {
    delete log;
}

void log_add(struct Log * log, enum ggml_log_level level, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log->add(level, fmt, args);
    va_end(args);
}

void log_set_file(struct Log * log, const char * file) {
    log->set_file(file);
}

void log_set_stdo(struct Log * log, bool enabled) {
    log->set_stdo(enabled);
}

void log_set_colors(struct Log * log, log_colors colors) {
    if (colors == LOG_COLORS_AUTO) {
        log->set_colors(tty_can_use_colors());
        return;
    }

    if (colors == LOG_COLORS_DISABLED) {
        log->set_colors(false);
        return;
    }

    assert(colors == LOG_COLORS_ENABLED);
    log->set_colors(true);
}

bool log_get_colors(struct Log * log) {
    return log->get_colors();
}

void log_set_prefix(struct Log * log, bool prefix) {
    log->set_prefix(prefix);
}

void log_set_timestamps(struct Log * log, bool timestamps) {
    log->set_timestamps(timestamps);
}

void log_flush(struct Log * log) {
    log->pause();
    log->resume();
}

int log_get_verbosity(enum ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: return LOG_LEVEL_DEBUG;
        case GGML_LOG_LEVEL_INFO:  return LOG_LEVEL_TRACE;
        case GGML_LOG_LEVEL_WARN:  return LOG_LEVEL_WARN;
        case GGML_LOG_LEVEL_ERROR: return LOG_LEVEL_ERROR;
        case GGML_LOG_LEVEL_CONT:  return LOG_LEVEL_TRACE;
        case GGML_LOG_LEVEL_NONE:
        default:
            return LOG_LEVEL_OUTPUT;
    }
}

void log_default_callback(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    auto verbosity = log_get_verbosity(level);
    if (verbosity <= log_verbosity_thold) {
        log_add(log_main(), level, "%s", text);
    }
}

} // namespace internal
} // namespace zinferlm
