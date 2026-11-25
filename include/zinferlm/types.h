#pragma once

#include <chrono>

namespace zinferlm {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

} // namespace zinferlm
