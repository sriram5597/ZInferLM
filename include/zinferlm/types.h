#pragma once

#include <chrono>

namespace zinferlm {

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

} // namespace zinferlm
