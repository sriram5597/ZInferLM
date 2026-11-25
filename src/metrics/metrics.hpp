#include "zinferlm/types.h"
#include <chrono>
#include <memory>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

using Clock = std::chrono::steady_clock;

namespace zinferlm {
namespace internal {
class ZinferMetrics {
private:
  std::shared_ptr<prometheus::Registry> registry_;
  explicit ZinferMetrics();

public:
  static ZinferMetrics &instance();
  prometheus::Registry &get_registry();
  std::string scrape_metrics() const;
};

class ScopedObserver {
private:
  prometheus::Histogram &hist_;
  Clock::time_point start_;

public:
  ScopedObserver(prometheus::Histogram &h) : hist_(h), start_(Clock::now()) {}
  ScopedObserver(const ScopedObserver &obs) = delete;
  ScopedObserver operator=(const ScopedObserver &) = delete;

  ~ScopedObserver() {
    hist_.Observe(std::chrono::duration<double>(Clock::now() - start_).count());
  }
};
} // namespace internal
} // namespace zinferlm
