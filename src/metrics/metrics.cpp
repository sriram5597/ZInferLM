#include <memory>
#include <prometheus/registry.h>
#include <prometheus/serializer.h>
#include <prometheus/text_serializer.h>
#include <string>
#include <zinferlm/metrics.hpp>

#include "metrics.hpp"

namespace zinferlm {
namespace internal {
ZinferMetrics::ZinferMetrics() {
  registry_ = std::make_shared<prometheus::Registry>();
}

ZinferMetrics &ZinferMetrics::instance() {
  static ZinferMetrics metrics;
  return metrics;
}

prometheus::Registry &ZinferMetrics::get_registry() { return *registry_; }

std::string ZinferMetrics::scrape_metrics() const {
  prometheus::TextSerializer serializer;
  return serializer.Serialize(registry_->Collect());
}

} // namespace internal
std::string get_metrics() {
  return internal::ZinferMetrics::instance().scrape_metrics();
}
} // namespace zinferlm
