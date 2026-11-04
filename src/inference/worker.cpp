#include <exception>
#include <future>
#include <string>

#include "inference/inference.h"
#include "worker.hpp"

namespace zinferlm {
namespace internal {

InferenceWorker &InferenceWorker::instance() {
  static InferenceWorker worker;
  return worker;
}

void InferenceWorker::execute_task_(inference_task_t &task,
                                    std::promise<std::string> &promise) {
  try {
    Inference inference(task.model, task.run_id);
    inference.set_debug(task.debug);
    inference.set_sampler(task.sampler);
    std::string out = inference.invoke(task.input, task.max_tokens);
    promise.set_value(out);
  } catch (...) {
    promise.set_exception(std::current_exception());
  }
}

} // namespace internal
} // namespace zinferlm
