#include "worker.hpp"
#include "inference/inference.h"
#include "zinferlm/models.h"
#include <exception>
#include <future>
#include <string>

InferenceWorker &InferenceWorker::instance() {
  static InferenceWorker worker;
  return worker;
}

void InferenceWorker::execute_task_(inference_task_t &task,
                                    std::promise<std::string> &promise) {
  try {
    zinferlm::Inference &inference = task.inference;
    std::string out = inference.invoke(task.input, task.max_tokens);
    promise.set_value(out);
  } catch (...) {
    promise.set_exception(std::current_exception());
  }
}
