#include "inference/inference.h"
#include "worker/worker.hpp"
#include <cstdint>
#include <future>
#include <string>

struct inference_task_t {
  zinferlm::Inference &inference;
  std::string input;
  uint64_t max_tokens;
};

class InferenceWorker : public Worker<inference_task_t, std::string> {
private:
  void execute_task_(inference_task_t &task,
                     std::promise<std::string> &result) override;
  InferenceWorker() = default;

public:
  static InferenceWorker &instance();
};
