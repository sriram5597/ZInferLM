#include <cstdint>
#include <future>
#include <string>
#include <vector>

#include "scheduler.hpp"
#include "worker.hpp"
#include "zinferlm/events.h"
#include "zinferlm/models.h"
#include "zinferlm/tokenizer.h"

namespace zinferlm {
namespace internal {

InferenceWorker::InferenceWorker() {}

InferenceWorker &InferenceWorker::instance() {
  static InferenceWorker worker;
  return worker;
}

std::vector<int32_t> InferenceWorker::tokenize(std::string input) {
  events::EventDispatcher &dispatcher = events::EventDispatcher::get_instance();
  Tokenizer tokenizer = Tokenizer::for_model(zinferlm::Model::instance());
  std::string output = "";
  // dispatcher.dispatch(run_id_, events::InferenceEvent::TOKENIZER_STARTED,
  //                     events::tokenizer_started_event_t{
  //                         .str_len = input.length(),
  //                     });
  std::vector<int32_t> tokens = tokenizer.tokenize(input);
  // dispatcher.dispatch(run_id_, events::InferenceEvent::TOKENIZER_COMPLETED,
  //                     events::tokenizer_completed_event_t{
  //                         .num_tokens = tokens.size(),
  //                     });
  return tokens;
}

void InferenceWorker::execute_task_() {
  std::pair<inference_task_t, std::promise<std::string>> task =
      std::move(get_task_());
  std::vector<int32_t> tokens = tokenize(task.first.input);
  seq_info_t seq_info = {.seq_id = task.first.task_id,
                         .past_tokens = 0,
                         .seq_len = 0,
                         .max_tokens = task.first.max_tokens,
                         .sample_config = task.first.sample_config,
                         .promise = std::move(task.second),
                         .inp_tokens = tokens};
  InferenceScheduler &scheduler = InferenceScheduler::instance();
  scheduler.submit(std::move(seq_info));
}
} // namespace internal
} // namespace zinferlm
