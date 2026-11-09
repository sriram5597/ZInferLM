#include <exception>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "inference/inference.h"
#include "kv_cache/cache.h"
#include "models/model_impl.h"
#include "worker.hpp"
#include "zinferlm/events.h"
#include "zinferlm/models.h"
#include "zinferlm/tokenizer.h"

namespace zinferlm {
namespace internal {

InferenceWorker::InferenceWorker() {
  zinferlm::model_config_t cfg = zinferlm::Model::instance().config();
  ModelImpl &model_impl = zinferlm::Model::instance().impl();
  cache_ = std::make_unique<KVCache>(
      model_impl.get_backend(), cfg.max_context_len, cfg.n_blocks, cfg.nkv,
      cfg.embedding_dim / cfg.nheads, GGML_TYPE_F16);
  inference_ =
      std::make_unique<Inference>(cache_.get(), zinferlm::Model::instance());
}

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

void InferenceWorker::execute_task_(inference_task_t &task,
                                    std::promise<std::string> &promise) {
  try {
    inference_->set_debug(task.debug);
    inference_->set_sampler(task.sampler);
    std::string out = inference_->invoke(task.input, task.max_tokens);
    promise.set_value(out);
  } catch (...) {
    promise.set_exception(std::current_exception());
  }
}

} // namespace internal
} // namespace zinferlm
