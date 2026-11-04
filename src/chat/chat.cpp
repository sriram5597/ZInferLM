#include <future>
#include <string>
#include <utility>

#include <zinferlm/chat.h>
#include <zinferlm/tokenizer.h>

#include "chat_message.h"
#include "inference/inference.h"
#include "inference/worker.hpp"
#include "sampling/samplers.h"

namespace zinferlm {

std::future<std::string> Chat::invoke(std::vector<ChatMessage> &messages,
                                      const chat_params_t &params) {
  zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(model_);
  zinferlm::tokenizer_info_t tok_info = model_.tokenizer_info();
  internal::ChatTemplate chat_template = internal::ChatTemplate::from_str(
      tok_info.template_str, tokenizer.get_bos(), tokenizer.get_eos());

  std::string chat_message = chat_template.render(messages, true);

  internal::Sampler sampler(
      internal::sampler_params_t{.temperature = params.temperature,
                                 .top_k = params.top_k,
                                 .top_p = params.top_p});

  internal::InferenceWorker &worker = internal::InferenceWorker::instance();
  std::promise<std::string> out_promise;
  std::future<std::string> future = out_promise.get_future();
  internal::inference_task_t task = {.run_id = chat_id_,
                                     .model = model_,
                                     .debug = debug_,
                                     .sampler = sampler,
                                     .input = std::move(chat_message),
                                     .max_tokens = params.max_tokens};
  worker.add_task(task, std::move(out_promise));
  return future;
}

} // namespace zinferlm
