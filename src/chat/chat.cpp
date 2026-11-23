#include <cinttypes>
#include <future>
#include <memory>
#include <string>
#include <utility>

#include <zinferlm/chat.h>
#include <zinferlm/tokenizer.h>

#include "chat_message.h"
#include "engine/engine.h"
#include "inference/worker.hpp"
#include "logging/log.h"
#include "sampling/samplers.h"

namespace zinferlm {

std::future<std::string> Chat::invoke(std::vector<ChatMessage> &messages,
                                      const chat_params_t &params) {
  LOG_INF("chat request: chat_id=%s messages=%zu max_tokens=%u "
          "temperature=%.2f top_p=%.2f\n",
          params.chat_id.c_str(), messages.size(), params.max_tokens,
          params.temperature, params.top_p);
  zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(model_);
  zinferlm::tokenizer_info_t tok_info = model_.tokenizer_info();
  internal::ChatTemplate chat_template = internal::ChatTemplate::from_str(
      tok_info.template_str, tokenizer.get_bos(), tokenizer.get_eos());

  std::string chat_message = chat_template.render(messages, true);
  LOG_DBG("rendered prompt: chat_id=%s chars=%zu\n", params.chat_id.c_str(),
          chat_message.size());

  internal::sampler_params_t sampler_params =
      internal::sampler_params_t{.temperature = params.temperature,
                                 .top_k = params.top_k,
                                 .top_p = params.top_p};

  // Hold shared ownership: a concurrent shutdown must not free the worker
  // between this lookup and add_task(), which rejects new work itself.
  std::shared_ptr<internal::InferenceWorker> worker =
      internal::Engine::instance().worker();
  std::promise<std::string> out_promise;
  std::future<std::string> future = out_promise.get_future();
  internal::inference_task_t task = {.task_id = params.chat_id,
                                     .sample_config = sampler_params,
                                     .input = std::move(chat_message),
                                     .max_tokens = params.max_tokens};
  worker->add_task(task, std::move(out_promise));
  return future;
}

void Chat::stop() { internal::Engine::instance().shutdown(); }

} // namespace zinferlm
