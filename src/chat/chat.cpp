#include "chat_message.h"
#include "inference/inference.h"
#include "inference/worker.hpp"
#include "sampling/samplers.h"
#include "zinferlm/models.h"
#include <future>
#include <iostream>
#include <string>
#include <utility>
#include <zinferlm/chat.h>
#include <zinferlm/tokenizer.h>

zinferlm::ChatMessage zinferlm::Chat::invoke(std::vector<ChatMessage> &messages,
                                             const chat_params_t &params) {
  zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(model_);
  zinferlm::tokenizer_info_t tok_info = model_.tokenizer_info();
  ChatTemplate chat_template = ChatTemplate::from_str(
      tok_info.template_str, tokenizer.get_bos(), tokenizer.get_eos());

  int max_tokens = model_.config().max_context_len;
  std::string chat_message = chat_template.render(messages, true);

  Inference inference(model_);
  inference.set_debug(debug_);
  Sampler sampler(sampler_params_t{.temperature = params.temperature,
                                   .top_k = params.top_k,
                                   .top_p = params.top_p});
  inference.set_sampler(sampler);
  InferenceWorker &worker = InferenceWorker::instance();
  std::promise<std::string> out_promise;
  std::future<std::string> future = out_promise.get_future();
  inference_task_t task = {.inference = inference,
                           .input = chat_message,
                           .max_tokens = params.max_tokens};
  worker.add_task(task, std::move(out_promise));
  std::string result = future.get();
  std::cout << "Output: " << result << std::endl;
  return AssistantMessage(result);
}
