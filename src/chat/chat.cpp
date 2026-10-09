#include "chat_message.h"
#include "inference/inference.h"
#include "sampling/samplers.h"
#include "zinferlm/models.h"
#include <zinferlm/chat.h>
#include <zinferlm/tokenizer.h>

zinferlm::ChatMessage
zinferlm::Chat::invoke(std::vector<ChatMessage>& messages) {
  zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(model_);
  zinferlm::tokenizer_info_t tok_info = model_.tokenizer_info();
  ChatTemplate chat_template = ChatTemplate::from_str(
      tok_info.template_str, tokenizer.get_bos(), tokenizer.get_eos());

  int max_tokens = model_.config().max_context_len;
  std::string chat_message = chat_template.render(messages, true);

  zinferlm::Inference inference(model_);
  inference.set_debug(debug_);
  std::string output = inference.invoke(chat_message, max_tokens);
  return AssistantMessage(output);
}
