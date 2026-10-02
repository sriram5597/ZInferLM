#include "chat_message.h"
#include "sampling/samplers.h"
#include "zinferlm/models.h"
#include <zinferlm/chat.h>
#include <zinferlm/tokenizer.h>

std::string zinferlm::Chat::invoke(std::string input) {
  zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(model_);
  zinferlm::tokenizer_info_t tok_info = model_.tokenizer_info();
  ChatTemplate chat_template = ChatTemplate::from_str(
      tok_info.template_str, tokenizer.get_bos(), tokenizer.get_eos());

  int max_tokens = 8192;
  UserMessage msg = UserMessage(input);
  std::vector<ChatMessage> messages = {msg};
  std::string chat_message = chat_template.render(messages, true);
  std::string output = model_.invoke(chat_message, max_tokens);
  return output;
}
