#include <string>
#include <vector>
#include <zinferlm/chat.h>

class ChatTemplate {
  private:
  std::string template_str_;
  ChatTemplate(std::string t): template_str_(t) {}

  public:
  static ChatTemplate from_str(std::string template_str);
  std::string render(std::vector<zinferlm::chat_message_t> messages);
};

