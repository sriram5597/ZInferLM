#include <boost/beast.hpp>
#include <boost/beast/http/status.hpp>
#include <boost/json.hpp>
#include <boost/json/object.hpp>
#include <boost/json/value.hpp>
#include <future>
#include <iostream>
#include <memory>
#include <vector>

#include "handlers.hpp"
#include "server.hpp"
#include "zinferlm/chat.h"
#include "zinferlm/events.h"
#include "zinferlm/models.h"

namespace json = boost::json;
namespace http = boost::beast::http;

void health_check(Session &session) {
  json::object obj = {{"status", "healthy"}};
  session.send_json_response(http::status::ok, obj);
}

std::string generate_chatcmpl_id(size_t length = 29) {
  static constexpr char charset[] = "0123456789"
                                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                    "abcdefghijklmnopqrstuvwxyz";

  // Non-deterministic seed device
  std::random_device rd;
  std::mt19937_64 rng(rd());
  std::uniform_int_distribution<size_t> dist(0, sizeof(charset) - 2);

  std::string random_suffix;
  random_suffix.reserve(length);

  for (size_t i = 0; i < length; ++i) {
    random_suffix += charset[dist(rng)];
  }

  return "chatcmpl-" + random_suffix;
}

void chat_completion(Session &session, json::object req_payload) {
  bool streaming = true;
  auto messages_it = req_payload.find("messages");
  if (messages_it == req_payload.end() || !messages_it->value().is_array()) {
    session.send_json_response(
        http::status::bad_request,
        {{"error", R"("Field: "messages" is required)"}});
    return;
  }
  std::vector<zinferlm::ChatMessage> chat_messages;
  for (const auto &msg : messages_it->value().as_array()) {
    if (msg.is_object()) {
      const auto &obj = msg.as_object();
      if (!obj.contains("role")) {
        session.send_json_response(
            http::status::bad_request,
            {{"error", R"("Field: "role" is required)"}});
        return;
      }
      if (!obj.contains("content")) {
        session.send_json_response(
            http::status::bad_request,
            {{"error", R"("Field: "content" is required)"}});
        return;
      }
      const auto &role_it = obj.find("role");
      const auto &content_it = obj.find("content");
      if (role_it->value().as_string() == "user") {
        chat_messages.push_back(zinferlm::UserMessage(
            json::value_to<std::string>(content_it->value())));
      } else if (role_it->value().as_string() == "assistant") {
        chat_messages.push_back(zinferlm::AssistantMessage(
            json::value_to<std::string>(content_it->value())));
      } else if (role_it->value().as_string() == "system") {
        chat_messages.push_back(zinferlm::SystemMessage(
            json::value_to<std::string>(content_it->value())));
      }
    }
  }

  auto current_session = session.shared_from_this();
  std::string chat_id = generate_chatcmpl_id();

  zinferlm::events::event_callback_t cb =
      [current_session, &chat_id](zinferlm::events::inference_event_t event) {
        if (auto generated_event =
                std::get_if<zinferlm::events::token_generated_event_t>(
                    &event)) {
          current_session->stream_json_message(
              {{"role", "assistant"}, {"content", generated_event->token}},
              false);
        }
        if (auto completed_event =
                std::get_if<zinferlm::events::generation_completed_event_t>(
                    &event)) {
          zinferlm::StreamStatus status = completed_event->status;
          json::value payload;
          if (status == zinferlm::StreamStatus::MAX_CTX_REACHED) {
            payload = {{"finish_reason", "Maximum Tokens reached"}};
          }
          if (status == zinferlm::StreamStatus::EOS) {
            payload = {{"finish_reason", "Stop"}};
          }
          current_session->stream_json_message(payload, true);
          zinferlm::events::EventDispatcher::get_instance().unlisten(chat_id);
        }
      };

  zinferlm::events::EventDispatcher &dispatcher =
      zinferlm::events::EventDispatcher::get_instance();

  zinferlm::chat_params_t params = {
      .max_tokens = 8192, .temperature = 0.9, .top_p = 0.7};

  if (streaming) {
    session.initiate_streaming();
    std::set<zinferlm::events::InferenceEvent> events = {
        zinferlm::events::InferenceEvent::GENERATION_COMPLETED,
        zinferlm::events::InferenceEvent::TOKEN_GENERATED};
    dispatcher.listen(chat_id, cb, events);
  }

  zinferlm::Chat chat(zinferlm::Model::instance(), chat_id);
  std::future<std::string> result = chat.invoke(chat_messages, params);

  if (!streaming) {
    std::string output = result.get();
    current_session->send_json_response(
        http::status::ok, {{"role", "assistant"}, {"content", output}});
  }
}
