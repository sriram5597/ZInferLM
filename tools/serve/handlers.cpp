#include <boost/beast.hpp>
#include <boost/beast/http/status.hpp>
#include <boost/json.hpp>
#include <boost/json/object.hpp>
#include <boost/json/value.hpp>
#include <iostream>
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

void chat_completion(Session &session, json::object req_payload) {
  bool streaming = true;
  auto messages_it = req_payload.find("messages");
  if (messages_it == req_payload.end() || !messages_it->value().is_array()) {
    session.send_json_response(
        http::status::bad_request,
        {{"error", R"("Field: "messages" is required)"}});
  }
  std::vector<zinferlm::ChatMessage> chat_messages;
  for (const auto &msg : messages_it->value().as_array()) {
    if (msg.is_object()) {
      const auto &obj = msg.as_object();
      if (!obj.contains("role")) {
        session.send_json_response(
            http::status::bad_request,
            {{"error", R"("Field: "role" is required)"}});
      }

      if (!obj.contains("content")) {
        session.send_json_response(
            http::status::bad_request,
            {{"error", R"("Field: "content" is required)"}});
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

  zinferlm::events::event_callback_t cb =
      [&session](zinferlm::events::inference_event_t event) -> void {
    if (auto generated_event =
            std::get_if<zinferlm::events::token_generated_event_t>(&event)) {
      session.stream_json_message(
          {{"role", "assistant"}, {"content", generated_event->token}}, false);
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
      session.stream_json_message(payload, true);
    }
  };

  int observer_id = -1;
  zinferlm::events::EventDispatcher &dispatcher =
      zinferlm::events::EventDispatcher::get_instance();

  if (streaming) {

    session.initiate_streaming();
    std::set<zinferlm::events::InferenceEvent> events = {
        zinferlm::events::InferenceEvent::GENERATION_COMPLETED,
        zinferlm::events::InferenceEvent::TOKEN_GENERATED};
    observer_id = dispatcher.listen(cb, events);
  }
  zinferlm::chat_params_t params = {
      .max_tokens = 8192, .temperature = 0.9, .top_p = 0.7};
  zinferlm::Chat chat(zinferlm::Model::instance());
  zinferlm::ChatMessage output = chat.invoke(chat_messages, params);
  if (streaming) {
    dispatcher.unlisten(observer_id);
  }

  session.send_json_response(
      http::status::ok, {{"role", output.role}, {"content", output.message}});
}
