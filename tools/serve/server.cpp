#include <boost/asio/buffer.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/address.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/signal_set.hpp>
#include <boost/asio/socket_base.hpp>
#include <boost/asio/strand.hpp>
#include <boost/asio/write.hpp>
#include <boost/beast/core/error.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/http/empty_body.hpp>
#include <boost/beast/http/field.hpp>
#include <boost/beast/http/impl/write.hpp>
#include <boost/beast/http/message_fwd.hpp>
#include <boost/beast/http/read.hpp>
#include <boost/beast/http/serializer_fwd.hpp>
#include <boost/beast/http/status.hpp>
#include <boost/beast/http/string_body_fwd.hpp>
#include <boost/beast/http/verb.hpp>
#include <boost/beast/version.hpp>
#include <boost/config.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/json.hpp>
#include <boost/json/object.hpp>
#include <boost/json/parse.hpp>
#include <boost/json/serialize.hpp>
#include <boost/json/value.hpp>
#include <boost/system/detail/error_code.hpp>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <exception>
#include <fstream>
#include <ios>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "handlers.hpp"
#include "server.hpp"

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;
namespace json = boost::json;

static std::string read_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return {};
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

Listener::Listener(net::io_context &ioc, tcp::endpoint endpoint)
    : ioc_(ioc), acceptor_(net::make_strand(ioc)) {
  beast::error_code ec;
  acceptor_.open(endpoint.protocol(), ec);
  if (ec) {
    std::cout << "Open error: " << ec.message() << std::endl;
    return;
  }
  acceptor_.set_option(net::socket_base::reuse_address(true), ec);
  if (ec) {
    std::cout << "set option error: " << ec.message() << std::endl;
    return;
  }
  acceptor_.bind(endpoint, ec);
  if (ec) {
    std::cout << "Unable to bind server: " << ec.message() << std::endl;
    return;
  }
  acceptor_.listen(net::socket_base::max_listen_connections, ec);
  if (ec) {
    std::cout << "Listen error: " << ec.message() << std::endl;
    return;
  }

  signals_ = std::make_shared<net::signal_set>(ioc, SIGINT, SIGTERM);
  signals_->async_wait([this](const boost::system::error_code &ec,
                              int signal_number) {
    if (!ec) {
      std::cout << "[Server] Interrupt Received: " << signal_number
                << "\nInitiating graceful shutdown....\n";
      stop();
    } else {
      std::cout << "Error handling interrupts: " << ec.message() << std::endl;
    }
  });
}

void Listener::do_accept() {
  acceptor_.async_accept(net::make_strand(ioc_),
                         [this](beast::error_code ec, tcp::socket socket) {
                           if (ec) {
                             return;
                           }
                           std::cout << "Request accepted.." << std::endl;
                           std::make_shared<Session>(std::move(socket))->run();
                           do_accept();
                         });
}

void Listener::run() { do_accept(); }

void Listener::stop() {
  boost::system::error_code ec;
  signals_->cancel(ec);
  acceptor_.close(ec);
  ioc_.stop();
}

void Session::run() { do_read_(); }

void Session::do_read_() {
  auto self(shared_from_this());
  req_ = {};
  stream_.expires_after(std::chrono::seconds(30));
  http::async_read(
      stream_, buffer_, req_,
      [this, self](beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);
        if (ec == http::error::end_of_stream) {
          self->do_close_();
          return;
        }
        if (ec) {
          std::cerr << "Read Error: " << ec.message() << std::endl;
          return;
        }
        self->handle_request_();
      });
}

void Session::handle_request_() {
  if (req_.method() == http::verb::get &&
      (req_.target() == "/" || req_.target() == "/index.html")) {
    std::string html = read_file("tools/serve/web/index.html");
    if (html.empty()) {
      send_html_response_(http::status::not_found,
                          "<html><body><h1>404 Not Found</h1></body></html>");
    } else {
      send_html_response_(http::status::ok, html);
    }
    return;
  }

  std::cout << "Path: " << req_.target() << std::endl;
  if (req_.method() == http::verb::get && req_.target() == "/health") {
    health_check(*this);
  }
  if (req_.method() == http::verb::post) {
    auto content_type = req_[http::field::content_type];
    if (content_type.find("application/json") == content_type.npos) {
      send_json_response(http::status::unsupported_media_type,
                         {"error", "Expected content type: application/json"});
      return;
    }
    boost::system::error_code ec;
    json::value body = json::parse(req_.body(), ec);
    if (ec) {
      send_json_response(http::status::bad_request,
                         {"error", "Parse error: " + ec.message()});
      return;
    }
    if (req_.target() == "/v1/chat/completions") {
      chat_completion(*this, body.as_object());
    }
  }
}

void Session::send_json_response(http::status status, json::value payload) {
  auto res = std::make_shared<http::response<http::string_body>>(
      http::status::ok, req_.version());
  res->set(http::field::server, "Boost.Beast Server");
  res->set(http::field::content_type, "application/json");
  res->keep_alive(req_.keep_alive());
  res->body() = json::serialize(payload);
  res->prepare_payload();
  send_response_(res);
}

void Session::initiate_streaming() {
  auto res = std::make_shared<http::response<http::empty_body>>(
      http::status::ok, req_.version());
  res->set(http::field::server, "zinferlm-server");
  res->set(http::field::content_type, "text/event-stream");
  res->set(http::field::cache_control, "no-cache");
  res->set(http::field::connection, "keep-alive");
  res->chunked(true);

  auto sr = std::make_shared<http::response_serializer<http::empty_body>>(*res);
  stream_.expires_after(std::chrono::seconds(60));
  http::async_write_header(
      stream_, *sr,
      [self = shared_from_this(), res, sr](beast::error_code ec, std::size_t) {
        if (ec) {
          std::cerr << "Failed to write to server stream: " << ec.message()
                    << std::endl;
          self->do_close_();
          return;
        }
      });
}

void Session::send_html_response_(http::status status, std::string body) {
  auto res = std::make_shared<http::response<http::string_body>>(
      status, req_.version());
  res->set(http::field::server, "Boost.Beast Server");
  res->set(http::field::content_type, "text/html");
  res->keep_alive(req_.keep_alive());
  res->body() = std::move(body);
  res->prepare_payload();
  send_response_(res);
}

void Session::send_response_(
    std::shared_ptr<http::response<http::string_body>> res) {
  stream_.expires_after(std::chrono::seconds(30));

  http::async_write(
      stream_, *res,
      [self = shared_from_this(), res](beast::error_code ec, std::size_t) {
        if (ec) {
          std::cout << "Write Error: " << ec.message() << std::endl;
          return;
        }
        if (res->keep_alive()) {
          self->do_read_();
        } else {
          self->do_close_();
        }
      });
}

std::string Session::make_chunk_(std::string raw_string) {
  std::ostringstream str_stream;
  str_stream << std::hex << raw_string.size() << "\r\n" << raw_string << "\r\n";
  return str_stream.str();
}

void Session::stream_json_message(json::value payload, bool is_end) {
  std::string raw_string = "data: " + json::serialize(payload) + "\n\n";
  if (is_end) {
    raw_string += "data: [DONE]\n\n";
  }
  std::string chunked_bytes = make_chunk_(raw_string);
  if (is_end) {
    chunked_bytes += "0\r\n\r\n";
  }
  streaming_queue_.push(chunked_bytes);
  if (!is_writing_) {
    flush_streaming_queue_();
  }
}

void Session::flush_streaming_queue_() {
  if (streaming_queue_.empty()) {
    is_writing_ = false;
    return;
  }
  is_writing_ = true;
  std::string current_chunk = streaming_queue_.front();
  streaming_queue_.pop();
  stream_.expires_after(std::chrono::seconds(30));

  net::async_write(stream_, net::buffer(current_chunk),
                   [self = shared_from_this(),
                    current_chunk](beast::error_code ec, std::size_t) {
                     if (ec) {
                       std::cerr
                           << "Client disconnected midstream: " << ec.message()
                           << std::endl;
                       self->is_writing_ = false;
                       self->do_close_();
                       return;
                     }
                     if (current_chunk.find("0\r\n\r\n") != std::string::npos) {
                       self->is_writing_ = false;
                       if (self->req_.keep_alive()) {
                         self->do_read_();
                       } else {
                         self->do_close_();
                       }
                       return;
                     }
                     self->flush_streaming_queue_();
                   });
}

void Session::do_close_() {
  beast::error_code ec;
  stream_.socket().shutdown(tcp::socket::shutdown_send, ec);
}

void Server::run() {
  try {
    auto const address = net::ip::make_address(ip_address_);
    net::io_context ioc{num_threads_};
    listener_ = std::make_shared<Listener>(ioc, tcp::endpoint{address, port_});
    listener_->run();
    std::vector<std::thread> threads;
    threads.reserve(num_threads_ - 1);
    for (int i = 0; i < num_threads_ - 1; i++) {
      threads.emplace_back([&ioc] { ioc.run(); });
    }
    std::cout << "HTTP Server Listening on " << address.to_string() << ":"
              << port_ << std::endl;
    ioc.run();
    for (auto &t : threads) {
      t.join();
    }
    std::cout << "Server Stopped with graceful shutdown..." << std::endl;
    listener_.reset();
  } catch (const std::exception &e) {
    std::cerr << "Fatal: " << e.what() << std::endl;
  }
}
