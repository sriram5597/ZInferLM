#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/address.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/signal_set.hpp>
#include <boost/asio/socket_base.hpp>
#include <boost/asio/strand.hpp>
#include <boost/beast/core/error.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/http/message_fwd.hpp>
#include <boost/beast/http/status.hpp>
#include <boost/beast/version.hpp>
#include <boost/config.hpp>
#include <boost/json.hpp>
#include <cstdint>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <thread>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;
namespace json = boost::json;

class Session : public std::enable_shared_from_this<Session> {
private:
  beast::tcp_stream stream_;
  bool is_writing_;
  http::request<http::string_body> req_;
  beast::flat_buffer buffer_;
  std::queue<std::string> streaming_queue_;
  void do_read_();
  void do_close_();
  void handle_request_();
  void send_response_(std::shared_ptr<http::response<http::string_body>> res);
  void send_html_response_(http::status status, std::string body);
  std::string make_chunk_(std::string raw_string);
  void flush_streaming_queue_();

public:
  explicit Session(tcp::socket &&socket) : stream_(std::move(socket)) {}
  void send_json_response(http::status status, json::value payload);
  void initiate_streaming();
  void stream_json_message(json::value payload, bool is_end);
  void run();
};

class Listener : public std::enable_shared_from_this<Listener> {
private:
  net::io_context &ioc_;
  tcp::acceptor acceptor_;
  std::shared_ptr<net::signal_set> signals_;

  void do_accept();

public:
  Listener(net::io_context &ioc, tcp::endpoint endpoint);
  void run();
  void stop();
};

class Server {
private:
  int num_threads_;
  std::string ip_address_;
  unsigned short port_;
  std::shared_ptr<Listener> listener_;

public:
  Server(std::string ip, unsigned short port, int num_threads)
      : ip_address_(ip), port_(port), num_threads_(num_threads) {}
  void run();
};
