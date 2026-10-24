#include <boost/beast.hpp>
#include <boost/json.hpp>
#include <boost/json/object.hpp>

namespace json = boost::json;
namespace http = boost::beast::http;

class Session;

struct handler_response {
  http::status status;
  json::value payload;
};

void health_check(Session &session);
void chat_completion(Session &session, json::object payload);
