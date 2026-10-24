#include "server.hpp"

#include <cstdlib>
#include <iostream>
#include <iterator>
#include <string>
#include <zinferlm/models.h>

int main(int argc, char *argv[]) {
  std::string ip = "0.0.0.0";
  unsigned short port = 8080;
  int num_threads = 4;
  std::string model_path = "";

  if (argc > 1)
    model_path = argv[1];

  std::cout << "Loading model from file: " << model_path << std::endl;
  auto is_loaded = zinferlm::Model::load(model_path.c_str());
  auto &model = zinferlm::Model::instance();
  zinferlm::model_info_t info = model.info();
  std::cout << "Model Loaded: " << info.name << std::endl;

  if (num_threads < 1)
    num_threads = 1;

  Server server(ip, port, num_threads);
  server.run();

  zinferlm::Model::unload();
  return 0;
}
