#include <iostream>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <memory>

#include <zinferlm/model_loader.h>
#include "gguf/gguf.h"
#include "models/gguf.h"

std::unique_ptr<zinferlm::Model> zinferlm::Model::instance_;

zinferlm::Model &zinferlm::Model::instance()
{
  return *instance_;
}

bool zinferlm::Model::load(const char *model_path)
{
  int raw_fd = open(model_path, O_RDONLY);
  if (raw_fd == -1)
  {
    std::cerr << "Unable to read model file: " << model_path << std::endl;
    return false;
  }
  std::unique_ptr<int, void (*)(int *)> fd_guard(&raw_fd, [](int *fd)
                                                 { close(*fd); });
  struct stat model_stats;
  if (fstat(raw_fd, &model_stats) == -1)
  {
    std::cerr << "Unable to get file descriptor" << std::endl;
    return false;
  }
  size_t file_size = model_stats.st_size;
  std::cout << "File Size: " << file_size / (1024 * 1024) << " MB " << std::endl;
  if (is_gguf_file(&raw_fd))
  {
    std::cout << "Detected GGUF File" << std::endl;
    std::unique_ptr<GGUFFile> f = load_gguf_file(&raw_fd, file_size);
    instance_ = std::make_unique<GGUFModel>(std::move(f));
    return true;
  }
  return false;
}
