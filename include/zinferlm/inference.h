#pragma once

#include<string>
#include<vector>
#include<cstdint>

namespace zinferlm {
  class Inference {
    public:
      std::string invoke(std::string input);
      std::vector<int32_t> tokenize(std::string input);
  };
}
