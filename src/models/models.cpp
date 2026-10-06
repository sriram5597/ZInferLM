#include <cassert>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

#include "engine/graph.h"
#include "ggml-cpp.h"
#include "kv_cache/cache.h"
#include "model_loader/gguf/gguf.h"
#include "qwen/model.h"
#include "sampling/samplers.h"
#include <ggml-cpu.h>
#include <ggml.h>
#include <zinferlm/events.h>
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

using EventDispatcher = zinferlm::events::EventDispatcher;
using ModelEvent = zinferlm::events::ModelEvent;

static_assert(sizeof(struct ggml_tensor) > 0, "ggml integration check");

std::unique_ptr<zinferlm::Model> zinferlm::Model::instance_;

zinferlm::Model &zinferlm::Model::instance() { return *instance_; }

bool zinferlm::Model::load(const char *model_path) {
  int raw_fd = open(model_path, O_RDONLY);
  if (raw_fd == -1) {
    std::cerr << "Unable to read model file: " << model_path << std::endl;
    return false;
  }
  std::unique_ptr<int, void (*)(int *)> fd_guard(&raw_fd,
                                                 [](int *fd) { close(*fd); });
  struct stat model_stats;
  if (fstat(raw_fd, &model_stats) == -1) {
    std::cerr << "Unable to get file descriptor" << std::endl;
    return false;
  }
  size_t file_size = model_stats.st_size;
  if (is_gguf_file(&raw_fd)) {
    std::unique_ptr<ModelLoader> f = load_gguf_file(&raw_fd, file_size);
    instance_ = std::make_unique<QwenModel>(std::move(f));
    return true;
  }
  return false;
}

std::vector<float> zinferlm::Model::predict(std::vector<int32_t> tokens,
                                            int past_tokens) {
  zinferlm::model_config_t cfg = config();
  ggml_context_ptr ctx = init_engine(cfg.n_blocks * 64 + 256);
  Graph graph(ctx.get());

  if (!cache_) {
    cache_ = std::make_unique<KVCache>(
        ggml_backend_get_default_buffer_type(graph.get_backend()), cfg.max_context_len,
        cfg.n_blocks, cfg.nkv, cfg.embedding_dim / cfg.nheads, GGML_TYPE_F16);
  }

  if (past_tokens == 0) {
    cache_->reset();
  }
  std::vector<std::unique_ptr<Layer>> layers =
      create_layers(ctx.get(), past_tokens, tokens.size(), cache_.get());
  std::vector<Layer *> layer_ptrs;
  for (const auto &l : layers)
    layer_ptrs.push_back(l.get());
  graph.set_layers(layer_ptrs);
  graph.set_debug_mode(debug_);
  ggml_tensor *output = graph.execute(tokens);
  GGML_ASSERT(tokens.size() <= output->ne[1]);
  std::vector<float> logits(output->ne[0]);
  uint64_t offset =
      (tokens.size() - 1) * output->ne[0] * ggml_type_size(output->type);
  ggml_backend_tensor_get(output, logits.data(), offset,
                          output->ne[0] * ggml_type_size(output->type));

  return logits;
}

std::string zinferlm::Model::invoke(std::string input, int max_tokens) {
  EventDispatcher &dispatcher = EventDispatcher::get_instance();
  zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(*this);
  std::string output = "";
  dispatcher.dispatch(ModelEvent::TOKENIZER_STARTED,
                      events::tokenizer_started_event_t{
                          .str_len = input.length(),
                      });
  std::vector<int32_t> tokens = tokenizer.tokenize(input);
  dispatcher.dispatch(ModelEvent::TOKENIZER_COMPLETED,
                      events::tokenizer_completed_event_t{
                          .num_tokens = tokens.size(),
                      });
  int past_tokens = 0;
  int i = 0;
  for (; i < max_tokens; i++) {
    if (past_tokens == 0) {
      events::prefill_start_event_t p_start;
      p_start.tokens_count = tokens.size();
      dispatcher.dispatch(ModelEvent::PREFILL_STARTED, p_start);
    }
    std::vector<float> logits = this->predict(tokens, past_tokens);
    if (past_tokens == 0) {
      events::prefill_end_event_t p_end;
      p_end.tokens_count = tokens.size();
      dispatcher.dispatch(ModelEvent::PREFILL_COMPLETED, p_end);
    }

    sampler_params_t params = {.temperature = 0.2f, .top_k = 0};
    Sampler sampler(params);
    std::pair<uint64_t, float> sample = sampler.sample(logits);
    past_tokens += tokens.size();
    tokens.clear();
    tokens.push_back(sample.first);
    std::string out_token =
        tokenizer.decode(static_cast<uint32_t>(sample.first));
    output += out_token;
    if (tokenizer.is_stop_token(out_token)) {
      dispatcher.dispatch(ModelEvent::GENERATION_COMPLETED,
                          zinferlm::events::generation_completed_event_t{
                              .status = StreamStatus::EOS,
                              .num_tokens = static_cast<uint32_t>(i)});

      break;
    }

    dispatcher.dispatch(ModelEvent::TOKEN_GENERATED,
                        zinferlm::events::token_generated_event_t{
                            .seq_id = static_cast<uint32_t>(i),
                            .token = std::string_view{out_token}});
  }
  if (i >= max_tokens) {
    dispatcher.dispatch(ModelEvent::GENERATION_COMPLETED,
                        zinferlm::events::generation_completed_event_t{
                            .status = StreamStatus::MAX_CTX_REACHED});
  }
  return output;
}
