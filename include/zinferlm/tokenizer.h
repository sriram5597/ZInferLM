#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include "models.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

namespace zinferlm
{
    class Tokenizer
    {
    private:
        using Pcre2CodePtr = std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)>;
        using Pcre2DataPtr = std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)>;

        std::string model_;
        std::string pre_;
        std::unordered_map<std::string, uint64_t> merges_map_;
        Pcre2CodePtr re_{nullptr, pcre2_code_free};
        Pcre2DataPtr match_data_{nullptr, pcre2_match_data_free};
        std::unordered_map<uint8_t, std::string> byte_to_unicode_map_;
        std::vector<std::string> tokens_;
        std::unordered_map<std::string, uint64_t> token_to_id_map_;

        Tokenizer(std::string m, std::string p, std::vector<std::string> t) : model_(m), pre_(p), tokens_(t) {}
        void init_pretokenize_();
        void build_byte_to_unicode_map_();
        void build_token_id_map_();
        void build_merges_map_(std::vector<std::string> merges);
        std::vector<std::string> split_and_merge_(std::string chunk);

    public:
        static Tokenizer for_model(zinferlm::Model &model);
        void init();
        std::vector<std::string> pretokenize(std::string text);
        std::vector<std::pair<std::string, uint64_t>> tokenize(std::string text);
    };
};
