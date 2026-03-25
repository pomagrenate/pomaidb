#pragma once

#include "cheese.h"

#include <string>
#include <vector>
#include <memory>

// pre-tokenization types
enum cheese_vocab_pre_type {
    CHEESE_VOCAB_PRE_TYPE_DEFAULT         = 0,
    CHEESE_VOCAB_PRE_TYPE_LLAMA3          = 1,
    CHEESE_VOCAB_PRE_TYPE_DEEPSEEK_LLM    = 2,
    CHEESE_VOCAB_PRE_TYPE_DEEPSEEK_CODER  = 3,
    CHEESE_VOCAB_PRE_TYPE_FALCON          = 4,
    CHEESE_VOCAB_PRE_TYPE_MPT             = 5,
    CHEESE_VOCAB_PRE_TYPE_STARCODER       = 6,
    CHEESE_VOCAB_PRE_TYPE_GPT2            = 7,
    CHEESE_VOCAB_PRE_TYPE_REFACT          = 8,
    CHEESE_VOCAB_PRE_TYPE_COMMAND_R       = 9,
    CHEESE_VOCAB_PRE_TYPE_STABLELM2       = 10,
    CHEESE_VOCAB_PRE_TYPE_QWEN2           = 11,
    CHEESE_VOCAB_PRE_TYPE_OLMO            = 12,
    CHEESE_VOCAB_PRE_TYPE_DBRX            = 13,
    CHEESE_VOCAB_PRE_TYPE_SMAUG           = 14,
    CHEESE_VOCAB_PRE_TYPE_PORO            = 15,
    CHEESE_VOCAB_PRE_TYPE_CHATGLM3        = 16,
    CHEESE_VOCAB_PRE_TYPE_CHATGLM4        = 17,
    CHEESE_VOCAB_PRE_TYPE_VIKING          = 18,
    CHEESE_VOCAB_PRE_TYPE_JAIS            = 19,
    CHEESE_VOCAB_PRE_TYPE_TEKKEN          = 20,
    CHEESE_VOCAB_PRE_TYPE_SMOLLM          = 21,
    CHEESE_VOCAB_PRE_TYPE_CODESHELL       = 22,
    CHEESE_VOCAB_PRE_TYPE_BLOOM           = 23,
    CHEESE_VOCAB_PRE_TYPE_GPT3_FINNISH    = 24,
    CHEESE_VOCAB_PRE_TYPE_EXAONE          = 25,
    CHEESE_VOCAB_PRE_TYPE_CHAMELEON       = 26,
    CHEESE_VOCAB_PRE_TYPE_MINERVA         = 27,
    CHEESE_VOCAB_PRE_TYPE_DEEPSEEK3_LLM   = 28,
    CHEESE_VOCAB_PRE_TYPE_GPT4O           = 29,
    CHEESE_VOCAB_PRE_TYPE_SUPERBPE        = 30,
    CHEESE_VOCAB_PRE_TYPE_TRILLION        = 31,
    CHEESE_VOCAB_PRE_TYPE_BAILINGMOE      = 32,
    CHEESE_VOCAB_PRE_TYPE_LLAMA4          = 33,
    CHEESE_VOCAB_PRE_TYPE_PIXTRAL         = 34,
    CHEESE_VOCAB_PRE_TYPE_SEED_CODER      = 35,
    CHEESE_VOCAB_PRE_TYPE_HUNYUAN         = 36,
    CHEESE_VOCAB_PRE_TYPE_KIMI_K2         = 37,
    CHEESE_VOCAB_PRE_TYPE_HUNYUAN_DENSE   = 38,
    CHEESE_VOCAB_PRE_TYPE_GROK_2          = 39,
    CHEESE_VOCAB_PRE_TYPE_GRANITE_DOCLING = 40,
    CHEESE_VOCAB_PRE_TYPE_MINIMAX_M2      = 41,
    CHEESE_VOCAB_PRE_TYPE_AFMOE           = 42,
    CHEESE_VOCAB_PRE_TYPE_SOLAR_OPEN      = 43,
    CHEESE_VOCAB_PRE_TYPE_YOUTU           = 44,
    CHEESE_VOCAB_PRE_TYPE_EXAONE_MOE      = 45,
    CHEESE_VOCAB_PRE_TYPE_QWEN35          = 46,
    CHEESE_VOCAB_PRE_TYPE_TINY_AYA        = 47,
    CHEESE_VOCAB_PRE_TYPE_JOYAI_LLM       = 48,
    CHEESE_VOCAB_PRE_TYPE_JAIS2           = 49,
};

struct LLM_KV;
struct cheese_model_loader;

struct cheese_vocab {
    struct token_data {
        std::string      text;
        float            score;
        cheese_token_attr attr;
    };

    cheese_vocab();
    ~cheese_vocab();

    void load(cheese_model_loader & ml, const LLM_KV & kv);

    std::string get_tokenizer_model() const;
    std::string get_tokenizer_pre() const;

    enum cheese_vocab_type     get_type()     const;
    enum cheese_vocab_pre_type get_pre_type() const;

    uint32_t n_tokens() const;
    uint32_t n_token_types() const;

    std::string type_name() const;

    bool is_normal      (cheese_token id) const;
    bool is_unknown     (cheese_token id) const;
    bool is_control     (cheese_token id) const;
    bool is_byte        (cheese_token id) const;
    bool is_user_defined(cheese_token id) const;
    bool is_unused      (cheese_token id) const;
    bool is_eog         (cheese_token id) const;

    uint8_t     token_to_byte(cheese_token id) const;
    cheese_token byte_to_token(uint8_t ch)     const;

    cheese_token text_to_token(const std::string & text) const;

    const token_data & get_token_data(cheese_token id) const;

    const char *     token_get_text (cheese_token id) const;
    float            token_get_score(cheese_token id) const;
    cheese_token_attr token_get_attr (cheese_token id) const;

    cheese_token token_bos() const;
    cheese_token token_eos() const;
    cheese_token token_eot() const;
    cheese_token token_eom() const;
    cheese_token token_unk() const;
    cheese_token token_sep() const;
    cheese_token token_nl () const;
    cheese_token token_pad() const;
    cheese_token token_mask() const;

    cheese_token token_prefix() const;
    cheese_token token_middle() const;
    cheese_token token_suffix() const;

    cheese_token token_fim_pre() const;
    cheese_token token_fim_suf() const;
    cheese_token token_fim_mid() const;
    cheese_token token_fim_pad() const;
    cheese_token token_fim_rep() const;
    cheese_token token_fim_sep() const;

    bool get_add_space_prefix          () const;
    bool get_add_bos                   () const;
    bool get_add_eos                   () const;
    bool get_add_sep                   () const;
    bool get_ignore_merges             () const;
    bool get_clean_spaces              () const;
    bool get_remove_extra_whitespaces  () const;
    bool get_escape_whitespaces        () const;
    bool get_treat_whitespace_as_suffix() const;

    int max_token_len() const;

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;
    std::vector<std::string> get_bpe_merges() const;

    std::vector<char> get_precompiled_charsmap() const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  cheese_token * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

    std::vector<cheese_token> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    // does not write null-terminator to buf
    int32_t token_to_piece(
                  cheese_token   token,
                         char * buf,
                      int32_t   length,
                      int32_t   lstrip,
                         bool   special) const;

    // use cached data
    const std::string & token_to_piece(cheese_token token) const;

    int32_t detokenize(
            const cheese_token * tokens,
                      int32_t   n_tokens,
                         char * text,
                      int32_t   text_len_max,
                         bool   remove_special,
                         bool   unparse_special) const;

    std::string detokenize(
            const std::vector<cheese_token> & tokens,
                                      bool   special) const;

    void print_info() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
