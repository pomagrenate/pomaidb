// C API wrappers for cheese_model (see cheese-model.cpp for class implementation).
#include "cheese-model.h"

#include <cstdio>

const cheese_vocab * cheese_model_get_vocab(const cheese_model * model) {
    return &model->vocab;
}

void cheese_free_model(cheese_model * model) {
    cheese_model_free(model);
}

void cheese_model_free(cheese_model * model) {
    delete model;
}

int32_t cheese_model_n_ctx_train(const cheese_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t cheese_model_n_embd(const cheese_model * model) {
    return model->hparams.n_embd;
}

int32_t cheese_model_n_embd_inp(const cheese_model * model) {
    return model->hparams.n_embd_inp();
}

int32_t cheese_model_n_embd_out(const cheese_model * model) {
    return model->hparams.n_embd_out();
}

int32_t cheese_model_n_layer(const cheese_model * model) {
    return model->hparams.n_layer;
}

int32_t cheese_model_n_head(const cheese_model * model) {
    return model->hparams.n_head();
}

int32_t cheese_model_n_head_kv(const cheese_model * model) {
    return model->hparams.n_head_kv();
}

int32_t cheese_model_n_swa(const cheese_model * model) {
    return model->hparams.n_swa;
}

uint32_t cheese_model_n_cls_out(const struct cheese_model * model) {
    return model->hparams.n_cls_out;
}

const char * cheese_model_cls_label(const struct cheese_model * model, uint32_t i) {
    if (i < model->classifier_labels.size()) {
        return model->classifier_labels[i].c_str();
    }

    return nullptr;
}

// deprecated
int32_t cheese_n_ctx_train(const cheese_model * model) {
    return cheese_model_n_ctx_train(model);
}

// deprecated
int32_t cheese_n_embd(const cheese_model * model) {
    return cheese_model_n_embd(model);
}

// deprecated
int32_t cheese_n_layer(const cheese_model * model) {
    return cheese_model_n_layer(model);
}

// deprecated
int32_t cheese_n_head(const cheese_model * model) {
    return cheese_model_n_head(model);
}

cheese_rope_type cheese_model_rope_type(const cheese_model * model) {
    switch (model->arch) {
        // these models do not use RoPE
        case LLM_ARCH_CLIP:
        case LLM_ARCH_GPT2:
        case LLM_ARCH_GPTJ:
        case LLM_ARCH_MPT:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_BLOOM:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_MAMBA2:
        case LLM_ARCH_JAMBA:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_JAIS:
        case LLM_ARCH_RWKV6:
        case LLM_ARCH_RWKV6QWEN2:
        case LLM_ARCH_RWKV7:
        case LLM_ARCH_ARWKV7:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_NEMOTRON_H:
        case LLM_ARCH_NEMOTRON_H_MOE:
        case LLM_ARCH_KIMI_LINEAR:
            return CHEESE_ROPE_TYPE_NONE;

        // use what we call a normal RoPE, operating on pairs of consecutive head values
        case LLM_ARCH_CHEESE:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_CHEESE4:
        case LLM_ARCH_DECI:
        case LLM_ARCH_BAICHUAN:
        case LLM_ARCH_STARCODER:
        case LLM_ARCH_INTERNLM2:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_XVERSE:
        case LLM_ARCH_COMMAND_R:
        case LLM_ARCH_COHERE2:
        case LLM_ARCH_OLMO:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_PLM:
        case LLM_ARCH_CHATGLM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_GRANITE_HYBRID:
        case LLM_ARCH_CHAMELEON:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_SMOLLM3:
        case LLM_ARCH_ARCEE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_CHEESE_EMBED:
        case LLM_ARCH_MAINCODER:
        case LLM_ARCH_GLM_DSA:
            return CHEESE_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
        case LLM_ARCH_FALCON:
        case LLM_ARCH_FALCON_H1:
        case LLM_ARCH_GROK:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_MODERN_BERT:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_EUROBERT:
        case LLM_ARCH_STABLELM:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
        case LLM_ARCH_OLMO2:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_PHI2:
        case LLM_ARCH_PHI3:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_PLAMO:
        case LLM_ARCH_PLAMO2:
        case LLM_ARCH_PLAMO3:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_GEMMA3:
        case LLM_ARCH_GEMMA3N:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_STARCODER2:
        case LLM_ARCH_OPENELM:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_ORION:
        case LLM_ARCH_NEMOTRON:
        case LLM_ARCH_EXAONE:
        case LLM_ARCH_EXAONE4:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_MINICPM3:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_JAIS2:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_HUNYUAN_DENSE:
        case LLM_ARCH_LFM2:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_SEED_OSS:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_APERTUS:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_COGVLM:
        case LLM_ARCH_PANGU_EMBED:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_STEP35:
            return CHEESE_ROPE_TYPE_NEOX;

        case LLM_ARCH_QWEN2VL:
        case LLM_ARCH_PADDLEOCR:
            return CHEESE_ROPE_TYPE_MROPE;
        case LLM_ARCH_QWEN3VL:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
            return CHEESE_ROPE_TYPE_IMROPE;

        case LLM_ARCH_GLM4:
            return model->hparams.use_mrope() ? CHEESE_ROPE_TYPE_MROPE : CHEESE_ROPE_TYPE_NORM;
        case LLM_ARCH_GLM4_MOE:
            return model->hparams.use_mrope() ? CHEESE_ROPE_TYPE_MROPE : CHEESE_ROPE_TYPE_NEOX;

        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            GGML_ABORT("unknown architecture");
    }

    return CHEESE_ROPE_TYPE_NONE;
}

float cheese_model_rope_freq_scale_train(const cheese_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t cheese_model_meta_val_str(const cheese_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t cheese_model_meta_count(const cheese_model * model) {
    return (int)model->gguf_kv.size();
}

const char * cheese_model_meta_key_str(cheese_model_meta_key key) {
    switch (key) {
        case CHEESE_MODEL_META_KEY_SAMPLING_SEQUENCE:        return "general.sampling.sequence";
        case CHEESE_MODEL_META_KEY_SAMPLING_TOP_K:           return "general.sampling.top_k";
        case CHEESE_MODEL_META_KEY_SAMPLING_TOP_P:           return "general.sampling.top_p";
        case CHEESE_MODEL_META_KEY_SAMPLING_MIN_P:           return "general.sampling.min_p";
        case CHEESE_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY: return "general.sampling.xtc_probability";
        case CHEESE_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD:   return "general.sampling.xtc_threshold";
        case CHEESE_MODEL_META_KEY_SAMPLING_TEMP:            return "general.sampling.temp";
        case CHEESE_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N:  return "general.sampling.penalty_last_n";
        case CHEESE_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT:  return "general.sampling.penalty_repeat";
        case CHEESE_MODEL_META_KEY_SAMPLING_MIROSTAT:        return "general.sampling.mirostat";
        case CHEESE_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU:    return "general.sampling.mirostat_tau";
        case CHEESE_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA:    return "general.sampling.mirostat_eta";
        default:                                            return nullptr;
    }
}

int32_t cheese_model_meta_key_by_index(const cheese_model * model, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t cheese_model_meta_val_str_by_index(const cheese_model * model, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t cheese_model_desc(const cheese_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s", model->desc().c_str());
}

uint64_t cheese_model_size(const cheese_model * model) {
    return model->size();
}

const char * cheese_model_chat_template(const cheese_model * model, const char * name) {
    const auto key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE)
        : LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        // one-off fix for very popular models (so we are not flooded with issues)
        // do not extend this list unless absolutely necessary
        // Mistral-Small-2503 does not have built-in chat template
        cheese_vocab_pre_type pre_type = model->vocab.get_pre_type();
        if (!name && pre_type == CHEESE_VOCAB_PRE_TYPE_TEKKEN && model->layers.size() == 40) {
            return "mistral-v7-tekken";
        }

        return nullptr;
    }

    return it->second.c_str();
}

uint64_t cheese_model_n_params(const cheese_model * model) {
    return model->n_elements();
}

bool cheese_model_has_encoder(const cheese_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5:        return true;
        case LLM_ARCH_T5ENCODER: return true;
        default:                 return false;
    }
}

bool cheese_model_has_decoder(const cheese_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5ENCODER: return false;
        default:                 return true;
    }
}

cheese_token cheese_model_decoder_start_token(const cheese_model * model) {
    return model->hparams.dec_start_token_id;
}

bool cheese_model_is_recurrent(const cheese_model * model) {
    return llm_arch_is_recurrent(model->arch);
}

bool cheese_model_is_hybrid(const cheese_model * model) {
    return llm_arch_is_hybrid(model->arch);
}

bool cheese_model_is_diffusion(const cheese_model * model) {
    return llm_arch_is_diffusion(model->arch);
}

const std::vector<std::pair<std::string, ggml_tensor *>> & cheese_internal_get_tensor_map(const cheese_model * model) {
    return model->tensors_by_name;
}
