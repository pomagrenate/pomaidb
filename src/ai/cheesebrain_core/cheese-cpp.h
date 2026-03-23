#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>

#include "cheese.h"

struct cheese_model_deleter {
    void operator()(cheese_model * model) { cheese_model_free(model); }
};

struct cheese_context_deleter {
    void operator()(cheese_context * context) { cheese_free(context); }
};

struct cheese_sampler_deleter {
    void operator()(cheese_sampler * sampler) { cheese_sampler_free(sampler); }
};

struct cheese_adapter_lora_deleter {
    void operator()(cheese_adapter_lora *) {
        // cheese_adapter_lora_free is deprecated
    }
};

typedef std::unique_ptr<cheese_model, cheese_model_deleter> cheese_model_ptr;
typedef std::unique_ptr<cheese_context, cheese_context_deleter> cheese_context_ptr;
typedef std::unique_ptr<cheese_sampler, cheese_sampler_deleter> cheese_sampler_ptr;
typedef std::unique_ptr<cheese_adapter_lora, cheese_adapter_lora_deleter> cheese_adapter_lora_ptr;
