#pragma once

#include "cheese.h"

#include <vector>

struct cheese_vocab;
struct cheese_grammar;

// sampler chain

struct cheese_sampler_chain {
    cheese_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    struct info {
        bool is_backend;

        cheese_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for cheese_sampler_sample to avoid repeated allocations
    std::vector<cheese_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

struct cheese_sampler * cheese_sampler_init_dry_testing(
        int32_t context_size,
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<cheese_token>> & seq_breakers);
