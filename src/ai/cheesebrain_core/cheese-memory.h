#pragma once

#include "cheese.h"

#include <map>
#include <memory>
#include <functional>

struct cheese_ubatch;

class cheese_batch_allocr;

class cheese_io_write_i;
class cheese_io_read_i;

struct cheese_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;
};

enum cheese_memory_status {
    CHEESE_MEMORY_STATUS_SUCCESS = 0,
    CHEESE_MEMORY_STATUS_NO_UPDATE,
    CHEESE_MEMORY_STATUS_FAILED_PREPARE,
    CHEESE_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
cheese_memory_status cheese_memory_status_combine(cheese_memory_status s0, cheese_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool cheese_memory_status_is_fail(cheese_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - cheese_kv_cache_context
//   - cheese_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is cheese_memory_i::apply()
struct cheese_memory_context_i {
    virtual ~cheese_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const cheese_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual cheese_memory_status get_status() const = 0;
};

using cheese_memory_context_ptr = std::unique_ptr<cheese_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct cheese_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    virtual ~cheese_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the cheese_memory_context_i::get_status() for the result
    virtual cheese_memory_context_ptr init_batch(
            cheese_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual cheese_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == CHEESE_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual cheese_memory_context_ptr init_update(cheese_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (cheese_seq_id seq_id,                              cheese_pos p0, cheese_pos p1) = 0;
    virtual void seq_cp  (cheese_seq_id seq_id_src, cheese_seq_id seq_id_dst, cheese_pos p0, cheese_pos p1) = 0;
    virtual void seq_keep(cheese_seq_id seq_id) = 0;
    virtual void seq_add (cheese_seq_id seq_id,                              cheese_pos p0, cheese_pos p1, cheese_pos shift) = 0;
    virtual void seq_div (cheese_seq_id seq_id,                              cheese_pos p0, cheese_pos p1, int d) = 0;

    virtual cheese_pos seq_pos_min(cheese_seq_id seq_id) const = 0;
    virtual cheese_pos seq_pos_max(cheese_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // state write/read
    //

    virtual void state_write(cheese_io_write_i & io, cheese_seq_id seq_id = -1, cheese_state_seq_flags flags = 0) const = 0;
    virtual void state_read (cheese_io_read_i  & io, cheese_seq_id seq_id = -1, cheese_state_seq_flags flags = 0) = 0;
};

using cheese_memory_ptr = std::unique_ptr<cheese_memory_i>;
