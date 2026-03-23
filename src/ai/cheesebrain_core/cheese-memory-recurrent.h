#pragma once

#include "cheese-batch.h"
#include "cheese-graph.h"
#include "cheese-memory.h"

#include <map>
#include <set>
#include <vector>

//
// cheese_memory_recurrent
//

// TODO: extract the cache state used for graph computation into cheese_memory_recurrent_context_i
//       see the implementation of cheese_kv_cache_context_i for an example how to do it
class cheese_memory_recurrent : public cheese_memory_i {
public:
    cheese_memory_recurrent(
            const cheese_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
        const layer_filter_cb & filter);

    ~cheese_memory_recurrent() = default;

    //
    // cheese_memory_i
    //

    cheese_memory_context_ptr init_batch(
            cheese_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    cheese_memory_context_ptr init_full() override;

    cheese_memory_context_ptr init_update(cheese_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (cheese_seq_id seq_id,                              cheese_pos p0, cheese_pos p1) override;
    void seq_cp  (cheese_seq_id seq_id_src, cheese_seq_id seq_id_dst, cheese_pos p0, cheese_pos p1) override;
    void seq_keep(cheese_seq_id seq_id)                                                          override;
    void seq_add (cheese_seq_id seq_id,                              cheese_pos p0, cheese_pos p1, cheese_pos shift) override;
    void seq_div (cheese_seq_id seq_id,                              cheese_pos p0, cheese_pos p1, int d) override;

    cheese_pos seq_pos_min(cheese_seq_id seq_id) const override;
    cheese_pos seq_pos_max(cheese_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<cheese_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const cheese_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(cheese_io_write_i & io, cheese_seq_id seq_id = -1, cheese_state_seq_flags flags = 0) const override;
    void state_read (cheese_io_read_i  & io, cheese_seq_id seq_id = -1, cheese_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        cheese_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<cheese_seq_id> seq_id;

        bool has_seq_id(const cheese_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;

private:
    //const cheese_model & model;
    const cheese_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;

    void state_write_meta(cheese_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, cheese_seq_id seq_id = -1) const;
    void state_write_data(cheese_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(cheese_io_read_i & io, uint32_t cell_count, cheese_seq_id dest_seq_id = -1);
    bool state_read_data(cheese_io_read_i & io, uint32_t cell_count);
};

class cheese_memory_recurrent_context : public cheese_memory_context_i {
public:
    // used for errors
    cheese_memory_recurrent_context(cheese_memory_status status);

    // used to create a full-cache or update context
    cheese_memory_recurrent_context(
            cheese_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    cheese_memory_recurrent_context(
            cheese_memory_recurrent * mem,
            std::vector<cheese_ubatch> ubatches);

    virtual ~cheese_memory_recurrent_context();

    //
    // cheese_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    cheese_memory_status  get_status() const override;
    const cheese_ubatch & get_ubatch() const override;

    //
    // cheese_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;

    int32_t s_copy(int i) const;

private:
    const cheese_memory_status status;

    cheese_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<cheese_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
