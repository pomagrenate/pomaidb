#pragma once

#include "cheese-batch.h"
#include "cheese-graph.h"
#include "cheese-kv-cache-iswa.h"
#include "cheese-memory.h"
#include "cheese-memory-recurrent.h"

#include <memory>
#include <vector>

//
// cheese_memory_hybrid_iswa
//

// utilizes instances of cheese_memory_recurrent and cheese_kv_cache_iswa to
//   support models where each layer may be either attention-based (with SWA support) or recurrent

class cheese_memory_hybrid_iswa : public cheese_memory_i {
public:
    cheese_memory_hybrid_iswa(
        const cheese_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   swa_full,
                 uint32_t   kv_size,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr);

    ~cheese_memory_hybrid_iswa() = default;

    //
    // cheese_memory_i
    //

    cheese_memory_context_ptr init_batch(
            cheese_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    cheese_memory_context_ptr init_full() override;

    cheese_memory_context_ptr init_update(cheese_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (cheese_seq_id seq_id,                              cheese_pos p0, cheese_pos p1) override;
    void seq_cp  (cheese_seq_id seq_id_src, cheese_seq_id seq_id_dst, cheese_pos p0, cheese_pos p1) override;
    void seq_keep(cheese_seq_id seq_id)                                                          override;
    void seq_add (cheese_seq_id seq_id,                              cheese_pos p0, cheese_pos p1, cheese_pos shift) override;
    void seq_div (cheese_seq_id seq_id,                              cheese_pos p0, cheese_pos p1, int d) override;

    cheese_pos seq_pos_min(cheese_seq_id seq_id) const override;
    cheese_pos seq_pos_max(cheese_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(cheese_io_write_i & io, cheese_seq_id seq_id = -1, cheese_state_seq_flags flags = 0) const override;
    void state_read (cheese_io_read_i  & io, cheese_seq_id seq_id = -1, cheese_state_seq_flags flags = 0)       override;

    //
    // cheese_memory_hybrid_iswa specific API
    //

    cheese_kv_cache_iswa * get_mem_attn() const;
    cheese_memory_recurrent * get_mem_recr() const;

private:
    const cheese_hparams & hparams;

    const std::unique_ptr<cheese_kv_cache_iswa> mem_attn;
    const std::unique_ptr<cheese_memory_recurrent> mem_recr;
};

class cheese_memory_hybrid_iswa_context : public cheese_memory_context_i {
public:
    using slot_info_vec_t = cheese_kv_cache::slot_info_vec_t;

    // init failure
    explicit cheese_memory_hybrid_iswa_context(cheese_memory_status status);

    // init full
    explicit cheese_memory_hybrid_iswa_context(cheese_memory_hybrid_iswa * mem);

    // init update
    explicit cheese_memory_hybrid_iswa_context(
        cheese_memory_hybrid_iswa * mem,
                   cheese_context * lctx,
                            bool   optimize);

    // init success
    cheese_memory_hybrid_iswa_context(
           cheese_memory_hybrid_iswa * mem,
                    slot_info_vec_t   sinfos_base,
                    slot_info_vec_t   sinfos_swa,
          std::vector<cheese_ubatch>   ubatches);

    ~cheese_memory_hybrid_iswa_context() = default;

    bool next()  override;
    bool apply() override;

    cheese_memory_status  get_status() const override;
    const cheese_ubatch & get_ubatch() const override;

    //
    // cheese_memory_hybrid_iswa_context
    //

    const cheese_kv_cache_iswa_context * get_attn() const;
    const cheese_memory_recurrent_context * get_recr() const;

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<cheese_ubatch> ubatches;

    const cheese_memory_context_ptr ctx_attn;
    const cheese_memory_context_ptr ctx_recr;

    const cheese_memory_status status;
};
