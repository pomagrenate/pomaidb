#pragma once

#include "cheese-kv-cache.h"

#include <vector>

//
// cheese_kv_cache_iswa
//

// utilizes two instances of cheese_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

class cheese_kv_cache_iswa : public cheese_memory_i {
public:
    cheese_kv_cache_iswa(
            const cheese_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~cheese_kv_cache_iswa() = default;

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
    void state_read (cheese_io_read_i  & io, cheese_seq_id seq_id = -1, cheese_state_seq_flags flags = 0) override;

    //
    // cheese_kv_cache_iswa specific API
    //

    cheese_kv_cache * get_base() const;
    cheese_kv_cache * get_swa () const;

private:
    const cheese_hparams & hparams;

    const bool unified;

    std::unique_ptr<cheese_kv_cache> kv_base;
    std::unique_ptr<cheese_kv_cache> kv_swa;
};

class cheese_kv_cache_iswa_context : public cheese_memory_context_i {
public:
    using slot_info_vec_t = cheese_kv_cache::slot_info_vec_t;

    // used for errors
    cheese_kv_cache_iswa_context(cheese_memory_status status);

    // used to create a full-cache context
    cheese_kv_cache_iswa_context(
            cheese_kv_cache_iswa * kv);

    // used to create an update context
    cheese_kv_cache_iswa_context(
            cheese_kv_cache_iswa * kv,
            cheese_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    cheese_kv_cache_iswa_context(
            cheese_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_swa,
            std::vector<cheese_ubatch> ubatches);

    virtual ~cheese_kv_cache_iswa_context();

    //
    // cheese_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    cheese_memory_status  get_status() const override;
    const cheese_ubatch & get_ubatch() const override;

    //
    // cheese_kv_cache_iswa_context specific API
    //

    const cheese_kv_cache_context * get_base() const;
    const cheese_kv_cache_context * get_swa()  const;

private:
    //cheese_kv_cache_iswa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<cheese_ubatch> ubatches;

    const cheese_memory_context_ptr ctx_base;
    const cheese_memory_context_ptr ctx_swa;

    const cheese_memory_status status;
};
