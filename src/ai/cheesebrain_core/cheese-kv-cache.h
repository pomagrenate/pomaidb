#pragma once

#include "cheese-batch.h"
#include "cheese-graph.h"
#include "cheese-kv-cells.h"
#include "cheese-memory.h"

#include <unordered_map>
#include <vector>

struct cheese_cparams;
struct cheese_hparams;
struct cheese_model;
struct cheese_context;

//
// cheese_kv_cache
//

class cheese_kv_cache : public cheese_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<cheese_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    cheese_kv_cache(
            const cheese_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               cheese_swa_type   swa_type,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~cheese_kv_cache() = default;

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
    // cheese_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<cheese_ubatch> & ubatches);

    bool update(cheese_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const cheese_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const cheese_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const cheese_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const cheese_ubatch & ubatch) const;

    void set_input_k_idxs(ggml_tensor * dst, const cheese_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const cheese_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const cheese_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const cheese_ubatch * ubatch) const;

private:
    const cheese_model & model;
    const cheese_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: CHEESE_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const cheese_swa_type swa_type = CHEESE_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    std::vector<cheese_kv_cells> v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const cheese_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  cheese_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(cheese_io_write_i & io, const cell_ranges_t & cr, cheese_seq_id seq_id = -1) const;
    void state_write_data(cheese_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(cheese_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, cheese_seq_id dest_seq_id = -1);
    bool state_read_data(cheese_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class cheese_kv_cache_context : public cheese_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = cheese_kv_cache::slot_info_vec_t;
    using stream_copy_info = cheese_kv_cache::stream_copy_info;

    // used for errors
    cheese_kv_cache_context(cheese_memory_status status);

    // used to create a full-cache context
    cheese_kv_cache_context(
            cheese_kv_cache * kv);

    // used to create an update context
    cheese_kv_cache_context(
            cheese_kv_cache * kv,
            cheese_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    cheese_kv_cache_context(
            cheese_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<cheese_ubatch> ubatches);

    virtual ~cheese_kv_cache_context();

    //
    // cheese_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    cheese_memory_status  get_status() const override;
    const cheese_ubatch & get_ubatch() const override;

    //
    // cheese_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be layed out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const cheese_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const cheese_ubatch & ubatch) const;

    void set_input_k_idxs(ggml_tensor * dst, const cheese_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const cheese_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const cheese_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const cheese_ubatch * ubatch) const;

private:
    cheese_memory_status status;

    cheese_kv_cache * kv;
    cheese_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<cheese_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};
