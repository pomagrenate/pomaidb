#ifndef POMAI_C_TYPES_H
#define POMAI_C_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Cross-platform symbol visibility for the C ABI.
#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(POMAI_C_BUILD_DLL)
#    define POMAI_API __declspec(dllexport)
#  elif defined(POMAI_C_USE_DLL)
#    define POMAI_API __declspec(dllimport)
#  else
#    define POMAI_API
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define POMAI_API __attribute__((visibility("default")))
#else
#  define POMAI_API
#endif

// Opaque handles
typedef struct pomai_db_t pomai_db_t;
typedef struct pomai_snapshot_t pomai_snapshot_t;
typedef struct pomai_iter_t pomai_iter_t;
typedef struct pomai_status_t pomai_status_t;

#define POMAI_QUERY_FLAG_ZERO_COPY 1

typedef enum {
    POMAI_FSYNC_POLICY_NEVER = 0,
    POMAI_FSYNC_POLICY_ALWAYS = 1,
} pomai_fsync_policy_t;

typedef enum {
    POMAI_METRIC_L2 = 0,
    POMAI_METRIC_IP = 1,
    POMAI_METRIC_COSINE = 2,
} pomai_metric_t;

typedef enum {
    POMAI_QUANT_NONE = 0,
    POMAI_QUANT_SQ8 = 1,
    POMAI_QUANT_FP16 = 2,
    POMAI_QUANT_BIT = 3,
    POMAI_QUANT_PQ8 = 4,
} pomai_quant_type_t;

typedef enum {
    POMAI_EMBEDDED_PRESET_GENERIC = 0,
    POMAI_EMBEDDED_PRESET_ESP32_S3 = 1,
    POMAI_EMBEDDED_PRESET_ARM_CORTEX_M85 = 2,
    POMAI_EMBEDDED_PRESET_RPI_ZERO_2W = 3,
} pomai_embedded_preset_t;

typedef struct {
    uint32_t struct_size;
    const char* path;
    uint32_t shards;
    uint32_t dim;
    uint32_t search_threads;
    pomai_fsync_policy_t fsync_policy;
    uint64_t memory_budget_bytes;
    uint32_t deadline_ms;
    
    // Indexing
    uint8_t index_type; // 0 = IVF, 1 = HNSW
    uint32_t hnsw_m;
    uint32_t hnsw_ef_construction;
    uint32_t hnsw_ef_search;
    uint32_t adaptive_threshold;
    uint8_t metric; // 0 = L2, 1 = IP, 2 = Cosine
    uint8_t edge_profile; // 0=user_defined, 1=low_ram, 2=balanced, 3=throughput
    uint32_t tick_max_ops;
    uint32_t tick_max_ms;
    bool strict_deterministic;

    // Advanced & Storage Configuration
    uint8_t quant_type; // pomai_quant_type_t (0=None, 1=SQ8, 2=FP16, 3=BIT, 4=PQ8)
    uint32_t pq_m;
    uint32_t memtable_flush_threshold_mb;
    bool auto_freeze_on_pressure;
    uint32_t max_memtable_mb;
    uint32_t write_coalesce_window_us;
    uint32_t write_coalesce_batch_size;
    bool enable_encryption_at_rest;
    const char* encryption_key_hex;
} pomai_options_t;

typedef struct {
    uint32_t struct_size;
    uint64_t id;
    const float* vector;
    uint32_t dim;
    const uint8_t* metadata;
    uint32_t metadata_len;
    // Multi-membrane & temporal extensions
    const char* membrane;
    uint64_t timestamp;
    const uint8_t* payload;
    uint32_t payload_len;
} pomai_upsert_t;

typedef struct {
    uint32_t struct_size;
    uint64_t id;
    uint32_t dim;
    const float* vector;
    const uint8_t* metadata;
    uint32_t metadata_len;
    bool is_deleted;
    // Multi-membrane & temporal extensions
    uint64_t timestamp;
    const uint8_t* payload;
    uint32_t payload_len;
} pomai_record_t;

// Current-row record view for iterators.
// Pointers are valid only until pomai_iter_next() or pomai_iter_free().
typedef struct {
    uint32_t struct_size;
    uint64_t id;
    uint32_t dim;
    const float* vector;
    const uint8_t* metadata;
    uint32_t metadata_len;
    bool is_deleted;
    // Multi-membrane & temporal extensions
    uint64_t timestamp;
    const uint8_t* payload;
    uint32_t payload_len;
} pomai_record_view_t;

typedef struct {
    uint32_t struct_size;
    const float* vector;
    uint32_t dim;
    uint32_t topk;
    const char* filter_expression;
    const char* partition_device_id;
    const char* partition_location_id;
    uint32_t deadline_ms;
    uint32_t flags;
    // Multi-membrane & temporal extensions
    const char* membrane;
    uint64_t as_of_ts;
    uint64_t as_of_lsn;
} pomai_query_t;

typedef struct {
    uint32_t struct_size;
    const void* raw_data_ptr;
    uint32_t dim;
    float quant_min;
    float quant_inv_scale;
    uint64_t session_id;
} pomai_semantic_pointer_t;

typedef struct {
    uint32_t struct_size;
    size_t count;
    uint64_t* ids;
    float* scores;
    uint32_t* shard_ids;
    uint32_t total_shards_count;
    uint32_t pruned_shards_count;
    pomai_semantic_pointer_t* zero_copy_pointers;
} pomai_search_results_t;

typedef struct {
    uint32_t struct_size;
    uint64_t start_id;
    bool has_start_id;
    uint32_t deadline_ms;
    // Multi-membrane extension
    const char* membrane;
} pomai_scan_options_t;

#define POMAI_MEMBRANE_KIND_VECTOR 0

#ifdef __cplusplus
}
#endif

#endif // POMAI_C_TYPES_H
