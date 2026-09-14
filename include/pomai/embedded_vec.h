#ifndef EMBEDDED_VEC_H
#define EMBEDDED_VEC_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(_WIN32) || defined(_WIN64)
  #if defined(EMBEDDED_VEC_BUILD_DLL)
    #define EV_EXPORT __declspec(dllexport)
  #else
    #define EV_EXPORT
  #endif
#else
  #define EV_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque database handle */
typedef struct ev_db_s ev_db_t;

/* Distance metric supported by the SIMD vector kernels */
typedef enum {
    EV_METRIC_L2_SQ = 0,   /* Squared Euclidean distance (AVX2 unrolled FMA) */
    EV_METRIC_COSINE = 1,  /* Cosine distance (1.0 - Cosine Similarity) */
    EV_METRIC_DOT    = 2   /* Raw inner dot product */
} ev_metric_t;

/* Database configuration options */
typedef struct {
    size_t      vector_dim;          /* Fixed vector dimensionality (e.g. 64, 128, 768, 1536) */
    ev_metric_t metric;              /* Distance metric selection */
    size_t      memtable_capacity;   /* Max in-memory vectors before seal/flush trigger */
    const char* wal_directory;       /* Path for Write-Ahead Log durability files (NULL for in-memory) */
    bool        enable_direct_io;    /* Bypass OS file cache with sector-aligned Direct I/O */
} ev_options_t;

/* Default options initialiser */
EV_EXPORT void ev_options_init(ev_options_t* options, size_t vector_dim);

/* Lifecycle management */
EV_EXPORT int ev_db_open(const char* storage_path, const ev_options_t* options, ev_db_t** out_db);
EV_EXPORT int ev_db_close(ev_db_t* db);

/* Ingestion & Mutation */
EV_EXPORT int ev_insert_batch(
    ev_db_t* db,
    const uint64_t* ids,
    const float* vectors,
    size_t count,
    const char** metadata_jsons
);

/* Approximate / Exact KNN Vector Search with Bitset-Interleaved Filtering */
EV_EXPORT int ev_query_knn(
    ev_db_t* db,
    const float* query_vector,
    size_t top_k,
    const uint64_t* filter_mask,     /* Dense 64-bit word mask (NULL for unfiltered search) */
    uint64_t* out_ids,
    float* out_distances,
    size_t* out_actual_k
);

/* Flush active MemTable to an immutable on-disk PVS segment and commit checkpoint */
EV_EXPORT int ev_checkpoint(ev_db_t* db);

/* Retrieve current engine operational statistics */
typedef struct {
    size_t total_vectors;
    size_t memtable_vectors;
    size_t sealed_segments;
    size_t arena_allocated_bytes;
    size_t arena_committed_bytes;
} ev_stats_t;

EV_EXPORT int ev_get_stats(ev_db_t* db, ev_stats_t* out_stats);

#ifdef __cplusplus
}
#endif

#endif /* EMBEDDED_VEC_H */
