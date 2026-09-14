#ifndef POMAIDB_H
#define POMAIDB_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(_WIN32) || defined(_WIN64)
  #if defined(POMAIDB_BUILD_DLL)
    #define PDB_EXPORT __declspec(dllexport)
  #else
    #define PDB_EXPORT
  #endif
#else
  #define PDB_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Status & Error Codes
 * ----------------------------------------------------------------------- */
typedef enum {
    PDB_SUCCESS              = 0,
    PDB_ERR_INVALID_ARGUMENT = 1,
    PDB_ERR_OUT_OF_MEMORY    = 2,
    PDB_ERR_IO_FAILURE       = 3,
    PDB_ERR_CORRUPTION       = 4,
    PDB_ERR_NOT_FOUND        = 5,
    PDB_ERR_CAPACITY_EXCEEDED= 6,
    PDB_ERR_INTERNAL         = 7
} pdb_status_t;

/* --------------------------------------------------------------------------
 * Distance Metrics
 * ----------------------------------------------------------------------- */
typedef enum {
    PDB_METRIC_L2     = 0,  /* Squared Euclidean Distance (AVX2 FMA unrolled) */
    PDB_METRIC_COSINE = 1,  /* Cosine Distance (1.0 - dot_product) */
    PDB_METRIC_DOT    = 2   /* Inner Dot Product */
} pdb_metric_t;

/* --------------------------------------------------------------------------
 * Explicit 64-Byte Aligned Vector Batch Descriptor
 * ----------------------------------------------------------------------- */
typedef struct {
    const uint64_t* ids;              /* Array of uint64_t vector IDs */
    const float*    vectors;          /* Contiguous float matrix (count * dim), 64-byte aligned */
    size_t          count;            /* Number of vectors in this batch */
    size_t          dim;              /* Dimensionality of each vector */
    const char**    metadata_jsons;   /* Optional null-terminated metadata JSON array (can be NULL) */
} pdb_vector_batch_t;

/* --------------------------------------------------------------------------
 * Engine Configuration Options
 * ----------------------------------------------------------------------- */
typedef struct {
    size_t       dim;                  /* Fixed vector dimension (e.g. 64, 128, 512, 1536) */
    pdb_metric_t metric;               /* Vector distance metric */
    size_t       memtable_capacity;    /* Max in-memory vectors before triggering segment flush (e.g. 65536) */
    const char*  wal_directory;        /* Directory path for WAL files (NULL for in-memory only) */
    bool         enable_direct_io;     /* Use unbuffered Direct I/O for WAL & Segment flushes */
    size_t       arena_reserve_bytes;  /* Virtual memory reservation size for palloc vector arena */
} pdb_options_t;

/* --------------------------------------------------------------------------
 * Engine Operational Statistics
 * ----------------------------------------------------------------------- */
typedef struct {
    size_t total_vectors;              /* Total live vectors across all segments + memtable */
    size_t memtable_vectors;           /* Active volatile vectors in current MemTable */
    size_t sealed_segments;            /* Total sealed immutable on-disk segments */
    size_t arena_allocated_bytes;      /* Active allocated bytes in volatile arena */
    size_t arena_committed_bytes;      /* Physical RAM committed by palloc vector arena */
    uint64_t current_lsn;              /* High-water Log Sequence Number */
} pdb_stats_t;

/* Opaque Engine Handle */
typedef struct pdb_s pdb_t;

/* --------------------------------------------------------------------------
 * Lifecycle & Administration API
 * ----------------------------------------------------------------------- */
PDB_EXPORT pdb_status_t pdb_options_init(pdb_options_t* options, size_t dim);
PDB_EXPORT pdb_status_t pdb_open(const char* storage_path, const pdb_options_t* options, pdb_t** out_db);
PDB_EXPORT pdb_status_t pdb_close(pdb_t* db);

/* --------------------------------------------------------------------------
 * Ingestion, Search & Deletion API
 * ----------------------------------------------------------------------- */
PDB_EXPORT pdb_status_t pdb_insert_batch(pdb_t* db, const pdb_vector_batch_t* batch);

PDB_EXPORT pdb_status_t pdb_query_knn(
    pdb_t* db,
    const float* query_vector,
    size_t top_k,
    const uint64_t* filter_mask,       /* 64-bit word mask (NULL for unfiltered search) */
    uint64_t* out_ids,
    float* out_distances,
    size_t* out_actual_k
);

PDB_EXPORT pdb_status_t pdb_delete_batch(pdb_t* db, const uint64_t* ids, size_t count);

/* --------------------------------------------------------------------------
 * Durability & Maintenance API
 * ----------------------------------------------------------------------- */
PDB_EXPORT pdb_status_t pdb_checkpoint(pdb_t* db);
PDB_EXPORT pdb_status_t pdb_get_stats(pdb_t* db, pdb_stats_t* out_stats);

#ifdef __cplusplus
}
#endif

#endif /* POMAIDB_H */
