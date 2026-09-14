#ifndef POMAIDB_HNSW_MMAP_H
#define POMAIDB_HNSW_MMAP_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(_WIN32) || defined(_WIN64)
  #if defined(POMAIDB_BUILD_DLL)
    #define PDB_HNSW_EXPORT __declspec(dllexport)
  #else
    #define PDB_HNSW_EXPORT
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#else
  #define PDB_HNSW_EXPORT __attribute__((visibility("default")))
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PDB_HNSW_MMAP_MAGIC 0x534E4849414D4F50ULL /* "POMAIHNS" */
#define PDB_HNSW_PAGE_SIZE  4096
#define PDB_HNSW_MAX_LEVELS 16

/* --------------------------------------------------------------------------
 * 4096-Byte Page-Aligned Segment File Header
 * ----------------------------------------------------------------------- */
#pragma pack(push, 1)
typedef struct {
    uint64_t magic;               /* "POMAIHNS" (8) */
    uint32_t version;             /* 1 (4) */
    uint32_t dim;                 /* Vector dimension (4) */
    uint32_t metric;              /* 0 = L2, 1 = Cosine (4) */
    uint32_t M;                   /* Max links per upper layer (4) */
    uint32_t M0;                  /* Max links for base layer 0 (4) */
    uint32_t max_level;           /* Maximum layer index in graph (4) */
    uint32_t entry_node;          /* Global entry node internal ID (4) */
    uint64_t total_nodes;         /* Total number of nodes in index (8) */
    uint64_t node_record_stride;  /* Stride per interleaved node record (8) */
    uint64_t vector_rel_offset;   /* Byte offset to vector payload inside record (8) */
    uint8_t  pad[4032];           /* Pad to exact 4096-byte boundary */
    uint32_t header_crc32c;       /* Castagnoli CRC32C over bytes 0..4091 (4) */
} pdb_hnsw_mmap_header_t;
#pragma pack(pop)

#ifdef __cplusplus
static_assert(sizeof(pdb_hnsw_mmap_header_t) == 4096, "pdb_hnsw_mmap_header_t must be 4096 bytes");
#else
_Static_assert(sizeof(pdb_hnsw_mmap_header_t) == 4096, "pdb_hnsw_mmap_header_t must be 4096 bytes");
#endif

/* --------------------------------------------------------------------------
 * Zero-Copy mmap Graph Handle
 * ----------------------------------------------------------------------- */
typedef struct {
    const pdb_hnsw_mmap_header_t* header;
    const uint8_t* raw_base;
    size_t         file_size;
#if defined(_WIN32) || defined(_WIN64)
    HANDLE         file_handle;
    HANDLE         mapping_handle;
#else
    int            fd;
#endif
} pdb_hnsw_mmap_t;

/* --------------------------------------------------------------------------
 * In-Memory HNSW Builder (Volatile construction before freezing to mmap file)
 * ----------------------------------------------------------------------- */
typedef struct pdb_hnsw_builder_s pdb_hnsw_builder_t;

PDB_HNSW_EXPORT pdb_hnsw_builder_t* pdb_hnsw_builder_create(
    uint32_t dim,
    uint32_t metric,
    uint32_t M,
    uint32_t ef_construction
);

PDB_HNSW_EXPORT int pdb_hnsw_builder_add(
    pdb_hnsw_builder_t* b,
    uint64_t external_id,
    const float* vector
);

PDB_HNSW_EXPORT int pdb_hnsw_builder_freeze_file(
    pdb_hnsw_builder_t* b,
    const char* out_filepath
);

PDB_HNSW_EXPORT void pdb_hnsw_builder_destroy(pdb_hnsw_builder_t* b);

/* --------------------------------------------------------------------------
 * Zero-Copy Query API (Operates directly on mapped page cache)
 * ----------------------------------------------------------------------- */
PDB_HNSW_EXPORT int pdb_hnsw_mmap_open(const char* filepath, pdb_hnsw_mmap_t* out_graph);
PDB_HNSW_EXPORT void pdb_hnsw_mmap_close(pdb_hnsw_mmap_t* graph);

PDB_HNSW_EXPORT int pdb_hnsw_mmap_search(
    const pdb_hnsw_mmap_t* graph,
    const float* query_vector,
    uint32_t top_k,
    uint32_t ef_search,
    const uint64_t* tombstone_mask,
    uint64_t* out_ids,
    float* out_distances,
    uint32_t* out_actual_k
);

#ifdef __cplusplus
}
#endif

#endif /* POMAIDB_HNSW_MMAP_H */
