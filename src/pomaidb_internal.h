#ifndef POMAIDB_INTERNAL_H
#define POMAIDB_INTERNAL_H

#include "pomai/pomaidb.h"

#include <palloc.h>
#include <palloc_vector.h>
#include <palloc/arena_pomai.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>

#if defined(_WIN32) || defined(_WIN64)
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
  #include <pthread.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <immintrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PDB_PAGE_SIZE 4096
#define PDB_VECTOR_ALIGNMENT 64
#define PDB_DEFAULT_MEMTABLE_CAPACITY 65536
#define PDB_MAX_SEGMENTS 1024

/* --------------------------------------------------------------------------
 * Hardware & Software CRC32C (Castagnoli 0x82F63B78)
 * ----------------------------------------------------------------------- */
static inline uint32_t pdb_crc32c(const void* data, size_t len, uint32_t seed) {
    const uint8_t* p = (const uint8_t*)data;
    uint32_t crc = ~seed;
    for (size_t i = 0; i < len; i++) {
        crc ^= p[i];
        for (int k = 0; k < 8; k++) {
            crc = (crc >> 1) ^ (0x82F63B78u & -(crc & 1u));
        }
    }
    return ~crc;
}

static inline int pdb_tzcnt_u64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return (int)idx;
#else
    if (x == 0) return 64;
    int n = 0;
    if ((x & 0xFFFFFFFF) == 0) { n += 32; x >>= 32; }
    if ((x & 0x0000FFFF) == 0) { n += 16; x >>= 16; }
    if ((x & 0x000000FF) == 0) { n += 8;  x >>= 8;  }
    if ((x & 0x0000000F) == 0) { n += 4;  x >>= 4;  }
    if ((x & 0x00000003) == 0) { n += 2;  x >>= 2;  }
    if ((x & 0x00000001) == 0) { n += 1; }
    return n;
#endif
}

/* --------------------------------------------------------------------------
 * Distance Kernels (AVX2 FMA & Portable)
 * ----------------------------------------------------------------------- */
static inline float pdb_dist_l2_sq_avx2(const float* a, const float* b, size_t dim) {
#if defined(__AVX2__)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 15 < dim; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 d0  = _mm256_sub_ps(va0, vb0);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 d1  = _mm256_sub_ps(va1, vb1);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
    }
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d  = _mm256_sub_ps(va, vb);
        sum0 = _mm256_fmadd_ps(d, d, sum0);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    __m128 vlow  = _mm256_castps256_ps128(sum0);
    __m128 vhigh = _mm256_extractf128_ps(sum0, 1);
    __m128 vsum  = _mm_add_ps(vlow, vhigh);
    vsum = _mm_add_ps(vsum, _mm_movehl_ps(vsum, vsum));
    vsum = _mm_add_ss(vsum, _mm_movehdup_ps(vsum));
    float res = _mm_cvtss_f32(vsum);
    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        res += diff * diff;
    }
    return res;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
#endif
}

static inline float pdb_dist_cosine_avx2(const float* a, const float* b, size_t dim) {
#if defined(__AVX2__)
    __m256 dot0 = _mm256_setzero_ps();
    __m256 dot1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 15 < dim; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        dot0 = _mm256_fmadd_ps(va0, vb0, dot0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        dot1 = _mm256_fmadd_ps(va1, vb1, dot1);
    }
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        dot0 = _mm256_fmadd_ps(va, vb, dot0);
    }
    dot0 = _mm256_add_ps(dot0, dot1);
    __m128 vlow  = _mm256_castps256_ps128(dot0);
    __m128 vhigh = _mm256_extractf128_ps(dot0, 1);
    __m128 vsum  = _mm_add_ps(vlow, vhigh);
    vsum = _mm_add_ps(vsum, _mm_movehl_ps(vsum, vsum));
    vsum = _mm_add_ss(vsum, _mm_movehdup_ps(vsum));
    float dot = _mm_cvtss_f32(vsum);
    for (; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return 1.0f - dot;
#else
    float dot = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return 1.0f - dot;
#endif
}

static inline float pdb_compute_distance(pdb_metric_t metric, const float* a, const float* b, size_t dim) {
    if (metric == PDB_METRIC_COSINE) {
        return pdb_dist_cosine_avx2(a, b, dim);
    }
    return pdb_dist_l2_sq_avx2(a, b, dim);
}

/* --------------------------------------------------------------------------
 * Top-K Max-Heap
 * ----------------------------------------------------------------------- */
typedef struct {
    uint64_t id;
    float    distance;
} pdb_candidate_t;

typedef struct {
    size_t           capacity;
    size_t           size;
    pdb_candidate_t* data;
} pdb_heap_t;

static inline void pdb_heap_swap(pdb_candidate_t* a, pdb_candidate_t* b) {
    pdb_candidate_t tmp = *a;
    *a = *b;
    *b = tmp;
}

static inline void pdb_heap_sift_down(pdb_heap_t* h, size_t idx) {
    while (2 * idx + 1 < h->size) {
        size_t left = 2 * idx + 1;
        size_t right = left + 1;
        size_t largest = idx;

        if (h->data[left].distance > h->data[largest].distance) largest = left;
        if (right < h->size && h->data[right].distance > h->data[largest].distance) largest = right;
        if (largest == idx) break;
        pdb_heap_swap(&h->data[idx], &h->data[largest]);
        idx = largest;
    }
}

static inline void pdb_heap_push(pdb_heap_t* h, uint64_t id, float dist) {
    if (h->size < h->capacity) {
        h->data[h->size] = (pdb_candidate_t){id, dist};
        size_t idx = h->size++;
        while (idx > 0) {
            size_t parent = (idx - 1) / 2;
            if (h->data[idx].distance > h->data[parent].distance) {
                pdb_heap_swap(&h->data[idx], &h->data[parent]);
                idx = parent;
            } else break;
        }
    } else if (dist < h->data[0].distance) {
        h->data[0] = (pdb_candidate_t){id, dist};
        pdb_heap_sift_down(h, 0);
    }
}

/* --------------------------------------------------------------------------
 * WAL Physical Binary Layout
 * ----------------------------------------------------------------------- */
#pragma pack(push, 1)
typedef struct __attribute__((packed)) {
    uint8_t  magic[8];             /* "POMAIWAL" */
    uint64_t lsn;                  /* Log Sequence Number (8) */
    uint64_t txn_id;               /* Transaction ID (8) */
    uint32_t vector_count;         /* Number of vectors in frame (4) */
    uint32_t vector_dim;           /* Dimension (4) */
    uint32_t flags;                /* 0x01 = Commit, 0x02 = Delete (4) */
    uint32_t header_crc32c;        /* CRC32C over header (4) */
    uint64_t payload_bytes;        /* Total payload bytes (8) */
    uint64_t timestamp_ns;         /* Unix timestamp in nanoseconds (8) */
    uint8_t  pad[8];               /* 8+8+8+4+4+4+4+8+8 = 56 bytes. Pad 8 = 64 bytes */
} pdb_wal_header_t;
_Static_assert(sizeof(pdb_wal_header_t) == 64, "pdb_wal_header_t must be 64 bytes");

typedef struct __attribute__((packed)) {
    uint64_t magic_tail;           /* 0x21444E455F4C4157 ("WAL_END!") (8) */
    uint64_t lsn_tail;             /* Matches header LSN (8) */
    uint32_t payload_crc32c;       /* CRC32C over vector payload + IDs (4) */
    uint32_t composite_crc32c;     /* CRC32C over header + payload (4) */
    uint8_t  pad[40];              /* 8+8+4+4 = 24. Pad 40 = 64 bytes */
} pdb_wal_footer_t;
_Static_assert(sizeof(pdb_wal_footer_t) == 64, "pdb_wal_footer_t must be 64 bytes");
#pragma pack(pop)

typedef struct pdb_wal_s {
    char     filepath[512];
    FILE*    file_handle;
    uint64_t current_lsn;
    bool     direct_io;
} pdb_wal_t;

/* --------------------------------------------------------------------------
 * Immutable Segment File Format & mmap Reader
 * ----------------------------------------------------------------------- */
#pragma pack(push, 1)
typedef struct __attribute__((packed)) {
    uint8_t  magic[8];             /* "POMAISEG" (8) */
    uint32_t format_version;       /* 1 (4) */
    uint32_t vector_dim;           /* (4) */
    uint16_t metric_type;          /* (2) */
    uint16_t reserved16;           /* (2) */
    uint64_t total_vectors;        /* (8) */
    uint64_t vector_data_offset;   /* (8) */
    uint64_t id_data_offset;       /* (8) */
    uint64_t tombstone_offset;     /* (8) */
    uint64_t segment_lsn;          /* (8) */
    uint8_t  pad[4032];            /* 8+4+4+2+2+8+8+8+8+8 = 60 bytes. 60 + 4032 + 4 = 4096 bytes */
    uint32_t header_crc32c;        /* (4) */
} pdb_seg_header_t;
_Static_assert(sizeof(pdb_seg_header_t) == 4096, "pdb_seg_header_t must be 4096 bytes");

#pragma pack(pop)

typedef struct pdb_segment_s {
    char     filepath[512];
    void*    mmap_base;
    size_t   mmap_size;
#if defined(_WIN32) || defined(_WIN64)
    HANDLE   file_handle;
    HANDLE   mapping_handle;
#else
    int      fd;
#endif
    const pdb_seg_header_t* header;
    const float*    vectors;       /* Direct pointer into mmap zero-copy space */
    const uint64_t* ids;           /* Direct pointer into mmap ID table */
    uint64_t*       tombstones;    /* Mutable tombstone bitmap (heap copy or cow) */
    size_t          count;
    size_t          dim;
} pdb_segment_t;

/* --------------------------------------------------------------------------
 * Volatile MemTable
 * ----------------------------------------------------------------------- */
typedef struct {
    pa_vec_arena_t* arena;         /* Backed by palloc vector arena */
    pa_vec_pool_t*  metadata_pool; /* Backed by palloc dense bitmap pool */
    uint64_t*       ids;
    float*          vectors;       /* 64-byte aligned SIMD matrix */
    uint64_t*       tombstones;    /* Tombstone bitmap for deleted vectors in memtable */
    size_t          count;
    size_t          capacity;
} pdb_memtable_t;

/* --------------------------------------------------------------------------
 * Database Internal Engine (pdb_t)
 * ----------------------------------------------------------------------- */
struct pdb_s {
    char            storage_path[512];
    pdb_options_t   options;
    pdb_memtable_t  memtable;
    pdb_wal_t       wal;
    pdb_segment_t*  segments[PDB_MAX_SEGMENTS];
    size_t          segment_count;
    size_t          total_vectors;
    uint64_t        current_lsn;
    uint64_t        global_epoch;  /* Atomic epoch counter for MVCC readers */
};

/* Internal function declarations */
pdb_status_t pdb_wal_init(pdb_wal_t* wal, const char* dir_path, bool direct_io);
pdb_status_t pdb_wal_append(pdb_wal_t* wal, const pdb_vector_batch_t* batch, uint64_t* out_lsn);
pdb_status_t pdb_wal_recover(pdb_wal_t* wal, pdb_t* db);
void         pdb_wal_close(pdb_wal_t* wal);

pdb_status_t pdb_segment_flush(pdb_t* db, const pdb_memtable_t* memtable, const char* out_filepath);
pdb_status_t pdb_segment_open(const char* filepath, pdb_segment_t** out_seg);
void         pdb_segment_close(pdb_segment_t* seg);

#ifdef __cplusplus
}
#endif

#endif /* POMAIDB_INTERNAL_H */
