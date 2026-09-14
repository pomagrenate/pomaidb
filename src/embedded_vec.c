#include "pomai/embedded_vec.h"

#include <palloc.h>
#include <palloc_vector.h>
#include <palloc/arena_pomai.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
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
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <immintrin.h>
#endif

#define EV_ALIGNMENT 64
#define EV_DEFAULT_MEMTABLE_CAPACITY 65536

/* --------------------------------------------------------------------------
 * Hardware Bit-Twiddling Helper
 * ----------------------------------------------------------------------- */
static inline int ev_tzcnt_u64(uint64_t x) {
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
 * Distance Kernels (AVX2 & Scalar Fallback)
 * ----------------------------------------------------------------------- */

static inline float ev_dist_l2_sq_avx2(const float* a, const float* b, size_t dim) {
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

static inline float ev_dist_cosine_avx2(const float* a, const float* b, size_t dim) {
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

static inline float ev_compute_distance(ev_metric_t metric, const float* a, const float* b, size_t dim) {
    if (metric == EV_METRIC_COSINE) {
        return ev_dist_cosine_avx2(a, b, dim);
    }
    return ev_dist_l2_sq_avx2(a, b, dim);
}

/* --------------------------------------------------------------------------
 * Top-K Max-Heap
 * ----------------------------------------------------------------------- */

typedef struct {
    uint64_t id;
    float    distance;
} ev_candidate_t;

typedef struct {
    size_t          capacity;
    size_t          size;
    ev_candidate_t* data;
} ev_heap_t;

static inline void ev_heap_swap(ev_candidate_t* a, ev_candidate_t* b) {
    ev_candidate_t tmp = *a;
    *a = *b;
    *b = tmp;
}

static void ev_heap_sift_down(ev_heap_t* h, size_t idx) {
    while (2 * idx + 1 < h->size) {
        size_t left = 2 * idx + 1;
        size_t right = left + 1;
        size_t largest = idx;

        if (h->data[left].distance > h->data[largest].distance) largest = left;
        if (right < h->size && h->data[right].distance > h->data[largest].distance) largest = right;
        if (largest == idx) break;
        ev_heap_swap(&h->data[idx], &h->data[largest]);
        idx = largest;
    }
}

static void ev_heap_push(ev_heap_t* h, uint64_t id, float dist) {
    if (h->size < h->capacity) {
        h->data[h->size] = (ev_candidate_t){id, dist};
        size_t idx = h->size++;
        while (idx > 0) {
            size_t parent = (idx - 1) / 2;
            if (h->data[idx].distance > h->data[parent].distance) {
                ev_heap_swap(&h->data[idx], &h->data[parent]);
                idx = parent;
            } else break;
        }
    } else if (dist < h->data[0].distance) {
        h->data[0] = (ev_candidate_t){id, dist};
        ev_heap_sift_down(h, 0);
    }
}

/* --------------------------------------------------------------------------
 * Core Internal Engine Structures
 * ----------------------------------------------------------------------- */

typedef struct {
    pa_arena_t* arena;            /* Palloc Vector Arena for contiguous bump allocations */
    uint64_t*   ids;              /* IDs array */
    float*      vectors;          /* Flat contiguous float buffer (dim * capacity) */
    size_t      count;
    size_t      capacity;
} ev_memtable_t;

struct ev_db_s {
    char*         storage_path;
    ev_options_t  options;
    ev_memtable_t active_memtable;
    size_t        total_committed_vectors;
};

void ev_options_init(ev_options_t* options, size_t vector_dim) {
    if (!options) return;
    options->vector_dim = vector_dim;
    options->metric = EV_METRIC_L2_SQ;
    options->memtable_capacity = EV_DEFAULT_MEMTABLE_CAPACITY;
    options->wal_directory = NULL;
    options->enable_direct_io = false;
}

int ev_db_open(const char* storage_path, const ev_options_t* options, ev_db_t** out_db) {
    if (!out_db) return EINVAL;
    if (!options || options->vector_dim == 0) return EINVAL;

    ev_db_t* db = (ev_db_t*)pa_malloc(sizeof(ev_db_t));
    if (!db) return ENOMEM;
    memset(db, 0, sizeof(ev_db_t));

    db->options = *options;
    if (db->options.memtable_capacity == 0) {
        db->options.memtable_capacity = EV_DEFAULT_MEMTABLE_CAPACITY;
    }
    if (storage_path) {
        size_t len = strlen(storage_path);
        db->storage_path = (char*)pa_malloc(len + 1);
        if (db->storage_path) memcpy(db->storage_path, storage_path, len + 1);
    }

    size_t cap = db->options.memtable_capacity;
    size_t dim = db->options.vector_dim;
    size_t vec_bytes = cap * dim * sizeof(float);
    size_t ids_bytes = cap * sizeof(uint64_t);
    size_t total_arena_bytes = vec_bytes + ids_bytes + 65536;

    db->active_memtable.arena = p_arena_create_for_vector_ex(total_arena_bytes, false);
    if (!db->active_memtable.arena) {
        if (db->storage_path) pa_free(db->storage_path);
        pa_free(db);
        return ENOMEM;
    }

    db->active_memtable.vectors = (float*)p_arena_alloc_vector(db->active_memtable.arena, cap * dim, sizeof(float));
    db->active_memtable.ids     = (uint64_t*)p_arena_alloc(db->active_memtable.arena, ids_bytes);
    db->active_memtable.capacity = cap;
    db->active_memtable.count = 0;

    *out_db = db;
    return 0;
}

int ev_insert_batch(
    ev_db_t* db,
    const uint64_t* ids,
    const float* vectors,
    size_t count,
    const char** metadata_jsons
) {
    (void)metadata_jsons;
    if (!db || !ids || !vectors || count == 0) return EINVAL;

    ev_memtable_t* m = &db->active_memtable;
    if (m->count + count > m->capacity) {
        /* MemTable full: flush checkpoint first */
        int rc = ev_checkpoint(db);
        if (rc != 0) return rc;
    }

    size_t dim = db->options.vector_dim;
    memcpy(m->ids + m->count, ids, count * sizeof(uint64_t));
    memcpy(m->vectors + (m->count * dim), vectors, count * dim * sizeof(float));
    m->count += count;
    db->total_committed_vectors += count;

    return 0;
}

int ev_query_knn(
    ev_db_t* db,
    const float* query_vector,
    size_t top_k,
    const uint64_t* filter_mask,
    uint64_t* out_ids,
    float* out_distances,
    size_t* out_actual_k
) {
    if (!db || !query_vector || top_k == 0 || !out_ids || !out_distances) return EINVAL;

    ev_memtable_t* m = &db->active_memtable;
    size_t dim = db->options.vector_dim;
    ev_metric_t metric = db->options.metric;

    ev_candidate_t* heap_buf = (ev_candidate_t*)pa_malloc(top_k * sizeof(ev_candidate_t));
    if (!heap_buf) return ENOMEM;

    ev_heap_t heap;
    heap.capacity = top_k;
    heap.size = 0;
    heap.data = heap_buf;

    size_t total_words = (m->count + 63) / 64;

    for (size_t w = 0; w < total_words; w++) {
        uint64_t word = filter_mask ? filter_mask[w] : ~0ULL;

        /* Hardware Bitwise Acceleration: Skip 64 vectors in 1 branch test */
        if (word == 0ULL) continue;

        while (word != 0ULL) {
            int bit = ev_tzcnt_u64(word);
            size_t idx = w * 64 + (size_t)bit;

            if (idx < m->count) {
                const float* target = m->vectors + (idx * dim);
                float dist = ev_compute_distance(metric, query_vector, target, dim);
                ev_heap_push(&heap, m->ids[idx], dist);
            }
            word &= (word - 1ULL);
        }
    }

    size_t actual = heap.size;
    for (size_t i = heap.size; i > 0; i--) {
        ev_candidate_t c = heap.data[0];
        heap.data[0] = heap.data[i - 1];
        heap.size--;
        ev_heap_sift_down(&heap, 0);
        out_ids[i - 1] = c.id;
        out_distances[i - 1] = c.distance;
    }

    if (out_actual_k) *out_actual_k = actual;
    pa_free(heap_buf);
    return 0;
}

int ev_checkpoint(ev_db_t* db) {
    if (!db) return EINVAL;
    /* Fast O(1) rewind of active Palloc Arena commits */
    if (db->active_memtable.arena) {
        p_arena_reset(db->active_memtable.arena);
        size_t cap = db->active_memtable.capacity;
        size_t dim = db->options.vector_dim;
        db->active_memtable.vectors = (float*)p_arena_alloc_vector(db->active_memtable.arena, cap * dim, sizeof(float));
        db->active_memtable.ids     = (uint64_t*)p_arena_alloc(db->active_memtable.arena, cap * sizeof(uint64_t));
        db->active_memtable.count   = 0;
    }
    return 0;
}

int ev_get_stats(ev_db_t* db, ev_stats_t* out_stats) {
    if (!db || !out_stats) return EINVAL;
    out_stats->total_vectors = db->total_committed_vectors;
    out_stats->memtable_vectors = db->active_memtable.count;
    out_stats->sealed_segments = 0;
    out_stats->arena_allocated_bytes = db->active_memtable.count * db->options.vector_dim * sizeof(float);
    out_stats->arena_committed_bytes = db->active_memtable.capacity * db->options.vector_dim * sizeof(float);
    return 0;
}

int ev_db_close(ev_db_t* db) {
    if (!db) return 0;
    if (db->active_memtable.arena) {
        p_arena_destroy(db->active_memtable.arena);
        db->active_memtable.arena = NULL;
    }
    if (db->storage_path) {
        pa_free(db->storage_path);
        db->storage_path = NULL;
    }
    pa_free(db);
    return 0;
}
