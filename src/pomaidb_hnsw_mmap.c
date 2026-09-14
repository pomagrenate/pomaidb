#include "pomai/pomaidb_hnsw_mmap.h"

#include <palloc.h>
#include <palloc_vector.h>
#include <palloc/arena_pomai.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <immintrin.h>
#endif

#define PDB_HNSW_MAX_EF 1024

/* --------------------------------------------------------------------------
 * CRC32C (Castagnoli)
 * ----------------------------------------------------------------------- */
static inline uint32_t pdb_hnsw_crc32c(const void* data, size_t len, uint32_t seed) {
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

/* --------------------------------------------------------------------------
 * AVX2 FMA Distance Functions
 * ----------------------------------------------------------------------- */
static inline float pdb_hnsw_l2_sq_avx2(const float* a, const float* b, size_t dim) {
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

static inline float pdb_hnsw_cosine_avx2(const float* a, const float* b, size_t dim) {
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

static inline float pdb_hnsw_compute_distance(uint32_t metric, const float* a, const float* b, size_t dim) {
    if (metric == 1) {
        return pdb_hnsw_cosine_avx2(a, b, dim);
    }
    return pdb_hnsw_l2_sq_avx2(a, b, dim);
}

/* --------------------------------------------------------------------------
 * In-Memory HNSW Builder (Volatile construction before freezing)
 * ----------------------------------------------------------------------- */

typedef struct {
    uint32_t id;
    float    dist;
} pdb_hnsw_cand_t;

typedef struct {
    uint64_t external_id;
    uint32_t level;
    float*   vector;
    /* Layer links: layer 0 has up to M0 links; layers 1..level have up to M links */
    uint32_t* links[PDB_HNSW_MAX_LEVELS];
    uint32_t  link_counts[PDB_HNSW_MAX_LEVELS];
} pdb_builder_node_t;

struct pdb_hnsw_builder_s {
    uint32_t            dim;
    uint32_t            metric;
    uint32_t            M;
    uint32_t            M0;
    uint32_t            ef_construction;
    double              level_mult;
    int32_t             entry_node;
    int32_t             max_level;
    size_t              count;
    size_t              capacity;
    pdb_builder_node_t* nodes;
    pa_vec_arena_t*     arena;
};

static inline int pdb_builder_random_level(pdb_hnsw_builder_t* b) {
    double r = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    int lvl = (int)(-log(r) * b->level_mult);
    if (lvl >= PDB_HNSW_MAX_LEVELS) lvl = PDB_HNSW_MAX_LEVELS - 1;
    return lvl;
}

pdb_hnsw_builder_t* pdb_hnsw_builder_create(
    uint32_t dim,
    uint32_t metric,
    uint32_t M,
    uint32_t ef_construction
) {
    pdb_hnsw_builder_t* b = (pdb_hnsw_builder_t*)pa_malloc(sizeof(pdb_hnsw_builder_t));
    if (!b) return NULL;
    memset(b, 0, sizeof(pdb_hnsw_builder_t));

    b->dim = dim;
    b->metric = metric;
    b->M = (M > 0) ? M : 16;
    b->M0 = 2 * b->M;
    b->ef_construction = (ef_construction > 0) ? ef_construction : 200;
    b->level_mult = 1.0 / log((double)b->M);
    b->entry_node = -1;
    b->max_level = -1;
    b->count = 0;
    b->capacity = 1024;

    /* Reserve 32MB initial palloc vector arena for construction */
    b->arena = pa_vec_arena_create(32 * 1024 * 1024, 64);
    b->nodes = (pdb_builder_node_t*)pa_malloc(b->capacity * sizeof(pdb_builder_node_t));
    if (!b->nodes) {
        if (b->arena) pa_vec_arena_destroy(b->arena);
        pa_free(b);
        return NULL;
    }
    memset(b->nodes, 0, b->capacity * sizeof(pdb_builder_node_t));

    return b;
}

static void pdb_builder_connect_link(
    pdb_hnsw_builder_t* b,
    uint32_t src,
    uint32_t target,
    uint32_t level
) {
    pdb_builder_node_t* src_node = &b->nodes[src];
    uint32_t max_m = (level == 0) ? b->M0 : b->M;

    for (uint32_t i = 0; i < src_node->link_counts[level]; i++) {
        if (src_node->links[level][i] == target) return; /* Already connected */
    }

    if (src_node->link_counts[level] < max_m) {
        src_node->links[level][src_node->link_counts[level]++] = target;
    } else {
        /* Replace furthest neighbor if target is closer */
        const float* src_v = src_node->vector;
        const float* target_v = b->nodes[target].vector;
        float target_d = pdb_hnsw_compute_distance(b->metric, src_v, target_v, b->dim);

        float max_d = target_d;
        int max_idx = -1;
        for (uint32_t i = 0; i < max_m; i++) {
            uint32_t cur = src_node->links[level][i];
            float d = pdb_hnsw_compute_distance(b->metric, src_v, b->nodes[cur].vector, b->dim);
            if (d > max_d) {
                max_d = d;
                max_idx = (int)i;
            }
        }
        if (max_idx >= 0) {
            src_node->links[level][max_idx] = target;
        }
    }
}

int pdb_hnsw_builder_add(pdb_hnsw_builder_t* b, uint64_t external_id, const float* vector) {
    if (!b || !vector) return EINVAL;

    if (b->count >= b->capacity) {
        size_t new_cap = b->capacity * 2;
        pdb_builder_node_t* new_nodes = (pdb_builder_node_t*)pa_malloc(new_cap * sizeof(pdb_builder_node_t));
        if (!new_nodes) return ENOMEM;
        memcpy(new_nodes, b->nodes, b->count * sizeof(pdb_builder_node_t));
        memset(new_nodes + b->count, 0, (new_cap - b->count) * sizeof(pdb_builder_node_t));
        pa_free(b->nodes);
        b->nodes = new_nodes;
        b->capacity = new_cap;
    }

    uint32_t node_idx = (uint32_t)b->count;
    pdb_builder_node_t* node = &b->nodes[node_idx];
    node->external_id = external_id;
    node->level = (uint32_t)pdb_builder_random_level(b);

    /* Allocate vector in palloc arena */
    node->vector = (float*)pa_vec_arena_alloc(b->arena, b->dim * sizeof(float));
    if (!node->vector) {
        node->vector = (float*)pa_malloc(b->dim * sizeof(float));
    }
    memcpy(node->vector, vector, b->dim * sizeof(float));

    /* Allocate link lists */
    for (uint32_t lvl = 0; lvl <= node->level; lvl++) {
        uint32_t max_m = (lvl == 0) ? b->M0 : b->M;
        node->links[lvl] = (uint32_t*)pa_malloc(max_m * sizeof(uint32_t));
        node->link_counts[lvl] = 0;
    }

    /* First node initialization */
    if (b->entry_node < 0) {
        b->entry_node = (int32_t)node_idx;
        b->max_level = (int32_t)node->level;
        b->count++;
        return 0;
    }

    uint32_t curr = (uint32_t)b->entry_node;
    float d_curr = pdb_hnsw_compute_distance(b->metric, vector, b->nodes[curr].vector, b->dim);

    /* Search layers from top down to node->level + 1 */
    for (int lvl = b->max_level; lvl > (int)node->level; --lvl) {
        bool changed = true;
        while (changed) {
            changed = false;
            pdb_builder_node_t* cnode = &b->nodes[curr];
            for (uint32_t i = 0; i < cnode->link_counts[lvl]; i++) {
                uint32_t cand = cnode->links[lvl][i];
                float d = pdb_hnsw_compute_distance(b->metric, vector, b->nodes[cand].vector, b->dim);
                if (d < d_curr) {
                    d_curr = d;
                    curr = cand;
                    changed = true;
                }
            }
        }
    }

    /* Connect layers from min(node->level, max_level) down to 0 */
    int start_lvl = ((int)node->level < b->max_level) ? (int)node->level : b->max_level;
    for (int lvl = start_lvl; lvl >= 0; --lvl) {
        pdb_hnsw_cand_t candidates[PDB_HNSW_MAX_EF];
        uint32_t cand_count = 1;
        candidates[0] = (pdb_hnsw_cand_t){curr, d_curr};

        uint8_t visited[4096];
        memset(visited, 0, sizeof(visited));
        visited[curr % 4096] = 1;

        uint32_t ef = b->ef_construction;
        if (ef > PDB_HNSW_MAX_EF) ef = PDB_HNSW_MAX_EF;

        while (cand_count > 0) {
            uint32_t closest_idx = 0;
            for (uint32_t i = 1; i < cand_count; i++) {
                if (candidates[i].dist < candidates[closest_idx].dist) {
                    closest_idx = i;
                }
            }
            pdb_hnsw_cand_t top_c = candidates[closest_idx];
            candidates[closest_idx] = candidates[--cand_count];

            pdb_builder_node_t* cnode = &b->nodes[top_c.id];
            for (uint32_t i = 0; i < cnode->link_counts[lvl]; i++) {
                uint32_t n = cnode->links[lvl][i];
                if (visited[n % 4096]) continue;
                visited[n % 4096] = 1;

                float d = pdb_hnsw_compute_distance(b->metric, vector, b->nodes[n].vector, b->dim);
                if (cand_count < ef) {
                    candidates[cand_count++] = (pdb_hnsw_cand_t){n, d};
                }
            }
        }

        /* Bi-directional links to closest entry point */
        pdb_builder_connect_link(b, node_idx, curr, (uint32_t)lvl);
        pdb_builder_connect_link(b, curr, node_idx, (uint32_t)lvl);
    }

    if ((int32_t)node->level > b->max_level) {
        b->max_level = (int32_t)node->level;
        b->entry_node = (int32_t)node_idx;
    }

    b->count++;
    return 0;
}

/* --------------------------------------------------------------------------
 * Freezing Routine: Write Interleaved, Cache-Line Aligned Zero-Copy Format
 * ----------------------------------------------------------------------- */

int pdb_hnsw_builder_freeze_file(pdb_hnsw_builder_t* b, const char* out_filepath) {
    if (!b || !out_filepath) return EINVAL;

    FILE* f = fopen(out_filepath, "w+b");
    if (!f) return EIO;

    size_t dim = b->dim;
    size_t count = b->count;

    /* Compute fixed record stride:
     * [External ID: uint64_t (8B)]
     * [Level: uint16_t (2B)]
     * [Pad: uint8_t[6]]
     * [Offset Table: uint32_t * 16 (64B)]
     * [Layer 0 Links: uint16_t count (2B) + uint32_t * M0 (M0*4B)]
     * [Layer 1..15 Links: 15 * (uint16_t count (2B) + uint32_t * M (M*4B))]
     * [Vector: 64-byte aligned float array (dim * 4B)]
     */
    size_t links_overhead = 8 + 2 + 6 + (16 * sizeof(uint32_t));
    size_t layer0_bytes = 2 + (b->M0 * sizeof(uint32_t));
    size_t upper_layers_bytes = 15 * (2 + (b->M * sizeof(uint32_t)));
    size_t total_header_and_links = links_overhead + layer0_bytes + upper_layers_bytes;

    /* Align vector start within record to 64 bytes */
    size_t vector_rel_offset = (total_header_and_links + 63) & ~63ULL;
    size_t total_record = vector_rel_offset + (dim * sizeof(float));
    /* Align whole record stride to 64 bytes */
    size_t node_record_stride = (total_record + 63) & ~63ULL;

    pdb_hnsw_mmap_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = PDB_HNSW_MMAP_MAGIC;
    hdr.version = 1;
    hdr.dim = (uint32_t)dim;
    hdr.metric = b->metric;
    hdr.M = b->M;
    hdr.M0 = b->M0;
    hdr.max_level = (uint32_t)b->max_level;
    hdr.entry_node = (uint32_t)b->entry_node;
    hdr.total_nodes = count;
    hdr.node_record_stride = node_record_stride;
    hdr.vector_rel_offset = vector_rel_offset;

    hdr.header_crc32c = pdb_hnsw_crc32c(&hdr, offsetof(pdb_hnsw_mmap_header_t, header_crc32c), 0);

    /* 1. Write 4096-byte Page-Aligned Header */
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return EIO;
    }

    /* 2. Write Interleaved Node Records */
    uint8_t* rec_buf = (uint8_t*)pa_malloc(node_record_stride);
    if (!rec_buf) {
        fclose(f);
        return ENOMEM;
    }

    for (size_t i = 0; i < count; i++) {
        memset(rec_buf, 0, node_record_stride);
        const pdb_builder_node_t* n = &b->nodes[i];

        /* ID & Level */
        *(uint64_t*)(rec_buf) = n->external_id;
        *(uint16_t*)(rec_buf + 8) = (uint16_t)n->level;

        /* Fill Layer Offsets */
        uint32_t* layer_offsets = (uint32_t*)(rec_buf + 16);
        uint32_t cur_off = (uint32_t)links_overhead;

        layer_offsets[0] = cur_off;
        /* Layer 0 link data */
        uint8_t* l0_ptr = rec_buf + cur_off;
        *(uint16_t*)l0_ptr = (uint16_t)n->link_counts[0];
        if (n->link_counts[0] > 0) {
            memcpy(l0_ptr + 2, n->links[0], n->link_counts[0] * sizeof(uint32_t));
        }
        cur_off += (uint32_t)layer0_bytes;

        /* Upper layers */
        for (uint32_t lvl = 1; lvl < 16; lvl++) {
            layer_offsets[lvl] = cur_off;
            uint8_t* lvl_ptr = rec_buf + cur_off;
            if (lvl <= n->level && n->link_counts[lvl] > 0) {
                *(uint16_t*)lvl_ptr = (uint16_t)n->link_counts[lvl];
                memcpy(lvl_ptr + 2, n->links[lvl], n->link_counts[lvl] * sizeof(uint32_t));
            }
            cur_off += (uint32_t)(2 + (b->M * sizeof(uint32_t)));
        }

        /* Vector payload */
        float* vec_dest = (float*)(rec_buf + vector_rel_offset);
        memcpy(vec_dest, n->vector, dim * sizeof(float));

        if (fwrite(rec_buf, node_record_stride, 1, f) != 1) {
            pa_free(rec_buf);
            fclose(f);
            return EIO;
        }
    }

    pa_free(rec_buf);
    fflush(f);
    fclose(f);

    return 0;
}

void pdb_hnsw_builder_destroy(pdb_hnsw_builder_t* b) {
    if (!b) return;
    for (size_t i = 0; i < b->count; i++) {
        pdb_builder_node_t* n = &b->nodes[i];
        for (uint32_t lvl = 0; lvl <= n->level; lvl++) {
            if (n->links[lvl]) pa_free(n->links[lvl]);
        }
    }
    if (b->nodes) pa_free(b->nodes);
    if (b->arena) pa_vec_arena_destroy(b->arena);
    pa_free(b);
}

/* --------------------------------------------------------------------------
 * Zero-Copy mmap Engine (Queries execute directly on page cache)
 * ----------------------------------------------------------------------- */

static inline const uint8_t* pdb_hnsw_get_node_record(const pdb_hnsw_mmap_t* g, uint32_t node_id) {
    return g->raw_base + sizeof(pdb_hnsw_mmap_header_t) + (node_id * g->header->node_record_stride);
}

static inline const float* pdb_hnsw_get_node_vector(const pdb_hnsw_mmap_t* g, uint32_t node_id) {
    const uint8_t* rec = pdb_hnsw_get_node_record(g, node_id);
    return (const float*)(rec + g->header->vector_rel_offset);
}

static inline void pdb_hnsw_get_node_links(
    const pdb_hnsw_mmap_t* g,
    uint32_t node_id,
    uint32_t level,
    const uint32_t** out_neighbors,
    uint32_t* out_count
) {
    const uint8_t* rec = pdb_hnsw_get_node_record(g, node_id);
    uint32_t node_max_level = *(const uint16_t*)(rec + 8);

    if (level > node_max_level) {
        *out_count = 0;
        *out_neighbors = NULL;
        return;
    }

    const uint32_t* layer_offsets = (const uint32_t*)(rec + 16);
    uint32_t link_offset = layer_offsets[level];

    const uint8_t* link_ptr = rec + link_offset;
    *out_count = *(const uint16_t*)link_ptr;
    *out_neighbors = (const uint32_t*)(link_ptr + 2);
}

int pdb_hnsw_mmap_open(const char* filepath, pdb_hnsw_mmap_t* out_graph) {
    if (!filepath || !out_graph) return EINVAL;
    memset(out_graph, 0, sizeof(pdb_hnsw_mmap_t));

#if defined(_WIN32) || defined(_WIN64)
    out_graph->file_handle = CreateFileA(
        filepath,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    if (out_graph->file_handle == INVALID_HANDLE_VALUE) return EIO;

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(out_graph->file_handle, &sz)) {
        CloseHandle(out_graph->file_handle);
        return EIO;
    }
    out_graph->file_size = (size_t)sz.QuadPart;

    out_graph->mapping_handle = CreateFileMappingA(
        out_graph->file_handle,
        NULL,
        PAGE_READONLY,
        0,
        0,
        NULL
    );
    if (!out_graph->mapping_handle) {
        CloseHandle(out_graph->file_handle);
        return EIO;
    }

    out_graph->raw_base = (const uint8_t*)MapViewOfFile(out_graph->mapping_handle, FILE_MAP_READ, 0, 0, out_graph->file_size);
    if (!out_graph->raw_base) {
        CloseHandle(out_graph->mapping_handle);
        CloseHandle(out_graph->file_handle);
        return EIO;
    }
#else
    out_graph->fd = open(filepath, O_RDONLY);
    if (out_graph->fd < 0) return EIO;

    struct stat st;
    if (fstat(out_graph->fd, &st) < 0) {
        close(out_graph->fd);
        return EIO;
    }
    out_graph->file_size = (size_t)st.st_size;

    out_graph->raw_base = (const uint8_t*)mmap(NULL, out_graph->file_size, PROT_READ, MAP_SHARED, out_graph->fd, 0);
    if (out_graph->raw_base == MAP_FAILED) {
        close(out_graph->fd);
        return EIO;
    }
    madvise((void*)out_graph->raw_base, out_graph->file_size, MADV_RANDOM);
#endif

    out_graph->header = (const pdb_hnsw_mmap_header_t*)out_graph->raw_base;
    if (out_graph->header->magic != PDB_HNSW_MMAP_MAGIC) {
        pdb_hnsw_mmap_close(out_graph);
        return EBADF;
    }

    uint32_t expected_crc = pdb_hnsw_crc32c(out_graph->header, offsetof(pdb_hnsw_mmap_header_t, header_crc32c), 0);
    if (expected_crc != out_graph->header->header_crc32c) {
        pdb_hnsw_mmap_close(out_graph);
        return EBADF;
    }

    return 0;
}

void pdb_hnsw_mmap_close(pdb_hnsw_mmap_t* graph) {
    if (!graph) return;
#if defined(_WIN32) || defined(_WIN64)
    if (graph->raw_base) {
        UnmapViewOfFile(graph->raw_base);
        graph->raw_base = NULL;
    }
    if (graph->mapping_handle) {
        CloseHandle(graph->mapping_handle);
        graph->mapping_handle = NULL;
    }
    if (graph->file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(graph->file_handle);
        graph->file_handle = INVALID_HANDLE_VALUE;
    }
#else
    if (graph->raw_base && graph->raw_base != MAP_FAILED) {
        munmap((void*)graph->raw_base, graph->file_size);
        graph->raw_base = NULL;
    }
    if (graph->fd >= 0) {
        close(graph->fd);
        graph->fd = -1;
    }
#endif
}

int pdb_hnsw_mmap_search(
    const pdb_hnsw_mmap_t* graph,
    const float* query_vector,
    uint32_t top_k,
    uint32_t ef_search,
    const uint64_t* tombstone_mask,
    uint64_t* out_ids,
    float* out_distances,
    uint32_t* out_actual_k
) {
    if (!graph || !graph->header || !query_vector || top_k == 0 || !out_ids || !out_distances) {
        return EINVAL;
    }
    if (graph->header->total_nodes == 0) {
        if (out_actual_k) *out_actual_k = 0;
        return 0;
    }

    uint32_t dim = graph->header->dim;
    uint32_t metric = graph->header->metric;
    uint32_t curr = graph->header->entry_node;
    float d_curr = pdb_hnsw_compute_distance(metric, query_vector, pdb_hnsw_get_node_vector(graph, curr), dim);

    /* 1. Greedy search upper layers (Zero-Copy node traversals) */
    for (int lvl = (int)graph->header->max_level; lvl > 0; --lvl) {
        bool changed = true;
        while (changed) {
            changed = false;
            const uint32_t* neighbors;
            uint32_t count;
            pdb_hnsw_get_node_links(graph, curr, (uint32_t)lvl, &neighbors, &count);

            for (uint32_t i = 0; i < count; i++) {
                uint32_t cand = neighbors[i];
                float d = pdb_hnsw_compute_distance(metric, query_vector, pdb_hnsw_get_node_vector(graph, cand), dim);
                if (d < d_curr) {
                    d_curr = d;
                    curr = cand;
                    changed = true;
                }
            }
        }
    }

    /* 2. Base layer beam search using stack/scratchpad visited set */
    uint32_t ef = (ef_search > PDB_HNSW_MAX_EF) ? PDB_HNSW_MAX_EF : ef_search;
    if (ef < top_k) ef = top_k;

    pdb_hnsw_cand_t candidates[PDB_HNSW_MAX_EF];
    uint32_t cand_count = 1;
    candidates[0] = (pdb_hnsw_cand_t){curr, d_curr};

    pdb_hnsw_cand_t top_set[PDB_HNSW_MAX_EF];
    uint32_t top_count = 1;
    top_set[0] = (pdb_hnsw_cand_t){curr, d_curr};

    /* 2048 words = 131,072 bits, zero heap allocation for visited set */
    uint64_t visited[2048];
    memset(visited, 0, sizeof(visited));
    visited[(curr / 64) % 2048] |= (1ULL << (curr % 64));

    while (cand_count > 0) {
        uint32_t closest_idx = 0;
        for (uint32_t i = 1; i < cand_count; i++) {
            if (candidates[i].dist < candidates[closest_idx].dist) {
                closest_idx = i;
            }
        }

        pdb_hnsw_cand_t curr_cand = candidates[closest_idx];
        candidates[closest_idx] = candidates[--cand_count];

        if (curr_cand.dist > top_set[top_count - 1].dist && top_count >= ef) {
            break;
        }

        const uint32_t* neighbors;
        uint32_t count;
        pdb_hnsw_get_node_links(graph, curr_cand.id, 0, &neighbors, &count);

        for (uint32_t i = 0; i < count; i++) {
            uint32_t neighbor = neighbors[i];
            uint32_t word_idx = (neighbor / 64) % 2048;
            uint64_t bit = 1ULL << (neighbor % 64);

            if (visited[word_idx] & bit) continue;
            visited[word_idx] |= bit;

            /* Check tombstone bitset if provided */
            if (tombstone_mask && (tombstone_mask[neighbor / 64] & (1ULL << (neighbor % 64)))) {
                continue;
            }

            /* Direct AVX2 SIMD computation across memory-mapped vector */
            float d = pdb_hnsw_compute_distance(metric, query_vector, pdb_hnsw_get_node_vector(graph, neighbor), dim);

            if (top_count < ef || d < top_set[top_count - 1].dist) {
                if (cand_count < PDB_HNSW_MAX_EF) {
                    candidates[cand_count++] = (pdb_hnsw_cand_t){neighbor, d};
                }

                uint32_t pos = (top_count < ef) ? top_count++ : ef - 1;
                while (pos > 0 && top_set[pos - 1].dist > d) {
                    top_set[pos] = top_set[pos - 1];
                    pos--;
                }
                top_set[pos] = (pdb_hnsw_cand_t){neighbor, d};
            }
        }
    }

    uint32_t k = (top_k < top_count) ? top_k : top_count;
    for (uint32_t i = 0; i < k; i++) {
        const uint8_t* rec = pdb_hnsw_get_node_record(graph, top_set[i].id);
        out_ids[i] = *(const uint64_t*)rec;
        out_distances[i] = top_set[i].dist;
    }

    if (out_actual_k) *out_actual_k = k;
    return 0;
}
