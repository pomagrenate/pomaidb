#include "pomaidb_internal.h"

pdb_status_t pdb_options_init(pdb_options_t* options, size_t dim) {
    if (!options || dim == 0) return PDB_ERR_INVALID_ARGUMENT;
    memset(options, 0, sizeof(pdb_options_t));
    options->dim = dim;
    options->metric = PDB_METRIC_L2;
    options->memtable_capacity = PDB_DEFAULT_MEMTABLE_CAPACITY;
    options->wal_directory = NULL;
    options->enable_direct_io = false;
    options->arena_reserve_bytes = 0; /* 0 triggers adaptive default */
    return PDB_SUCCESS;
}

static pdb_status_t pdb_memtable_init(pdb_memtable_t* m, size_t capacity, size_t dim, size_t reserve_bytes) {
    memset(m, 0, sizeof(pdb_memtable_t));
    m->capacity = capacity;
    m->count = 0;

    size_t vec_bytes = capacity * dim * sizeof(float);
    size_t id_bytes  = capacity * sizeof(uint64_t);
    size_t num_words = (capacity + 63) / 64;
    size_t tomb_bytes = num_words * sizeof(uint64_t);

    size_t total_arena_bytes = reserve_bytes > 0 ? reserve_bytes : (vec_bytes + id_bytes + tomb_bytes + 1048576);

    /* 1. Allocate volatile vector memory via palloc vector arena with 64-byte AVX-512 alignment */
    m->arena = pa_vec_arena_create(total_arena_bytes, PDB_VECTOR_ALIGNMENT);
    if (!m->arena) return PDB_ERR_OUT_OF_MEMORY;

    /* 2. Carve flat contiguous SIMD vector payload */
    m->vectors = (float*)pa_vec_arena_alloc(m->arena, vec_bytes);
    if (!m->vectors) {
        pa_vec_arena_destroy(m->arena);
        return PDB_ERR_OUT_OF_MEMORY;
    }

    /* 3. Carve ID array */
    m->ids = (uint64_t*)pa_vec_arena_alloc(m->arena, id_bytes);
    if (!m->ids) {
        pa_vec_arena_destroy(m->arena);
        return PDB_ERR_OUT_OF_MEMORY;
    }

    /* 4. Carve tombstone bitmap */
    m->tombstones = (uint64_t*)pa_vec_arena_alloc(m->arena, tomb_bytes);
    if (!m->tombstones) {
        pa_vec_arena_destroy(m->arena);
        return PDB_ERR_OUT_OF_MEMORY;
    }
    memset(m->tombstones, 0, tomb_bytes);

    /* 5. Initialize dense 64-bit word metadata bitmap pool */
    m->metadata_pool = pa_vec_pool_create(sizeof(uint64_t) * 8, 256);

    return PDB_SUCCESS;
}

static void pdb_memtable_destroy(pdb_memtable_t* m) {
    if (!m) return;
    if (m->metadata_pool) {
        pa_vec_pool_destroy(m->metadata_pool);
        m->metadata_pool = NULL;
    }
    if (m->arena) {
        pa_vec_arena_destroy(m->arena);
        m->arena = NULL;
    }
    m->vectors = NULL;
    m->ids = NULL;
    m->tombstones = NULL;
    m->count = 0;
}

pdb_status_t pdb_open(const char* storage_path, const pdb_options_t* options, pdb_t** out_db) {
    if (!options || options->dim == 0 || !out_db) return PDB_ERR_INVALID_ARGUMENT;

    pdb_t* db = (pdb_t*)pa_malloc(sizeof(pdb_t));
    if (!db) return PDB_ERR_OUT_OF_MEMORY;
    memset(db, 0, sizeof(pdb_t));

    db->options = *options;
    if (db->options.memtable_capacity == 0) {
        db->options.memtable_capacity = PDB_DEFAULT_MEMTABLE_CAPACITY;
    }

    if (storage_path) {
        snprintf(db->storage_path, sizeof(db->storage_path), "%s", storage_path);
    }

    /* Initialize volatile MemTable */
    pdb_status_t status = pdb_memtable_init(
        &db->memtable,
        db->options.memtable_capacity,
        db->options.dim,
        db->options.arena_reserve_bytes
    );
    if (status != PDB_SUCCESS) {
        pa_free(db);
        return status;
    }

    /* Initialize Durability WAL */
    status = pdb_wal_init(&db->wal, db->options.wal_directory, db->options.enable_direct_io);
    if (status != PDB_SUCCESS) {
        pdb_memtable_destroy(&db->memtable);
        pa_free(db);
        return status;
    }

    /* Run Crash Recovery Replay */
    status = pdb_wal_recover(&db->wal, db);
    if (status != PDB_SUCCESS) {
        pdb_wal_close(&db->wal);
        pdb_memtable_destroy(&db->memtable);
        pa_free(db);
        return status;
    }

    *out_db = db;
    return PDB_SUCCESS;
}

pdb_status_t pdb_insert_batch(pdb_t* db, const pdb_vector_batch_t* batch) {
    if (!db || !batch || !batch->ids || !batch->vectors || batch->count == 0) {
        return PDB_ERR_INVALID_ARGUMENT;
    }
    if (batch->dim != db->options.dim) {
        return PDB_ERR_INVALID_ARGUMENT;
    }

    pdb_memtable_t* m = &db->memtable;

    /* If batch exceeds remaining MemTable capacity, flush checkpoint first */
    if (m->count + batch->count > m->capacity) {
        pdb_status_t flush_st = pdb_checkpoint(db);
        if (flush_st != PDB_SUCCESS) return flush_st;
    }

    /* Write to WAL before touching memory */
    uint64_t assigned_lsn = 0;
    pdb_status_t wal_st = pdb_wal_append(&db->wal, batch, &assigned_lsn);
    if (wal_st != PDB_SUCCESS) return wal_st;

    /* Ingest into 64-byte aligned SIMD MemTable */
    size_t dim = db->options.dim;
    memcpy(m->ids + m->count, batch->ids, batch->count * sizeof(uint64_t));
    memcpy(m->vectors + (m->count * dim), batch->vectors, batch->count * dim * sizeof(float));
    m->count += batch->count;
    db->total_vectors += batch->count;
    db->current_lsn = assigned_lsn;

    return PDB_SUCCESS;
}

pdb_status_t pdb_delete_batch(pdb_t* db, const uint64_t* ids, size_t count) {
    if (!db || !ids || count == 0) return PDB_ERR_INVALID_ARGUMENT;

    /* Mark tombstones in active MemTable */
    pdb_memtable_t* m = &db->memtable;
    for (size_t i = 0; i < count; i++) {
        uint64_t target_id = ids[i];
        for (size_t k = 0; k < m->count; k++) {
            if (m->ids[k] == target_id) {
                m->tombstones[k / 64] |= (1ULL << (k % 64));
                break;
            }
        }
    }

    /* Mark tombstones across sealed immutable segments */
    for (size_t s = 0; s < db->segment_count; s++) {
        pdb_segment_t* seg = db->segments[s];
        if (!seg || !seg->tombstones) continue;
        for (size_t i = 0; i < count; i++) {
            uint64_t target_id = ids[i];
            for (size_t k = 0; k < seg->count; k++) {
                if (seg->ids[k] == target_id) {
                    seg->tombstones[k / 64] |= (1ULL << (k % 64));
                    break;
                }
            }
        }
    }

    return PDB_SUCCESS;
}

pdb_status_t pdb_query_knn(
    pdb_t* db,
    const float* query_vector,
    size_t top_k,
    const uint64_t* filter_mask,
    uint64_t* out_ids,
    float* out_distances,
    size_t* out_actual_k
) {
    if (!db || !query_vector || top_k == 0 || !out_ids || !out_distances) {
        return PDB_ERR_INVALID_ARGUMENT;
    }

    size_t dim = db->options.dim;
    pdb_metric_t metric = db->options.metric;

    pdb_candidate_t* heap_buf = (pdb_candidate_t*)pa_malloc(top_k * sizeof(pdb_candidate_t));
    if (!heap_buf) return PDB_ERR_OUT_OF_MEMORY;

    pdb_heap_t heap;
    heap.capacity = top_k;
    heap.size = 0;
    heap.data = heap_buf;

    /* 1. Scan sealed immutable on-disk segments (zero-copy from mmap page cache) */
    for (size_t s = 0; s < db->segment_count; s++) {
        const pdb_segment_t* seg = db->segments[s];
        if (!seg) continue;

        size_t seg_words = (seg->count + 63) / 64;
        for (size_t w = 0; w < seg_words; w++) {
            uint64_t active_mask = filter_mask ? filter_mask[w] : ~0ULL;
            if (seg->tombstones) {
                active_mask &= ~seg->tombstones[w];
            }

            /* Single-cycle hardware early skip */
            if (active_mask == 0ULL) continue;

            while (active_mask != 0ULL) {
                int bit = pdb_tzcnt_u64(active_mask);
                size_t idx = w * 64 + (size_t)bit;

                if (idx < seg->count) {
                    const float* target = seg->vectors + (idx * dim);
                    float dist = pdb_compute_distance(metric, query_vector, target, dim);
                    pdb_heap_push(&heap, seg->ids[idx], dist);
                }
                active_mask &= (active_mask - 1ULL);
            }
        }
    }

    /* 2. Scan volatile in-memory MemTable */
    const pdb_memtable_t* m = &db->memtable;
    size_t mem_words = (m->count + 63) / 64;
    for (size_t w = 0; w < mem_words; w++) {
        uint64_t active_mask = filter_mask ? filter_mask[w] : ~0ULL;
        if (m->tombstones) {
            active_mask &= ~m->tombstones[w];
        }

        if (active_mask == 0ULL) continue;

        while (active_mask != 0ULL) {
            int bit = pdb_tzcnt_u64(active_mask);
            size_t idx = w * 64 + (size_t)bit;

            if (idx < m->count) {
                const float* target = m->vectors + (idx * dim);
                float dist = pdb_compute_distance(metric, query_vector, target, dim);
                pdb_heap_push(&heap, m->ids[idx], dist);
            }
            active_mask &= (active_mask - 1ULL);
        }
    }

    /* Drain heap into sorted output arrays */
    size_t actual = heap.size;
    for (size_t i = heap.size; i > 0; i--) {
        pdb_candidate_t c = heap.data[0];
        heap.data[0] = heap.data[i - 1];
        heap.size--;
        pdb_heap_sift_down(&heap, 0);
        out_ids[i - 1] = c.id;
        out_distances[i - 1] = c.distance;
    }

    if (out_actual_k) *out_actual_k = actual;
    pa_free(heap_buf);
    return PDB_SUCCESS;
}

pdb_status_t pdb_checkpoint(pdb_t* db) {
    if (!db) return PDB_ERR_INVALID_ARGUMENT;
    if (db->memtable.count == 0) return PDB_SUCCESS;

    /* If storage_path is specified, flush to disk segment */
    if (db->storage_path[0] != '\0' && db->segment_count < PDB_MAX_SEGMENTS) {
        char seg_path[512];
        snprintf(seg_path, sizeof(seg_path), "%s/seg_%06zu.pvs", db->storage_path, db->segment_count);

        pdb_status_t flush_st = pdb_segment_flush(db, &db->memtable, seg_path);
        if (flush_st == PDB_SUCCESS) {
            pdb_segment_t* seg = NULL;
            if (pdb_segment_open(seg_path, &seg) == PDB_SUCCESS) {
                db->segments[db->segment_count++] = seg;
            }
        }
    }

    /* Fast O(1) zero-syscall rewind of palloc vector arena using pa_vec_arena_clear() */
    pa_vec_arena_clear(db->memtable.arena);

    size_t cap = db->memtable.capacity;
    size_t dim = db->options.dim;
    size_t vec_bytes = cap * dim * sizeof(float);
    size_t id_bytes  = cap * sizeof(uint64_t);
    size_t num_words = (cap + 63) / 64;
    size_t tomb_bytes = num_words * sizeof(uint64_t);

    db->memtable.vectors    = (float*)pa_vec_arena_alloc(db->memtable.arena, vec_bytes);
    db->memtable.ids        = (uint64_t*)pa_vec_arena_alloc(db->memtable.arena, id_bytes);
    db->memtable.tombstones = (uint64_t*)pa_vec_arena_alloc(db->memtable.arena, tomb_bytes);
    if (db->memtable.tombstones) {
        memset(db->memtable.tombstones, 0, tomb_bytes);
    }
    db->memtable.count = 0;

    return PDB_SUCCESS;
}

pdb_status_t pdb_get_stats(pdb_t* db, pdb_stats_t* out_stats) {
    if (!db || !out_stats) return PDB_ERR_INVALID_ARGUMENT;
    memset(out_stats, 0, sizeof(pdb_stats_t));

    out_stats->total_vectors = db->total_vectors;
    out_stats->memtable_vectors = db->memtable.count;
    out_stats->sealed_segments = db->segment_count;
    out_stats->current_lsn = db->current_lsn;

    if (db->memtable.arena) {
        pa_vec_arena_stats_t ast;
        memset(&ast, 0, sizeof(ast));
        pa_vec_arena_get_stats(db->memtable.arena, &ast);
        out_stats->arena_allocated_bytes = ast.used_bytes;
        out_stats->arena_committed_bytes = ast.committed_bytes;
    }
    return PDB_SUCCESS;
}

pdb_status_t pdb_close(pdb_t* db) {
    if (!db) return PDB_SUCCESS;

    /* Close all active segments */
    for (size_t i = 0; i < db->segment_count; i++) {
        if (db->segments[i]) {
            pdb_segment_close(db->segments[i]);
            db->segments[i] = NULL;
        }
    }
    db->segment_count = 0;

    /* Close WAL */
    pdb_wal_close(&db->wal);

    /* Destroy volatile MemTable arena */
    pdb_memtable_destroy(&db->memtable);

    pa_free(db);
    return PDB_SUCCESS;
}
