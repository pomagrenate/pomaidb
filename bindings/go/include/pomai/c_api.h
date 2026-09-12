#ifndef POMAI_C_API_H
#define POMAI_C_API_H

#include "c_types.h"
#include "c_status.h"
#include "c_version.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options
POMAI_API void pomai_options_init(pomai_options_t* opts);
POMAI_API void pomai_scan_options_init(pomai_scan_options_t* opts);
// Resolve effective runtime options after applying edge profile.
POMAI_API pomai_status_t* pomai_options_resolve_json(const pomai_options_t* opts, char** out_json, size_t* out_len);
/** Apply a hardware-specific configuration preset to the options. */
POMAI_API void pomai_options_apply_preset(pomai_options_t* opts, pomai_embedded_preset_t preset);

// Database Management & Lifecycle
POMAI_API pomai_status_t* pomai_open(const pomai_options_t* opts, pomai_db_t** out_db);
POMAI_API pomai_status_t* pomai_close(pomai_db_t* db);
POMAI_API pomai_status_t* pomai_flush(pomai_db_t* db);
POMAI_API pomai_status_t* pomai_freeze(pomai_db_t* db);
POMAI_API pomai_status_t* pomai_freeze_membrane(pomai_db_t* db, const char* membrane);
POMAI_API pomai_status_t* pomai_compact(pomai_db_t* db);
POMAI_API pomai_status_t* pomai_compact_membrane(pomai_db_t* db, const char* membrane_name);
POMAI_API pomai_status_t* pomai_get_stats_json(pomai_db_t* db, char** out_json, size_t* out_len);

// Mutation (Default membrane or item->membrane if specified)
POMAI_API pomai_status_t* pomai_put(pomai_db_t* db, const pomai_upsert_t* item);
POMAI_API pomai_status_t* pomai_put_batch(pomai_db_t* db, const pomai_upsert_t* items, size_t n);
POMAI_API pomai_status_t* pomai_delete(pomai_db_t* db, uint64_t id);

// Point Query (Default membrane)
POMAI_API pomai_status_t* pomai_exists(pomai_db_t* db, uint64_t id, bool* out_exists);
POMAI_API pomai_status_t* pomai_get(pomai_db_t* db, uint64_t id, pomai_record_t** out_record);
POMAI_API void pomai_record_free(pomai_record_t* record);

// Search (Default membrane or query->membrane if specified)
POMAI_API pomai_status_t* pomai_search(pomai_db_t* db, const pomai_query_t* query, pomai_search_results_t** out);
POMAI_API void pomai_search_results_free(pomai_search_results_t* results);
POMAI_API pomai_status_t* pomai_search_batch(
    pomai_db_t* db, const pomai_query_t* queries, size_t num_queries,
    pomai_search_results_t** out_results);
POMAI_API void pomai_search_batch_free(pomai_search_results_t* results, size_t num_queries);

// Named Vector Collections (Membranes) - Lifecycle & Retention
POMAI_API pomai_status_t* pomai_create_membrane_kind(pomai_db_t* db, const char* name, uint32_t dim, uint32_t shard_count, uint32_t kind);
POMAI_API pomai_status_t* pomai_drop_membrane(pomai_db_t* db, const char* membrane_name);
POMAI_API pomai_status_t* pomai_open_membrane(pomai_db_t* db, const char* membrane_name);
POMAI_API pomai_status_t* pomai_close_membrane(pomai_db_t* db, const char* membrane_name);
POMAI_API pomai_status_t* pomai_list_membranes_json(pomai_db_t* db, char** out_json, size_t* out_len);
POMAI_API pomai_status_t* pomai_update_membrane_retention(
    pomai_db_t* db, const char* membrane_name,
    uint32_t ttl_sec, uint32_t retention_max_count, uint64_t retention_max_bytes);
POMAI_API pomai_status_t* pomai_get_membrane_retention(
    pomai_db_t* db, const char* membrane_name,
    uint32_t* ttl_sec, uint32_t* retention_max_count, uint64_t* retention_max_bytes);

// Explicit Named Membrane CRUD
POMAI_API pomai_status_t* pomai_put_membrane(pomai_db_t* db, const char* membrane, const pomai_upsert_t* item);
POMAI_API pomai_status_t* pomai_put_batch_membrane(pomai_db_t* db, const char* membrane, const pomai_upsert_t* items, size_t n);
POMAI_API pomai_status_t* pomai_get_membrane(pomai_db_t* db, const char* membrane, uint64_t id, pomai_record_t** out_record);
POMAI_API pomai_status_t* pomai_exists_membrane(pomai_db_t* db, const char* membrane, uint64_t id, bool* out_exists);
POMAI_API pomai_status_t* pomai_delete_membrane(pomai_db_t* db, const char* membrane, uint64_t id);
POMAI_API pomai_status_t* pomai_search_membrane(pomai_db_t* db, const char* membrane, const pomai_query_t* query, pomai_search_results_t** out);
POMAI_API pomai_status_t* pomai_get_snapshot_membrane(pomai_db_t* db, const char* membrane, pomai_snapshot_t** out_snap);

// Snapshot & Scan
POMAI_API pomai_status_t* pomai_get_snapshot(pomai_db_t* db, pomai_snapshot_t** out_snap);
POMAI_API void pomai_snapshot_free(pomai_snapshot_t* snap);

POMAI_API pomai_status_t* pomai_scan(
    pomai_db_t* db,
    const pomai_scan_options_t* opts,
    const pomai_snapshot_t* snap,
    pomai_iter_t** out_iter);

POMAI_API bool pomai_iter_valid(const pomai_iter_t* iter);
POMAI_API void pomai_iter_next(pomai_iter_t* iter);
POMAI_API pomai_status_t* pomai_iter_status(const pomai_iter_t* iter);
POMAI_API pomai_status_t* pomai_iter_get_record(const pomai_iter_t* iter, pomai_record_view_t* out_view);
POMAI_API void pomai_iter_free(pomai_iter_t* iter);

// Utils
/** Release a zero-copy semantic pointer session (see POMAI_QUERY_FLAG_ZERO_COPY). */
POMAI_API void pomai_release_pointer(uint64_t session_id);
POMAI_API void pomai_free(void* p);

#ifdef __cplusplus
}
#endif

#endif // POMAI_C_API_H
