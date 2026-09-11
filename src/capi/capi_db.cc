#include "c_api.h"
#include "c_version.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "palloc_compat.h"
#include "capi_utils.h"
#include "pin_manager.h"
#include "options.h"
#include "pomai.h"
#include "version.h"

namespace {

constexpr const char* kDefaultMembrane = "__default__";

struct RecordWrapper {
    pomai_record_t pub{};
    std::vector<float> vec_data;
    std::vector<uint8_t> meta_data;
};

struct SearchResultsWrapper {
    pomai_search_results_t pub{};
    std::vector<uint64_t> ids;
    std::vector<float> scores;
    std::vector<uint32_t> shard_ids;
};

constexpr uint32_t MinOptionsStructSize() {
    return static_cast<uint32_t>(offsetof(pomai_options_t, hnsw_ef_search) + sizeof(uint32_t));
}

constexpr uint32_t MinUpsertStructSize() {
    return static_cast<uint32_t>(offsetof(pomai_upsert_t, metadata_len) + sizeof(uint32_t));
}

constexpr uint32_t MinQueryStructSize() {
    return static_cast<uint32_t>(offsetof(pomai_query_t, filter_expression) + sizeof(const char*));
}

bool DeadlineExceeded(uint32_t deadline_ms) {
    if (deadline_ms == 0) {
        return false;
    }
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    return now_ms.count() >= deadline_ms;
}

bool ParseTenantFilter(const char* expr, pomai::SearchOptions* out_opts) {
    if (expr == nullptr || *expr == '\0') {
        return true;
    }

    std::string s(expr);
    const auto eq = s.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    auto field = s.substr(0, eq);
    auto value = s.substr(eq + 1);
    auto trim = [](std::string* v) {
        while (!v->empty() && std::isspace(static_cast<unsigned char>(v->front()))) v->erase(v->begin());
        while (!v->empty() && std::isspace(static_cast<unsigned char>(v->back()))) v->pop_back();
    };
    trim(&field);
    trim(&value);
    if (field != "tenant" && field != "device_id" && field != "location_id") {
        return false;
    }
    out_opts->filters.push_back(pomai::Filter(field, value));
    return true;
}

std::string JsonEscape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        const unsigned char uc = static_cast<unsigned char>(c);
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (uc < 0x20) {
                    out += "\\u00";
                    const char* hex = "0123456789abcdef";
                    out.push_back(hex[(uc >> 4) & 0x0f]);
                    out.push_back(hex[uc & 0x0f]);
                } else {
                    out.push_back(c);
                }
                break;
        }
    }
    return out;
}

pomai::Metadata ToMetadata(const pomai_upsert_t& item) {
    if (item.metadata == nullptr || item.metadata_len == 0) {
        return pomai::Metadata();
    }
    pomai::Metadata m(std::string(reinterpret_cast<const char*>(item.metadata), item.metadata_len));
    m.device_id = m.tenant;
    m.location_id = m.tenant;
    return m;
}

/**
 * @brief Sink that collects hits directly for the C ABI, avoiding metadata overhead.
 */
class CApiHitSink final : public pomai::SearchHitSink {
public:
    explicit CApiHitSink(uint32_t capacity) {
        ids_.reserve(capacity);
        scores_.reserve(capacity);
    }

    void Push(pomai::VectorId id, float score) override {
        ids_.push_back(id);
        scores_.push_back(score);
    }

    void Clear() {
        ids_.clear();
        scores_.clear();
    }

    size_t Size() const {
        return ids_.size();
    }

    std::vector<uint64_t> ids_;
    std::vector<float> scores_;
};

} // namespace

extern "C" {

void pomai_options_init(pomai_options_t* opts) {
    if (opts == nullptr) {
        return;
    }
    std::memset(opts, 0, sizeof(pomai_options_t));
    opts->struct_size = sizeof(pomai_options_t);
    opts->shards = 1;
    opts->dim = 128;
    opts->search_threads = 0;
    opts->fsync_policy = POMAI_FSYNC_POLICY_NEVER;
    opts->memory_budget_bytes = 64ULL * 1024 * 1024;
    opts->deadline_ms = 0;

    opts->index_type = 1; // Native HNSW
    opts->hnsw_m = 16;
    opts->hnsw_ef_construction = 200;
    opts->hnsw_ef_search = 50;
    opts->adaptive_threshold = 50000;
    opts->metric = 0; // L2
    opts->edge_profile = 0;
    opts->tick_max_ops = 64;
    opts->tick_max_ms = 10;
    opts->strict_deterministic = false;
}

void pomai_scan_options_init(pomai_scan_options_t* opts) {
    if (opts == nullptr) {
        return;
    }
    std::memset(opts, 0, sizeof(pomai_scan_options_t));
    opts->struct_size = sizeof(pomai_scan_options_t);
    opts->start_id = 0;
    opts->has_start_id = false;
    opts->deadline_ms = 0;
}

void pomai_options_apply_preset(pomai_options_t* opts, pomai_embedded_preset_t preset) {
    if (opts == nullptr) return;

    switch (preset) {
        case POMAI_EMBEDDED_PRESET_ESP32_S3:
            opts->shards = 1;
            opts->search_threads = 1;
            opts->fsync_policy = POMAI_FSYNC_POLICY_NEVER;
            opts->memory_budget_bytes = 4ULL * 1024 * 1024; // 4MB PSRAM limit
            opts->index_type = 1; // HNSW
            opts->hnsw_m = 8;
            opts->hnsw_ef_construction = 40;
            opts->hnsw_ef_search = 16;
            opts->edge_profile = 1; // low_ram
            opts->tick_max_ops = 16;
            opts->tick_max_ms = 5;
            break;

        case POMAI_EMBEDDED_PRESET_ARM_CORTEX_M85:
            opts->shards = 1;
            opts->search_threads = 1;
            opts->fsync_policy = POMAI_FSYNC_POLICY_NEVER;
            opts->memory_budget_bytes = 2ULL * 1024 * 1024; // 2MB SRAM
            opts->index_type = 1;
            opts->hnsw_m = 12;
            opts->hnsw_ef_construction = 64;
            opts->hnsw_ef_search = 24;
            opts->edge_profile = 1; // low_ram
            opts->tick_max_ops = 32;
            opts->tick_max_ms = 5;
            break;

        case POMAI_EMBEDDED_PRESET_RPI_ZERO_2W:
            opts->shards = 2;
            opts->search_threads = 4;
            opts->fsync_policy = POMAI_FSYNC_POLICY_NEVER;
            opts->memory_budget_bytes = 256ULL * 1024 * 1024; // 256MB
            opts->index_type = 1;
            opts->hnsw_m = 16;
            opts->hnsw_ef_construction = 128;
            opts->hnsw_ef_search = 48;
            opts->edge_profile = 2; // balanced
            opts->tick_max_ops = 128;
            opts->tick_max_ms = 20;
            break;

        case POMAI_EMBEDDED_PRESET_GENERIC:
        default:
            opts->shards = 1;
            opts->search_threads = 0;
            opts->fsync_policy = POMAI_FSYNC_POLICY_NEVER;
            opts->memory_budget_bytes = 64ULL * 1024 * 1024;
            opts->index_type = 1;
            opts->hnsw_m = 16;
            opts->hnsw_ef_construction = 200;
            opts->hnsw_ef_search = 50;
            opts->edge_profile = 0;
            opts->tick_max_ops = 64;
            opts->tick_max_ms = 10;
            break;
    }
}

pomai_status_t* pomai_options_resolve_json(const pomai_options_t* opts, char** out_json, size_t* out_len) {
    if (opts == nullptr || out_json == nullptr || out_len == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid resolve arguments");
    }
    pomai::DBOptions db_opts;
    db_opts.path = opts->path ? opts->path : "";
    db_opts.shard_count = opts->shards;
    db_opts.dim = opts->dim;
    db_opts.edge_profile = static_cast<pomai::EdgeProfile>(opts->edge_profile);
    db_opts.ApplyEdgeProfile();

    std::string json = "{";
    json += "\"path\":\"" + JsonEscape(db_opts.path) + "\",";
    json += "\"dim\":" + std::to_string(db_opts.dim) + ",";
    json += "\"shards\":" + std::to_string(db_opts.shard_count) + ",";
    json += "\"edge_profile\":" + std::to_string(static_cast<uint8_t>(db_opts.edge_profile));
    json += "}";

    char* p = static_cast<char*>(palloc_malloc_aligned(json.size() + 1, alignof(char)));
    if (!p) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "allocation failed");
    std::memcpy(p, json.data(), json.size());
    p[json.size()] = '\0';
    *out_json = p;
    *out_len = json.size();
    return nullptr;
}

pomai_status_t* pomai_open(const pomai_options_t* opts, pomai_db_t** out_db) {
    if (opts == nullptr || out_db == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "opts and out_db must be non-null");
    }
    if (opts->struct_size < MinOptionsStructSize()) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "opts.struct_size is too small");
    }
    if (opts->path == nullptr || opts->path[0] == '\0') {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "opts.path must be non-empty");
    }
    if (opts->dim == 0) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "opts.dim must be > 0");
    }

    pomai::DBOptions db_opts;
    db_opts.path = opts->path;
    db_opts.shard_count = opts->shards > 0 ? opts->shards : 1;
    db_opts.dim = opts->dim;
    db_opts.fsync = (opts->fsync_policy == POMAI_FSYNC_POLICY_ALWAYS)
                        ? pomai::FsyncPolicy::kAlways
                        : pomai::FsyncPolicy::kNever;

    if (opts->index_type == 1) {
        db_opts.index_params.type = pomai::IndexType::kHnsw;
        db_opts.index_params.hnsw_m = opts->hnsw_m > 0 ? opts->hnsw_m : 16;
        db_opts.index_params.hnsw_ef_construction = opts->hnsw_ef_construction > 0 ? opts->hnsw_ef_construction : 200;
        db_opts.index_params.hnsw_ef_search = opts->hnsw_ef_search > 0 ? opts->hnsw_ef_search : 50;
    } else {
        db_opts.index_params.type = pomai::IndexType::kIvfFlat;
    }

    db_opts.metric = (opts->metric == 1) ? pomai::MetricType::kInnerProduct : pomai::MetricType::kL2;
    db_opts.edge_profile = static_cast<pomai::EdgeProfile>(opts->edge_profile);
    db_opts.ApplyEdgeProfile();

    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(db_opts, &db);
    if (!st.ok()) {
        return ToCStatus(st);
    }

    void* raw = palloc_malloc_aligned(sizeof(pomai_db_t), alignof(pomai_db_t));
    if (!raw) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "db handle allocation failed");
    *out_db = new (raw) pomai_db_t{std::move(db)};
    return nullptr;
}

pomai_status_t* pomai_close(pomai_db_t* db) {
    if (db == nullptr) {
        return nullptr;
    }
    auto st = db->db->Close();
    db->~pomai_db_t();
    palloc_free(db);
    return ToCStatus(st);
}

pomai_status_t* pomai_freeze(pomai_db_t* db) {
    if (db == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db must be non-null");
    }
    return ToCStatus(db->db->Freeze(kDefaultMembrane));
}

pomai_status_t* pomai_put(pomai_db_t* db, const pomai_upsert_t* item) {
    if (db == nullptr || item == nullptr || item->vector == nullptr || item->dim == 0) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid put arguments");
    }
    if (item->struct_size < MinUpsertStructSize()) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "upsert.struct_size is too small");
    }
    std::span<const float> vec(item->vector, item->dim);
    return ToCStatus(db->db->PutVector(item->id, vec, ToMetadata(*item)));
}

pomai_status_t* pomai_put_batch(pomai_db_t* db, const pomai_upsert_t* items, size_t n) {
    if (db == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db must be non-null");
    }
    if (n == 0) {
        return nullptr;
    }
    if (items == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "items must be non-null");
    }

    std::vector<pomai::VectorId> ids;
    std::vector<std::span<const float>> vecs;
    ids.reserve(n);
    vecs.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        if (items[i].struct_size < MinUpsertStructSize()) {
            return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "all batch items require valid struct_size");
        }
        if (items[i].vector == nullptr || items[i].dim == 0) {
            return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "all batch items require vector and dim");
        }
        ids.push_back(items[i].id);
        vecs.emplace_back(items[i].vector, items[i].dim);
    }
    return ToCStatus(db->db->PutBatch(ids, vecs));
}

pomai_status_t* pomai_delete(pomai_db_t* db, uint64_t id) {
    if (db == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db must be non-null");
    }
    return ToCStatus(db->db->Delete(id));
}

pomai_status_t* pomai_exists(pomai_db_t* db, uint64_t id, bool* out_exists) {
    if (db == nullptr || out_exists == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid exists arguments");
    }
    return ToCStatus(db->db->Exists(id, out_exists));
}

pomai_status_t* pomai_get(pomai_db_t* db, uint64_t id, pomai_record_t** out_record) {
    if (db == nullptr || out_record == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid get arguments");
    }

    void* raw = palloc_malloc_aligned(sizeof(RecordWrapper), alignof(RecordWrapper));
    if (!raw) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "record allocation failed");
    auto* wrapper = new (raw) RecordWrapper();

    pomai::Metadata meta;
    auto st = db->db->Get(id, &wrapper->vec_data, &meta);
    if (!st.ok()) {
        wrapper->~RecordWrapper();
        palloc_free(wrapper);
        return ToCStatus(st);
    }

    wrapper->pub.struct_size = sizeof(pomai_record_t);
    wrapper->pub.id = id;
    wrapper->pub.dim = static_cast<uint32_t>(wrapper->vec_data.size());
    wrapper->pub.vector = wrapper->vec_data.data();
    wrapper->pub.is_deleted = false;

    if (!meta.tenant.empty()) {
        wrapper->meta_data.assign(meta.tenant.begin(), meta.tenant.end());
        wrapper->pub.metadata = wrapper->meta_data.data();
        wrapper->pub.metadata_len = static_cast<uint32_t>(wrapper->meta_data.size());
    } else {
        wrapper->pub.metadata = nullptr;
        wrapper->pub.metadata_len = 0;
    }

    *out_record = &wrapper->pub;
    return nullptr;
}

void pomai_record_free(pomai_record_t* record) {
    if (record == nullptr) {
        return;
    }
    auto* wrapper = reinterpret_cast<RecordWrapper*>(record);
    wrapper->~RecordWrapper();
    palloc_free(wrapper);
}

pomai_status_t* pomai_search(pomai_db_t* db, const pomai_query_t* query, pomai_search_results_t** out) {
    if (db == nullptr || query == nullptr || out == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid search arguments");
    }
    if (query->struct_size < MinQueryStructSize()) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "query.struct_size is too small");
    }
    if (query->vector == nullptr || query->dim == 0) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "query vector and dim required");
    }
    if (DeadlineExceeded(query->deadline_ms)) {
        return MakeStatus(POMAI_STATUS_DEADLINE_EXCEEDED, "deadline exceeded before search execution");
    }

    pomai::SearchOptions opts;
    opts.zero_copy = ((query->flags & POMAI_QUERY_FLAG_ZERO_COPY) != 0);
    if (query->partition_device_id != nullptr && *query->partition_device_id != '\0') {
        opts.partition_device_id = query->partition_device_id;
    }
    if (query->partition_location_id != nullptr && *query->partition_location_id != '\0') {
        opts.partition_location_id = query->partition_location_id;
    }
    if (!ParseTenantFilter(query->filter_expression, &opts)) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "unsupported filter expression (supported: tenant/device_id/location_id=<val>)");
    }

    std::span<const float> q(query->vector, query->dim);

    if (!opts.zero_copy) {
        CApiHitSink sink(query->topk);
        auto st = db->db->SearchVector(q, query->topk, opts, sink);
        if (!st.ok()) {
            return ToCStatus(st);
        }

        void* raw = palloc_malloc_aligned(sizeof(SearchResultsWrapper), alignof(SearchResultsWrapper));
        if (!raw) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "search results allocation failed");
        auto* wrapper = new (raw) SearchResultsWrapper();

        wrapper->ids = std::move(sink.ids_);
        wrapper->scores = std::move(sink.scores_);

        wrapper->pub.struct_size = sizeof(pomai_search_results_t);
        wrapper->pub.count = wrapper->ids.size();
        wrapper->pub.ids = wrapper->ids.data();
        wrapper->pub.scores = wrapper->scores.data();
        wrapper->pub.total_shards_count = 1;
        wrapper->pub.pruned_shards_count = 0;
        wrapper->pub.zero_copy_pointers = nullptr;

        *out = &wrapper->pub;
        return nullptr;
    }

    pomai::SearchResult res;
    auto st = db->db->SearchVector(q, query->topk, opts, &res);
    if (!st.ok()) {
        return ToCStatus(st);
    }

    void* raw = palloc_malloc_aligned(sizeof(SearchResultsWrapper), alignof(SearchResultsWrapper));
    if (!raw) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "search results allocation failed");
    auto* wrapper = new (raw) SearchResultsWrapper();

    wrapper->ids.reserve(res.hits.size());
    wrapper->scores.reserve(res.hits.size());
    for (const auto& h : res.hits) {
        wrapper->ids.push_back(h.id);
        wrapper->scores.push_back(h.score);
    }

    wrapper->pub.struct_size = sizeof(pomai_search_results_t);
    wrapper->pub.count = wrapper->ids.size();
    wrapper->pub.ids = wrapper->ids.data();
    wrapper->pub.scores = wrapper->scores.data();
    wrapper->pub.total_shards_count = res.total_shards_count;
    wrapper->pub.pruned_shards_count = res.pruned_shards_count;

    if (!res.zero_copy_pointers.empty()) {
        wrapper->pub.zero_copy_pointers = static_cast<pomai_semantic_pointer_t*>(
            palloc_malloc_aligned(res.zero_copy_pointers.size() * sizeof(pomai_semantic_pointer_t), alignof(pomai_semantic_pointer_t)));
        for (size_t i = 0; i < res.zero_copy_pointers.size(); ++i) {
            wrapper->pub.zero_copy_pointers[i].struct_size = sizeof(pomai_semantic_pointer_t);
            wrapper->pub.zero_copy_pointers[i].raw_data_ptr = res.zero_copy_pointers[i].raw_data_ptr;
            wrapper->pub.zero_copy_pointers[i].dim = res.zero_copy_pointers[i].dim;
            wrapper->pub.zero_copy_pointers[i].quant_min = res.zero_copy_pointers[i].quant_min;
            wrapper->pub.zero_copy_pointers[i].quant_inv_scale = res.zero_copy_pointers[i].quant_inv_scale;
            wrapper->pub.zero_copy_pointers[i].session_id = res.zero_copy_pointers[i].session_id;
        }
    }

    *out = &wrapper->pub;
    return nullptr;
}

void pomai_search_results_free(pomai_search_results_t* results) {
    if (results == nullptr) {
        return;
    }
    if (results->zero_copy_pointers != nullptr) {
        palloc_free(results->zero_copy_pointers);
    }
    auto* wrapper = reinterpret_cast<SearchResultsWrapper*>(results);
    wrapper->~SearchResultsWrapper();
    palloc_free(wrapper);
}

pomai_status_t* pomai_search_batch(
    pomai_db_t* db, const pomai_query_t* queries, size_t num_queries,
    pomai_search_results_t** out_results) {
    if (db == nullptr || out_results == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db/out_results must be non-null");
    }
    if (num_queries == 0) {
        *out_results = nullptr;
        return nullptr;
    }
    if (queries == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "queries must be non-null");
    }

    const uint32_t dim = queries[0].dim;
    if (dim == 0) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "dim must be > 0");
    }

    std::vector<float> flat_queries;
    flat_queries.reserve(num_queries * dim);
    for (size_t i = 0; i < num_queries; ++i) {
        if (queries[i].dim != dim || queries[i].vector == nullptr) {
            return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "inconsistent batch query dim or null vector");
        }
        flat_queries.insert(flat_queries.end(), queries[i].vector, queries[i].vector + dim);
    }

    pomai::SearchOptions opts;
    opts.zero_copy = ((queries[0].flags & POMAI_QUERY_FLAG_ZERO_COPY) != 0);

    std::vector<pomai::SearchResult> batch_res;
    auto st = db->db->SearchBatch(flat_queries, static_cast<uint32_t>(num_queries), queries[0].topk, opts, &batch_res);
    if (!st.ok()) return ToCStatus(st);

    pomai_search_results_t* arr = static_cast<pomai_search_results_t*>(
        palloc_malloc_aligned(num_queries * sizeof(pomai_search_results_t), alignof(pomai_search_results_t)));
    if (!arr) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "batch results allocation failed");

    for (size_t i = 0; i < num_queries; ++i) {
        const auto& r = batch_res[i];
        arr[i].struct_size = sizeof(pomai_search_results_t);
        arr[i].count = r.hits.size();
        arr[i].ids = nullptr;
        arr[i].scores = nullptr;
        arr[i].shard_ids = nullptr;
        arr[i].total_shards_count = r.total_shards_count;
        arr[i].pruned_shards_count = r.pruned_shards_count;
        arr[i].zero_copy_pointers = nullptr;

        if (!r.hits.empty()) {
            arr[i].ids = static_cast<uint64_t*>(palloc_malloc_aligned(r.hits.size() * sizeof(uint64_t), alignof(uint64_t)));
            arr[i].scores = static_cast<float*>(palloc_malloc_aligned(r.hits.size() * sizeof(float), alignof(float)));
            for (size_t j = 0; j < r.hits.size(); ++j) {
                arr[i].ids[j] = r.hits[j].id;
                arr[i].scores[j] = r.hits[j].score;
            }
        }
    }

    *out_results = arr;
    return nullptr;
}

void pomai_search_batch_free(pomai_search_results_t* results, size_t num_queries) {
    if (!results) return;
    for (size_t i = 0; i < num_queries; ++i) {
        palloc_free(results[i].ids);
        palloc_free(results[i].scores);
        palloc_free(results[i].shard_ids);
        palloc_free(results[i].zero_copy_pointers);
    }
    palloc_free(results);
}

pomai_status_t* pomai_create_membrane_kind(pomai_db_t* db, const char* name, uint32_t dim, uint32_t shard_count, uint32_t kind) {
    (void)kind;
    if (db == nullptr || name == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db/name must be non-null");
    }
    pomai::MembraneSpec spec;
    spec.name = name;
    spec.dim = dim;
    spec.shard_count = shard_count > 0 ? shard_count : 1u;
    spec.kind = pomai::MembraneKind::kVector;
    auto st = db->db->CreateMembrane(spec);
    if (!st.ok()) return ToCStatus(st);
    return ToCStatus(db->db->OpenMembrane(name));
}

pomai_status_t* pomai_list_membranes_json(pomai_db_t* db, char** out_json, size_t* out_len) {
    if (db == nullptr || out_json == nullptr || out_len == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    }
    std::vector<std::string> membranes;
    auto st = db->db->ListMembranes(&membranes);
    if (!st.ok()) return ToCStatus(st);
    std::string json = "[";
    for (size_t i = 0; i < membranes.size(); ++i) {
        if (i > 0) json += ",";
        json += "\"" + JsonEscape(membranes[i]) + "\"";
    }
    json += "]";
    char* p = static_cast<char*>(palloc_malloc_aligned(json.size() + 1, alignof(char)));
    if (!p) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "allocation failed");
    std::memcpy(p, json.data(), json.size());
    p[json.size()] = '\0';
    *out_json = p;
    *out_len = json.size();
    return nullptr;
}

pomai_status_t* pomai_compact_membrane(pomai_db_t* db, const char* membrane_name) {
    if (db == nullptr || membrane_name == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    }
    return ToCStatus(db->db->Compact(membrane_name));
}

void pomai_release_pointer(uint64_t session_id) {
    pomai::core::MemoryPinManager::Instance().Unpin(session_id);
}

void pomai_free(void* p) {
    palloc_free(p);
}

POMAI_API uint32_t pomai_abi_version(void) {
    return POMAI_C_ABI_VERSION;
}

POMAI_API const char* pomai_version_string(void) {
    return "PomaiDB 1.1.0";
}


} // extern "C"
