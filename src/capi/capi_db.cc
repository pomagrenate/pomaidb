#include "c_api.h"
#include "c_version.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "palloc_compat.h"
#include "capi_utils.h"
#include "pin_manager.h"
#include "options.h"
#include "pomai.h"
#include "version.h"

namespace {

std::mutex g_handles_mutex;
std::unordered_set<const pomai_db_t*> g_active_handles;

bool IsValidHandle(const pomai_db_t* db) {
    if (db == nullptr) return false;
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    return g_active_handles.find(db) != g_active_handles.end();
}

void RegisterHandle(const pomai_db_t* db) {
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    g_active_handles.insert(db);
}

bool UnregisterHandle(const pomai_db_t* db) {
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    return g_active_handles.erase(db) > 0;
}

constexpr const char* kDefaultMembrane = "__default__";

struct RecordWrapper {
    pomai_record_t pub{};
    std::vector<float> vec_data;
    std::vector<uint8_t> meta_data;
    std::vector<uint8_t> payload_data;
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
    pomai::Metadata m;
    if (item.metadata != nullptr && item.metadata_len > 0) {
        m.tenant.assign(reinterpret_cast<const char*>(item.metadata), item.metadata_len);
        m.device_id = m.tenant;
        m.location_id = m.tenant;
    }
    if (item.struct_size >= offsetof(pomai_upsert_t, timestamp) + sizeof(uint64_t)) {
        m.timestamp = item.timestamp;
    }
    if (item.struct_size >= offsetof(pomai_upsert_t, payload_len) + sizeof(uint32_t) &&
        item.payload != nullptr && item.payload_len > 0) {
        m.payload.assign(reinterpret_cast<const char*>(item.payload), item.payload_len);
    }
    return m;
}

inline const char* ResolveMembrane(const char* explicit_membrane, const char* struct_membrane) {
    if (explicit_membrane != nullptr && explicit_membrane[0] != '\0') {
        return explicit_membrane;
    }
    if (struct_membrane != nullptr && struct_membrane[0] != '\0') {
        return struct_membrane;
    }
    return kDefaultMembrane;
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

    opts->quant_type = POMAI_QUANT_NONE;
    opts->pq_m = 8;
    opts->memtable_flush_threshold_mb = 64;
    opts->auto_freeze_on_pressure = true;
    opts->max_memtable_mb = 0;
    opts->write_coalesce_window_us = 0;
    opts->write_coalesce_batch_size = 256;
    opts->enable_encryption_at_rest = false;
    opts->encryption_key_hex = nullptr;
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
    opts->membrane = nullptr;
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

    if (opts->metric == 1) {
        db_opts.metric = pomai::MetricType::kInnerProduct;
    } else if (opts->metric == 2) {
        db_opts.metric = pomai::MetricType::kCosine;
    } else {
        db_opts.metric = pomai::MetricType::kL2;
    }

    if (opts->memory_budget_bytes > 0) {
        uint32_t budget_mb = static_cast<uint32_t>(opts->memory_budget_bytes / (1024ULL * 1024ULL));
        if (budget_mb > 0) {
            db_opts.max_memtable_mb = budget_mb;
            db_opts.memtable_flush_threshold_mb = std::max(1u, budget_mb / 4);
        }
    }

    if (opts->struct_size >= offsetof(pomai_options_t, quant_type) + sizeof(uint8_t)) {
        db_opts.index_params.quant_type = static_cast<pomai::QuantizationType>(opts->quant_type);
        db_opts.enable_quantization = (opts->quant_type != POMAI_QUANT_NONE);
    }
    if (opts->struct_size >= offsetof(pomai_options_t, pq_m) + sizeof(uint32_t) && opts->pq_m > 0) {
        db_opts.index_params.pq_m = opts->pq_m;
    }
    if (opts->struct_size >= offsetof(pomai_options_t, memtable_flush_threshold_mb) + sizeof(uint32_t) && opts->memtable_flush_threshold_mb > 0) {
        db_opts.memtable_flush_threshold_mb = opts->memtable_flush_threshold_mb;
    }
    if (opts->struct_size >= offsetof(pomai_options_t, auto_freeze_on_pressure) + sizeof(bool)) {
        db_opts.auto_freeze_on_pressure = opts->auto_freeze_on_pressure;
    }
    if (opts->struct_size >= offsetof(pomai_options_t, max_memtable_mb) + sizeof(uint32_t) && opts->max_memtable_mb > 0) {
        db_opts.max_memtable_mb = opts->max_memtable_mb;
    }
    if (opts->struct_size >= offsetof(pomai_options_t, write_coalesce_window_us) + sizeof(uint32_t)) {
        db_opts.write_coalesce_window_us = opts->write_coalesce_window_us;
    }
    if (opts->struct_size >= offsetof(pomai_options_t, write_coalesce_batch_size) + sizeof(uint32_t) && opts->write_coalesce_batch_size > 0) {
        db_opts.write_coalesce_batch_size = opts->write_coalesce_batch_size;
    }
    if (opts->struct_size >= offsetof(pomai_options_t, enable_encryption_at_rest) + sizeof(bool)) {
        db_opts.enable_encryption_at_rest = opts->enable_encryption_at_rest;
    }
    if (opts->struct_size >= offsetof(pomai_options_t, encryption_key_hex) + sizeof(const char*) && opts->encryption_key_hex != nullptr) {
        db_opts.encryption_key_hex = opts->encryption_key_hex;
    }

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
    RegisterHandle(*out_db);
    return nullptr;
}

pomai_status_t* pomai_close(pomai_db_t* db) {
    if (db == nullptr) {
        return nullptr;
    }
    if (!UnregisterHandle(db)) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "database handle already closed or invalid");
    }
    auto st = db->db->Close();
    db->~pomai_db_t();
    palloc_free(db);
    return ToCStatus(st);
}

pomai_status_t* pomai_freeze_membrane(pomai_db_t* db, const char* membrane) {
    if (!IsValidHandle(db)) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db must be non-null and valid");
    }
    const char* memb = (membrane && membrane[0] != '\0') ? membrane : kDefaultMembrane;
    return ToCStatus(db->db->Freeze(memb));
}

pomai_status_t* pomai_freeze(pomai_db_t* db) {
    return pomai_freeze_membrane(db, nullptr);
}

pomai_status_t* pomai_flush(pomai_db_t* db) {
    if (!IsValidHandle(db)) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db must be non-null and valid");
    }
    return ToCStatus(db->db->Flush());
}

pomai_status_t* pomai_compact(pomai_db_t* db) {
    if (!IsValidHandle(db)) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db must be non-null and valid");
    }
    return ToCStatus(db->db->Compact(kDefaultMembrane));
}

pomai_status_t* pomai_put_membrane(pomai_db_t* db, const char* membrane, const pomai_upsert_t* item) {
    if (!IsValidHandle(db) || item == nullptr || item->vector == nullptr || item->dim == 0) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid put arguments");
    }
    if (item->struct_size < MinUpsertStructSize()) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "upsert.struct_size is too small");
    }
    for (uint32_t d = 0; d < item->dim; ++d) {
        if (!std::isfinite(item->vector[d])) {
            return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "vector contains non-finite values (NaN or Inf)");
        }
    }
    const char* struct_memb = (item->struct_size >= offsetof(pomai_upsert_t, membrane) + sizeof(const char*))
                                ? item->membrane : nullptr;
    const char* memb = ResolveMembrane(membrane, struct_memb);
    std::span<const float> vec(item->vector, item->dim);
    return ToCStatus(db->db->PutVector(memb, item->id, vec, ToMetadata(*item)));
}

pomai_status_t* pomai_put(pomai_db_t* db, const pomai_upsert_t* item) {
    return pomai_put_membrane(db, nullptr, item);
}

pomai_status_t* pomai_put_batch_membrane(pomai_db_t* db, const char* membrane, const pomai_upsert_t* items, size_t n) {
    if (!IsValidHandle(db)) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db must be non-null and valid");
    }
    if (n == 0) {
        return nullptr;
    }
    if (items == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "items must be non-null");
    }

    const char* struct_memb = (items[0].struct_size >= offsetof(pomai_upsert_t, membrane) + sizeof(const char*))
                                ? items[0].membrane : nullptr;
    const char* memb = ResolveMembrane(membrane, struct_memb);

    for (size_t i = 0; i < n; ++i) {
        if (items[i].struct_size < MinUpsertStructSize()) {
            return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "all batch items require valid struct_size");
        }
        if (items[i].vector == nullptr || items[i].dim == 0) {
            return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "all batch items require vector and dim");
        }
        for (uint32_t d = 0; d < items[i].dim; ++d) {
            if (!std::isfinite(items[i].vector[d])) {
                return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "vector contains non-finite values (NaN or Inf)");
            }
        }
        std::span<const float> vec(items[i].vector, items[i].dim);
        auto s = db->db->PutVector(memb, items[i].id, vec, ToMetadata(items[i]));
        if (!s.ok()) {
            return ToCStatus(s);
        }
    }
    return nullptr;
}

pomai_status_t* pomai_put_batch(pomai_db_t* db, const pomai_upsert_t* items, size_t n) {
    return pomai_put_batch_membrane(db, nullptr, items, n);
}

pomai_status_t* pomai_delete_membrane(pomai_db_t* db, const char* membrane, uint64_t id) {
    if (db == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db must be non-null");
    }
    const char* memb = (membrane && membrane[0] != '\0') ? membrane : kDefaultMembrane;
    return ToCStatus(db->db->Delete(memb, id));
}

pomai_status_t* pomai_delete(pomai_db_t* db, uint64_t id) {
    return pomai_delete_membrane(db, nullptr, id);
}

pomai_status_t* pomai_exists_membrane(pomai_db_t* db, const char* membrane, uint64_t id, bool* out_exists) {
    if (db == nullptr || out_exists == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid exists arguments");
    }
    const char* memb = (membrane && membrane[0] != '\0') ? membrane : kDefaultMembrane;
    return ToCStatus(db->db->Exists(memb, id, out_exists));
}

pomai_status_t* pomai_exists(pomai_db_t* db, uint64_t id, bool* out_exists) {
    return pomai_exists_membrane(db, nullptr, id, out_exists);
}

pomai_status_t* pomai_get_membrane(pomai_db_t* db, const char* membrane, uint64_t id, pomai_record_t** out_record) {
    if (db == nullptr || out_record == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid get arguments");
    }
    const char* memb = (membrane && membrane[0] != '\0') ? membrane : kDefaultMembrane;

    void* raw = palloc_malloc_aligned(sizeof(RecordWrapper), alignof(RecordWrapper));
    if (!raw) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "record allocation failed");
    auto* wrapper = new (raw) RecordWrapper();

    pomai::Metadata meta;
    auto st = db->db->Get(memb, id, &wrapper->vec_data, &meta);
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

    wrapper->pub.timestamp = meta.timestamp;
    if (!meta.payload.empty()) {
        wrapper->payload_data.assign(meta.payload.begin(), meta.payload.end());
        wrapper->pub.payload = wrapper->payload_data.data();
        wrapper->pub.payload_len = static_cast<uint32_t>(wrapper->payload_data.size());
    } else {
        wrapper->pub.payload = nullptr;
        wrapper->pub.payload_len = 0;
    }

    *out_record = &wrapper->pub;
    return nullptr;
}

pomai_status_t* pomai_get(pomai_db_t* db, uint64_t id, pomai_record_t** out_record) {
    return pomai_get_membrane(db, nullptr, id, out_record);
}

void pomai_record_free(pomai_record_t* record) {
    if (record == nullptr) {
        return;
    }
    auto* wrapper = reinterpret_cast<RecordWrapper*>(record);
    wrapper->~RecordWrapper();
    palloc_free(wrapper);
}

pomai_status_t* pomai_search_membrane(pomai_db_t* db, const char* membrane, const pomai_query_t* query, pomai_search_results_t** out) {
    if (!IsValidHandle(db) || query == nullptr || out == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid search arguments");
    }
    if (query->struct_size < MinQueryStructSize()) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "query.struct_size is too small");
    }
    if (query->vector == nullptr || query->dim == 0) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "query vector and dim required");
    }
    for (uint32_t d = 0; d < query->dim; ++d) {
        if (!std::isfinite(query->vector[d])) {
            return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "query vector contains non-finite values (NaN or Inf)");
        }
    }
    if (DeadlineExceeded(query->deadline_ms)) {
        return MakeStatus(POMAI_STATUS_DEADLINE_EXCEEDED, "deadline exceeded before search execution");
    }

    const char* struct_memb = (query->struct_size >= offsetof(pomai_query_t, membrane) + sizeof(const char*))
                                ? query->membrane : nullptr;
    const char* memb = ResolveMembrane(membrane, struct_memb);

    pomai::SearchOptions opts;
    opts.zero_copy = ((query->flags & POMAI_QUERY_FLAG_ZERO_COPY) != 0);
    if (query->partition_device_id != nullptr && *query->partition_device_id != '\0') {
        opts.partition_device_id = query->partition_device_id;
    }
    if (query->partition_location_id != nullptr && *query->partition_location_id != '\0') {
        opts.partition_location_id = query->partition_location_id;
    }
    if (query->struct_size >= offsetof(pomai_query_t, as_of_ts) + sizeof(uint64_t)) {
        opts.as_of_ts = query->as_of_ts;
    }
    if (query->struct_size >= offsetof(pomai_query_t, as_of_lsn) + sizeof(uint64_t)) {
        opts.as_of_lsn = query->as_of_lsn;
    }
    if (!ParseTenantFilter(query->filter_expression, &opts)) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "unsupported filter expression (supported: tenant/device_id/location_id=<val>)");
    }

    std::span<const float> q(query->vector, query->dim);

    if (!opts.zero_copy) {
        CApiHitSink sink(query->topk);
        auto st = db->db->SearchVector(memb, q, query->topk, opts, sink);
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
    auto st = db->db->SearchVector(memb, q, query->topk, opts, &res);
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

pomai_status_t* pomai_search(pomai_db_t* db, const pomai_query_t* query, pomai_search_results_t** out) {
    return pomai_search_membrane(db, nullptr, query, out);
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

    const char* struct_memb = (queries[0].struct_size >= offsetof(pomai_query_t, membrane) + sizeof(const char*))
                                ? queries[0].membrane : nullptr;
    const char* memb = ResolveMembrane(nullptr, struct_memb);

    pomai::SearchOptions opts;
    opts.zero_copy = ((queries[0].flags & POMAI_QUERY_FLAG_ZERO_COPY) != 0);

    std::vector<pomai::SearchResult> batch_res;
    auto st = db->db->SearchBatch(memb, flat_queries, static_cast<uint32_t>(num_queries), queries[0].topk, opts, &batch_res);
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

pomai_status_t* pomai_drop_membrane(pomai_db_t* db, const char* membrane_name) {
    if (db == nullptr || membrane_name == nullptr || membrane_name[0] == '\0') {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db and membrane_name must be non-null");
    }
    return ToCStatus(db->db->DropMembrane(membrane_name));
}

pomai_status_t* pomai_open_membrane(pomai_db_t* db, const char* membrane_name) {
    if (db == nullptr || membrane_name == nullptr || membrane_name[0] == '\0') {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db and membrane_name must be non-null");
    }
    return ToCStatus(db->db->OpenMembrane(membrane_name));
}

pomai_status_t* pomai_close_membrane(pomai_db_t* db, const char* membrane_name) {
    if (db == nullptr || membrane_name == nullptr || membrane_name[0] == '\0') {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db and membrane_name must be non-null");
    }
    return ToCStatus(db->db->CloseMembrane(membrane_name));
}

pomai_status_t* pomai_update_membrane_retention(
    pomai_db_t* db, const char* membrane_name,
    uint32_t ttl_sec, uint32_t retention_max_count, uint64_t retention_max_bytes) {
    if (db == nullptr || membrane_name == nullptr || membrane_name[0] == '\0') {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db and membrane_name must be non-null");
    }
    return ToCStatus(db->db->UpdateMembraneRetention(membrane_name, ttl_sec, retention_max_count, retention_max_bytes));
}

pomai_status_t* pomai_get_membrane_retention(
    pomai_db_t* db, const char* membrane_name,
    uint32_t* ttl_sec, uint32_t* retention_max_count, uint64_t* retention_max_bytes) {
    if (db == nullptr || membrane_name == nullptr || membrane_name[0] == '\0') {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db and membrane_name must be non-null");
    }
    return ToCStatus(db->db->GetMembraneRetention(membrane_name, ttl_sec, retention_max_count, retention_max_bytes));
}

pomai_status_t* pomai_get_stats_json(pomai_db_t* db, char** out_json, size_t* out_len) {
    if (db == nullptr || out_json == nullptr || out_len == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid get_stats arguments");
    }
    std::vector<std::string> membranes;
    auto st = db->db->ListMembranes(&membranes);
    if (!st.ok()) return ToCStatus(st);

    std::string json = "{";
    json += "\"version\":\"" + std::string(pomai_version_string()) + "\",";
    json += "\"abi_version\":" + std::to_string(pomai_abi_version()) + ",";
    json += "\"membranes\":[";
    for (size_t i = 0; i < membranes.size(); ++i) {
        if (i > 0) json += ",";
        json += "\"" + JsonEscape(membranes[i]) + "\"";
    }
    json += "]}";

    char* p = static_cast<char*>(palloc_malloc_aligned(json.size() + 1, alignof(char)));
    if (!p) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "allocation failed");
    std::memcpy(p, json.data(), json.size());
    p[json.size()] = '\0';
    *out_json = p;
    *out_len = json.size();
    return nullptr;
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
