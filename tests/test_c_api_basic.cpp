#include "tests/common/test_main.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "c_api.h"

namespace {

class CApiFixture {
public:
    CApiFixture() {
        pomai_options_init(&opts_);
        dir_ = "test_db_capi_" + std::to_string(std::rand());
        opts_.path = dir_.c_str();
        opts_.dim = 8;
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
        auto* st = pomai_open(&opts_, &db_);
        if (st != nullptr) {
            std::abort();
        }
    }

    ~CApiFixture() {
        if (db_ != nullptr) {
            pomai_status_t* st = pomai_close(db_);
            if (st != nullptr) {
                pomai_status_free(st);
            }
        }
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    pomai_db_t* db() { return db_; }

    std::array<float, 8> Vec(float v0) {
        std::array<float, 8> out{};
        out[0] = v0;
        return out;
    }

private:
    std::string dir_;
    pomai_options_t opts_{};
    pomai_db_t* db_ = nullptr;
};

#define CAPI_EXPECT_OK(expr)                                            \
    do {                                                                 \
        pomai_status_t* _st = (expr);                                    \
        if (_st != nullptr) {                                            \
            std::cerr << "status=" << pomai_status_code(_st) << " msg=" \
                      << pomai_status_message(_st) << "\n";             \
            pomai_status_free(_st);                                      \
            std::abort();                                                \
        }                                                                \
    } while (0)

POMAI_TEST(CApiOpenCloseAndVersion) {
    CApiFixture fx;
    POMAI_EXPECT_TRUE(pomai_abi_version() > 0);
    POMAI_EXPECT_TRUE(std::string(pomai_version_string()).size() > 0);
}

POMAI_TEST(CApiPutGetExistsDelete) {
    CApiFixture fx;
    const auto v = fx.Vec(1.5f);
    const char tenant[] = "tenantA";

    pomai_upsert_t up{};
    up.struct_size = sizeof(pomai_upsert_t);
    up.id = 42;
    up.vector = v.data();
    up.dim = static_cast<uint32_t>(v.size());
    up.metadata = reinterpret_cast<const uint8_t*>(tenant);
    up.metadata_len = static_cast<uint32_t>(sizeof(tenant) - 1);

    CAPI_EXPECT_OK(pomai_put(fx.db(), &up));

    bool exists = false;
    CAPI_EXPECT_OK(pomai_exists(fx.db(), 42, &exists));
    POMAI_EXPECT_TRUE(exists);

    pomai_record_t* rec = nullptr;
    CAPI_EXPECT_OK(pomai_get(fx.db(), 42, &rec));
    POMAI_EXPECT_TRUE(rec != nullptr);
    POMAI_EXPECT_EQ(rec->id, 42u);
    POMAI_EXPECT_EQ(rec->dim, 8u);
    POMAI_EXPECT_EQ(std::string(reinterpret_cast<const char*>(rec->metadata), rec->metadata_len), "tenantA");
    pomai_record_free(rec);

    CAPI_EXPECT_OK(pomai_delete(fx.db(), 42));
    CAPI_EXPECT_OK(pomai_exists(fx.db(), 42, &exists));
    POMAI_EXPECT_TRUE(!exists);
}

POMAI_TEST(CApiPutBatchSearchAndSnapshotScan) {
    CApiFixture fx;
    std::vector<std::array<float, 8>> vectors(16);
    std::vector<pomai_upsert_t> batch(16);
    for (size_t i = 0; i < batch.size(); ++i) {
        vectors[i][0] = static_cast<float>(i);
        batch[i].struct_size = sizeof(pomai_upsert_t);
        batch[i].id = static_cast<uint64_t>(i + 1);
        batch[i].vector = vectors[i].data();
        batch[i].dim = 8;
    }
    CAPI_EXPECT_OK(pomai_put_batch(fx.db(), batch.data(), batch.size()));

    auto qv = fx.Vec(11.0f);
    pomai_query_t query{};
    query.struct_size = sizeof(pomai_query_t);
    query.vector = qv.data();
    query.dim = 8;
    query.topk = 5;

    pomai_search_results_t* results = nullptr;
    CAPI_EXPECT_OK(pomai_search(fx.db(), &query, &results));
    POMAI_EXPECT_TRUE(results != nullptr);
    POMAI_EXPECT_TRUE(results->count >= 1);
    bool found = false;
    for (size_t i = 0; i < results->count; ++i) {
        if (results->ids[i] == 12u) {
            found = true;
        }
    }
    POMAI_EXPECT_TRUE(found);
    pomai_search_results_free(results);

    auto v1 = fx.Vec(100.0f);
    pomai_upsert_t up{};
    up.struct_size = sizeof(pomai_upsert_t);
    up.id = 100;
    up.vector = v1.data();
    up.dim = 8;
    CAPI_EXPECT_OK(pomai_put(fx.db(), &up));

    pomai_snapshot_t* snap = nullptr;
    CAPI_EXPECT_OK(pomai_get_snapshot(fx.db(), &snap));

    auto v2 = fx.Vec(200.0f);
    up.id = 200;
    up.vector = v2.data();
    CAPI_EXPECT_OK(pomai_put(fx.db(), &up));

    pomai_scan_options_t scan_opts{};
    pomai_scan_options_init(&scan_opts);
    pomai_iter_t* it = nullptr;
    CAPI_EXPECT_OK(pomai_scan(fx.db(), &scan_opts, snap, &it));

    bool saw200 = false;
    while (pomai_iter_valid(it)) {
        pomai_record_view_t view{};
        view.struct_size = sizeof(pomai_record_view_t);
        CAPI_EXPECT_OK(pomai_iter_get_record(it, &view));
        if (view.id == 200) saw200 = true;
        pomai_iter_next(it);
    }
    CAPI_EXPECT_OK(pomai_iter_status(it));
    pomai_iter_free(it);
    pomai_snapshot_free(snap);

    POMAI_EXPECT_TRUE(!saw200);
}




POMAI_TEST(CApiFreezePublishesToSnapshotScan) {
    CApiFixture fx;

    pomai_snapshot_t* before = nullptr;
    CAPI_EXPECT_OK(pomai_get_snapshot(fx.db(), &before));

    auto v = fx.Vec(3.0f);
    pomai_upsert_t up{};
    up.struct_size = sizeof(pomai_upsert_t);
    up.id = 778;
    up.vector = v.data();
    up.dim = 8;

    CAPI_EXPECT_OK(pomai_put(fx.db(), &up));
    CAPI_EXPECT_OK(pomai_freeze(fx.db()));

    pomai_snapshot_t* after = nullptr;
    CAPI_EXPECT_OK(pomai_get_snapshot(fx.db(), &after));

    auto scan_has_id = [&](pomai_snapshot_t* snap, uint64_t id) {
        pomai_scan_options_t scan_opts{};
        pomai_scan_options_init(&scan_opts);
        pomai_iter_t* it = nullptr;
        CAPI_EXPECT_OK(pomai_scan(fx.db(), &scan_opts, snap, &it));

        bool found = false;
        while (pomai_iter_valid(it)) {
            pomai_record_view_t view{};
            view.struct_size = sizeof(pomai_record_view_t);
            CAPI_EXPECT_OK(pomai_iter_get_record(it, &view));
            if (view.id == id) {
                found = true;
            }
            pomai_iter_next(it);
        }
        CAPI_EXPECT_OK(pomai_iter_status(it));
        pomai_iter_free(it);
        return found;
    };

    POMAI_EXPECT_TRUE(!scan_has_id(before, 778));
    POMAI_EXPECT_TRUE(scan_has_id(after, 778));

    pomai_snapshot_free(before);
    pomai_snapshot_free(after);
}

POMAI_TEST(CApiMembraneAndFeatures) {
    CApiFixture fx;

    // Test membrane creation
    CAPI_EXPECT_OK(pomai_create_membrane_kind(fx.db(), "custom_memb", 8, 0));

    char* list_json = nullptr;
    size_t list_len = 0;
    CAPI_EXPECT_OK(pomai_list_membranes_json(fx.db(), &list_json, &list_len));
    POMAI_EXPECT_TRUE(list_json != nullptr);
    POMAI_EXPECT_TRUE(std::string(list_json).find("custom_memb") != std::string::npos);
    pomai_free(list_json);

    // Put into custom membrane with metadata and payload
    auto v = fx.Vec(4.5f);
    std::string payload_str = "{\"key\":\"val\"}";
    pomai_upsert_t up{};
    up.struct_size = sizeof(pomai_upsert_t);
    up.id = 999;
    up.vector = v.data();
    up.dim = 8;
    up.membrane = "custom_memb";
    up.timestamp = 12345678ULL;
    up.payload = reinterpret_cast<const uint8_t*>(payload_str.data());
    up.payload_len = static_cast<uint32_t>(payload_str.size());

    CAPI_EXPECT_OK(pomai_put(fx.db(), &up));

    // Exists & Get from custom membrane
    bool exists = false;
    CAPI_EXPECT_OK(pomai_exists_membrane(fx.db(), "custom_memb", 999, &exists));
    POMAI_EXPECT_TRUE(exists);

    pomai_record_t* rec = nullptr;
    CAPI_EXPECT_OK(pomai_get_membrane(fx.db(), "custom_memb", 999, &rec));
    POMAI_EXPECT_TRUE(rec != nullptr);
    POMAI_EXPECT_EQ(rec->id, 999ULL);
    POMAI_EXPECT_EQ(rec->dim, 8u);
    POMAI_EXPECT_EQ(rec->timestamp, 12345678ULL);
    POMAI_EXPECT_TRUE(rec->payload != nullptr);
    std::string got_payload(reinterpret_cast<const char*>(rec->payload), rec->payload_len);
    POMAI_EXPECT_EQ(got_payload, payload_str);
    pomai_record_free(rec);

    // Search inside custom membrane
    pomai_query_t query{};
    query.struct_size = sizeof(pomai_query_t);
    query.vector = v.data();
    query.dim = 8;
    query.topk = 5;
    query.membrane = "custom_memb";

    pomai_search_results_t* res = nullptr;
    CAPI_EXPECT_OK(pomai_search(fx.db(), &query, &res));
    POMAI_EXPECT_TRUE(res != nullptr);
    POMAI_EXPECT_EQ(res->count, 1u);
    POMAI_EXPECT_EQ(res->ids[0], 999ULL);
    pomai_search_results_free(res);

    // Test flush and compact
    CAPI_EXPECT_OK(pomai_flush(fx.db()));
    CAPI_EXPECT_OK(pomai_compact_membrane(fx.db(), "custom_memb"));

    // Test stats json
    char* stats_json = nullptr;
    size_t stats_len = 0;
    CAPI_EXPECT_OK(pomai_get_stats_json(fx.db(), &stats_json, &stats_len));
    POMAI_EXPECT_TRUE(stats_json != nullptr);
    POMAI_EXPECT_TRUE(std::string(stats_json).find("PomaiDB") != std::string::npos);
    pomai_free(stats_json);

    // Drop membrane
    CAPI_EXPECT_OK(pomai_drop_membrane(fx.db(), "custom_memb"));
}

POMAI_TEST(CApiDeadlines) {
    CApiFixture fx;
    auto qv = fx.Vec(1.0f);
    pomai_query_t query{};
    query.struct_size = sizeof(pomai_query_t);
    query.vector = qv.data();
    query.dim = 8;
    query.topk = 1;
    query.deadline_ms = 1;

    pomai_search_results_t* results = nullptr;
    pomai_status_t* st = pomai_search(fx.db(), &query, &results);
    POMAI_EXPECT_TRUE(st != nullptr);
    POMAI_EXPECT_EQ(pomai_status_code(st), static_cast<int>(POMAI_STATUS_DEADLINE_EXCEEDED));
    pomai_status_free(st);
}

POMAI_TEST(CApiInvalidArgs) {
    pomai_status_t* st = pomai_put(nullptr, nullptr);
    POMAI_EXPECT_TRUE(st != nullptr);
    POMAI_EXPECT_EQ(pomai_status_code(st), static_cast<int>(POMAI_STATUS_INVALID_ARGUMENT));
    pomai_status_free(st);
}

}  // namespace
