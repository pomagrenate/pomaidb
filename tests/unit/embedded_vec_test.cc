#include "pomai/embedded_vec.h"
#include "tests/common/test_main.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

namespace pomai::embedded_qa {

POMAI_TEST(EmbeddedVec_LifecycleAndBatchInsert) {
    ev_options_t opts;
    ev_options_init(&opts, 64);
    opts.memtable_capacity = 1000;
    opts.metric = EV_METRIC_L2_SQ;

    ev_db_t* db = nullptr;
    int rc = ev_db_open("test_ev_db", &opts, &db);
    POMAI_EXPECT_EQ(rc, 0);
    POMAI_EXPECT_TRUE(db != nullptr);

    constexpr size_t kBatch = 100;
    constexpr size_t kDim = 64;
    uint64_t ids[kBatch];
    float vectors[kBatch * kDim];

    for (size_t i = 0; i < kBatch; ++i) {
        ids[i] = 1000 + i;
        for (size_t d = 0; d < kDim; ++d) {
            vectors[i * kDim + d] = static_cast<float>(i * 0.01f + d * 0.001f);
        }
    }

    rc = ev_insert_batch(db, ids, vectors, kBatch, nullptr);
    POMAI_EXPECT_EQ(rc, 0);

    ev_stats_t stats{};
    rc = ev_get_stats(db, &stats);
    POMAI_EXPECT_EQ(rc, 0);
    POMAI_EXPECT_EQ(stats.memtable_vectors, kBatch);
    POMAI_EXPECT_EQ(stats.total_vectors, kBatch);

    rc = ev_db_close(db);
    POMAI_EXPECT_EQ(rc, 0);
}

POMAI_TEST(EmbeddedVec_KNNQueryWithFilterMask) {
    constexpr size_t kDim = 128;
    constexpr size_t kCount = 500;

    ev_options_t opts;
    ev_options_init(&opts, kDim);
    opts.memtable_capacity = 2000;
    opts.metric = EV_METRIC_L2_SQ;

    ev_db_t* db = nullptr;
    int rc = ev_db_open(nullptr, &opts, &db);
    POMAI_EXPECT_EQ(rc, 0);

    uint64_t ids[kCount];
    float vectors[kCount * kDim];

    for (size_t i = 0; i < kCount; ++i) {
        ids[i] = 50000 + i;
        for (size_t d = 0; d < kDim; ++d) {
            vectors[i * kDim + d] = static_cast<float>(i) * 0.1f;
        }
    }

    rc = ev_insert_batch(db, ids, vectors, kCount, nullptr);
    POMAI_EXPECT_EQ(rc, 0);

    // Query close to vector index 10 (target: id 50010)
    float query[kDim];
    for (size_t d = 0; d < kDim; ++d) {
        query[d] = 1.0f; // 10 * 0.1 = 1.0
    }

    // 1. Unfiltered query
    constexpr size_t kTopK = 5;
    uint64_t out_ids[kTopK];
    float out_dist[kTopK];
    size_t actual_k = 0;

    rc = ev_query_knn(db, query, kTopK, nullptr, out_ids, out_dist, &actual_k);
    POMAI_EXPECT_EQ(rc, 0);
    POMAI_EXPECT_EQ(actual_k, kTopK);
    POMAI_EXPECT_EQ(out_ids[0], 50010u);
    POMAI_EXPECT_TRUE(out_dist[0] < 0.0001f);

    // 2. Filtered query: Mask out vector 10 (filter only odd-numbered IDs)
    uint64_t filter_mask[16];
    memset(filter_mask, 0, sizeof(filter_mask));
    for (size_t i = 0; i < kCount; ++i) {
        if (i % 2 != 0) { // only odd indices enabled (skips index 10)
            filter_mask[i / 64] |= (1ULL << (i % 64));
        }
    }

    rc = ev_query_knn(db, query, kTopK, filter_mask, out_ids, out_dist, &actual_k);
    POMAI_EXPECT_EQ(rc, 0);
    POMAI_EXPECT_EQ(actual_k, kTopK);
    // Index 10 is filtered out! Nearest odd are index 9 (50009) or 11 (50011)
    POMAI_EXPECT_TRUE(out_ids[0] == 50009u || out_ids[0] == 50011u);

    // 3. Checkpoint clears active MemTable
    rc = ev_checkpoint(db);
    POMAI_EXPECT_EQ(rc, 0);

    ev_stats_t stats{};
    ev_get_stats(db, &stats);
    POMAI_EXPECT_EQ(stats.memtable_vectors, 0u);

    rc = ev_db_close(db);
    POMAI_EXPECT_EQ(rc, 0);
}

} // namespace pomai::embedded_qa
