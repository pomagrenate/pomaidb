#include "pomai/pomaidb.h"
#include "tests/common/test_main.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

namespace pomai::pomaidb_qa {

POMAI_TEST(PomaiDB_LifecycleBatchInsertAndKNN) {
    pdb_options_t opts;
    pdb_status_t st = pdb_options_init(&opts, 128);
    POMAI_EXPECT_EQ(st, PDB_SUCCESS);
    opts.memtable_capacity = 2000;
    opts.metric = PDB_METRIC_L2;

    pdb_t* db = nullptr;
    st = pdb_open(nullptr, &opts, &db);
    POMAI_EXPECT_EQ(st, PDB_SUCCESS);
    POMAI_EXPECT_TRUE(db != nullptr);

    constexpr size_t kCount = 200;
    constexpr size_t kDim = 128;
    uint64_t ids[kCount];
    float vectors[kCount * kDim];

    for (size_t i = 0; i < kCount; ++i) {
        ids[i] = 100000 + i;
        for (size_t d = 0; d < kDim; ++d) {
            vectors[i * kDim + d] = static_cast<float>(i) * 0.05f;
        }
    }

    pdb_vector_batch_t batch;
    batch.ids = ids;
    batch.vectors = vectors;
    batch.count = kCount;
    batch.dim = kDim;
    batch.metadata_jsons = nullptr;

    st = pdb_insert_batch(db, &batch);
    POMAI_EXPECT_EQ(st, PDB_SUCCESS);

    pdb_stats_t stats{};
    st = pdb_get_stats(db, &stats);
    POMAI_EXPECT_EQ(st, PDB_SUCCESS);
    POMAI_EXPECT_EQ(stats.memtable_vectors, kCount);
    POMAI_EXPECT_EQ(stats.total_vectors, kCount);

    // Query exact point close to vector 20 (id 100020)
    float query[kDim];
    for (size_t d = 0; d < kDim; ++d) {
        query[d] = 20.0f * 0.05f; // 1.0f
    }

    constexpr size_t kTopK = 5;
    uint64_t out_ids[kTopK];
    float out_dist[kTopK];
    size_t actual_k = 0;

    st = pdb_query_knn(db, query, kTopK, nullptr, out_ids, out_dist, &actual_k);
    POMAI_EXPECT_EQ(st, PDB_SUCCESS);
    POMAI_EXPECT_EQ(actual_k, kTopK);
    POMAI_EXPECT_EQ(out_ids[0], 100020u);
    POMAI_EXPECT_TRUE(out_dist[0] < 0.0001f);

    // Test deletion
    uint64_t del_ids[1] = {100020u};
    st = pdb_delete_batch(db, del_ids, 1);
    POMAI_EXPECT_EQ(st, PDB_SUCCESS);

    // Re-query: deleted vector 100020 must be omitted via tombstone bitset
    st = pdb_query_knn(db, query, kTopK, nullptr, out_ids, out_dist, &actual_k);
    POMAI_EXPECT_EQ(st, PDB_SUCCESS);
    POMAI_EXPECT_TRUE(out_ids[0] != 100020u);

    // Checkpoint
    st = pdb_checkpoint(db);
    POMAI_EXPECT_EQ(st, PDB_SUCCESS);

    st = pdb_close(db);
    POMAI_EXPECT_EQ(st, PDB_SUCCESS);
}

} // namespace pomai::pomaidb_qa
