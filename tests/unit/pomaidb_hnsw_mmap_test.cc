#include "pomai/pomaidb_hnsw_mmap.h"
#include "tests/common/test_main.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

namespace pomai::hnsw_mmap_qa {

POMAI_TEST(HnswMmap_BuildFreezeAndZeroCopyQuery) {
    constexpr uint32_t kDim = 64;
    constexpr size_t kNumNodes = 300;
    const char* kPath = "test_hnsw_segment.phns";

    // 1. Build in-memory graph
    pdb_hnsw_builder_t* b = pdb_hnsw_builder_create(kDim, 0 /* L2 */, 16, 64);
    POMAI_EXPECT_TRUE(b != nullptr);

    float vec[kDim];
    for (size_t i = 0; i < kNumNodes; ++i) {
        for (uint32_t d = 0; d < kDim; ++d) {
            vec[d] = static_cast<float>(i) * 0.1f + static_cast<float>(d) * 0.001f;
        }
        int rc = pdb_hnsw_builder_add(b, 200000 + i, vec);
        POMAI_EXPECT_EQ(rc, 0);
    }

    // 2. Freeze directly into cache-aligned zero-copy mmap file
    int rc = pdb_hnsw_builder_freeze_file(b, kPath);
    POMAI_EXPECT_EQ(rc, 0);

    pdb_hnsw_builder_destroy(b);

    // 3. Open via mmap (ZERO HEAP ALLOCATIONS for vectors or graph adjacency edges)
    pdb_hnsw_mmap_t graph;
    rc = pdb_hnsw_mmap_open(kPath, &graph);
    POMAI_EXPECT_EQ(rc, 0);
    POMAI_EXPECT_EQ(graph.header->total_nodes, kNumNodes);
    POMAI_EXPECT_EQ(graph.header->dim, kDim);

    // 4. Query vector close to node 25 (id 200025)
    float query[kDim];
    for (uint32_t d = 0; d < kDim; ++d) {
        query[d] = 25.0f * 0.1f + static_cast<float>(d) * 0.001f;
    }

    constexpr uint32_t kTopK = 5;
    uint64_t out_ids[kTopK];
    float out_dists[kTopK];
    uint32_t actual_k = 0;

    rc = pdb_hnsw_mmap_search(&graph, query, kTopK, 32, nullptr, out_ids, out_dists, &actual_k);
    POMAI_EXPECT_EQ(rc, 0);
    POMAI_EXPECT_EQ(actual_k, kTopK);
    // Nearest result must be exact match node 25
    POMAI_EXPECT_EQ(out_ids[0], 200025u);
    POMAI_EXPECT_TRUE(out_dists[0] < 0.0001f);

    // 5. Test with tombstone mask (mask out node 25)
    uint64_t tombstone_mask[(kNumNodes + 63) / 64];
    memset(tombstone_mask, 0, sizeof(tombstone_mask));
    tombstone_mask[25 / 64] |= (1ULL << (25 % 64));

    rc = pdb_hnsw_mmap_search(&graph, query, kTopK, 32, tombstone_mask, out_ids, out_dists, &actual_k);
    POMAI_EXPECT_EQ(rc, 0);
    POMAI_EXPECT_EQ(actual_k, kTopK);
    POMAI_EXPECT_TRUE(out_ids[0] != 200025u);

    pdb_hnsw_mmap_close(&graph);
    remove(kPath);
}

} // namespace pomai::hnsw_mmap_qa
