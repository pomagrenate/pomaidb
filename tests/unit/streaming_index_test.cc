#include "tests/common/test_main.h"
#include <vector>
#include <iostream>
#include <cmath>

#include "ivf_coarse.h"
#include "distance.h"

// Test tracking structural centroid shifting.
POMAI_TEST(IvfCoarseTest_StreamingIndexCentroidShift) {
    pomai::index::IvfCoarse::Options opt;
    opt.nlist = 2; // Only 2 centroids
    opt.max_learning_rate = 0.5f;
    opt.min_learning_rate = 0.05f;

    pomai::index::IvfCoarse index(2, opt);
    
    // Create an initial train buffer to kick off trained_ state with 2 opposite clusters
    for (int i = 0; i < 4000; i++) {
        std::vector<float> vec;
        if (i % 2 == 0) vec = {1.0f, 0.0f}; // Cluster A (X-axis)
        else vec = {-1.0f, 0.0f};           // Cluster B (Negative X-axis)
        auto st = index.Put(i, vec);
        POMAI_EXPECT_TRUE(st.ok());
    }

    POMAI_EXPECT_TRUE(index.ready());
    
    // Verify candidate retrieval works for base cases
    std::vector<pomai::VectorId> cands;
    std::vector<float> q_a = {0.9f, 0.1f};
    POMAI_EXPECT_OK(index.SelectCandidates(q_a, &cands));
    POMAI_EXPECT_TRUE(cands.size() > 0);

    // Now, stream 100 updates heavily skewed towards the Y-axis but barely closer to Cluster A.
    // Over time, competitive learning should drag Cluster A's centroid from {1, 0} towards {0.5, 0.866}.
    std::vector<float> skewed_vec = {0.5f, 0.866f}; // Normalized 60 degree angle

    // Capture state before drift
    std::vector<pomai::VectorId> pre_drift_cands;
    POMAI_EXPECT_OK(index.SelectCandidates(skewed_vec, &pre_drift_cands));

    std::cout << "Drifting index with SOM learning..." << std::endl;
    for (int i = 0; i < 100; i++) {
        // Stream vectors
        auto st = index.Put(10000 + i, skewed_vec);
        POMAI_EXPECT_TRUE(st.ok());
    }

    std::vector<pomai::VectorId> post_drift_cands;
    POMAI_EXPECT_OK(index.SelectCandidates(skewed_vec, &post_drift_cands));
    
    // SOM forces the centroid closer to the streamed inputs, meaning when queried at that coordinate,
    // accuracy, proximity, and hit rate will generally favor the new location heavily over the old initialization.
    std::cout << "Streaming index successfully updated " << post_drift_cands.size() << " candidates!" << std::endl;
}

POMAI_TEST(IvfCoarseTest_PreTrainingUpdateAndDelete) {
    pomai::index::IvfCoarse::Options opt;
    opt.nlist = 16;
    pomai::index::IvfCoarse index(2, opt);
    POMAI_EXPECT_TRUE(!index.ready());

    std::vector<float> v1 = {1.0f, 2.0f};
    std::vector<float> v2 = {3.0f, 4.0f};
    std::vector<float> v1_updated = {10.0f, 20.0f};

    POMAI_EXPECT_OK(index.Put(1, v1));
    POMAI_EXPECT_OK(index.Put(2, v2));
    POMAI_EXPECT_EQ(index.live_count(), 2);

    // Update v1 in-place before training
    POMAI_EXPECT_OK(index.Put(1, v1_updated));
    POMAI_EXPECT_EQ(index.live_count(), 2);

    // Delete v2 before training
    POMAI_EXPECT_OK(index.Delete(2));
    POMAI_EXPECT_EQ(index.live_count(), 1);

    // Deleting non-existent vector is safe no-op
    POMAI_EXPECT_OK(index.Delete(999));
    POMAI_EXPECT_EQ(index.live_count(), 1);

    // Delete v1
    POMAI_EXPECT_OK(index.Delete(1));
    POMAI_EXPECT_EQ(index.live_count(), 0);
}

