// tests/unit/spatial_press_test.cc — Unit test for Spatial Locule Partitioning in Press
//
// Verifies:
// 1. Press::Compact partitions live vectors into cohesive spatial clusters rather than sequential IDs.
// 2. Locule anchors have distinct centroids and tight bounding radii.
// 3. Compass::Orient ranks the spatially closest Locule first.
// 4. Compass::Peel effectively prunes non-candidate Locules.
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "tests/common/test_main.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "compass.h"
#include "fruit_map.h"
#include "press.h"
#include "rind.h"
#include "status.h"
#include "types.h"
#include "utils/env.h"

namespace {

POMAI_TEST(SpatialPress_ClustersSeparatedCorrectly) {
    auto* env = pomai::Env::Default();
    std::string test_dir = "test_spatial_press_db";
    (void)env->DeleteFile(test_dir);

    const uint32_t dim = 4;
    pomai::ingest::Rind rind(env, test_dir, dim, pomai::MetricType::kL2,
                             pomai::FsyncPolicy::kNever, 4 * 1024 * 1024);
    POMAI_EXPECT_OK(rind.Open());

    // Generate 300 vectors across 3 distinct spatial clusters
    // Cluster 0: centered at (10, 0, 0, 0)
    // Cluster 1: centered at (0, 10, 0, 0)
    // Cluster 2: centered at (0, 0, 10, 0)
    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, 0.2f);

    const size_t pts_per_cluster = 100;
    for (size_t c = 0; c < 3; ++c) {
        for (size_t i = 0; i < pts_per_cluster; ++i) {
            pomai::VectorId id = static_cast<pomai::VectorId>(c * pts_per_cluster + i + 1);
            std::vector<float> vec(dim, 0.0f);
            vec[c] = 10.0f;
            for (size_t d = 0; d < dim; ++d) {
                vec[d] += noise(rng);
            }
            POMAI_EXPECT_OK(rind.Put(id, vec));
        }
    }

    // Freeze memtable to make it eligible for compaction
    POMAI_EXPECT_OK(rind.Freeze());

    pomai::manifest::FruitMap fruit_map(env, test_dir, dim, pomai::MetricType::kL2);
    POMAI_EXPECT_OK(fruit_map.Open());

    pomai::compact::PressOptions press_opts;
    press_opts.target_aril_vector_count = 100;
    press_opts.target_locule_aril_count = 1; // 1 aril per locule -> 100 vectors per locule -> 3 locules

    pomai::compact::Press press(env, test_dir, dim, pomai::MetricType::kL2, press_opts);
    POMAI_EXPECT_OK(press.Compact(&rind, &fruit_map));

    auto snapshot = fruit_map.CurrentSnapshot();
    POMAI_EXPECT_TRUE(snapshot != nullptr);
    POMAI_EXPECT_EQ(snapshot->locules().size(), 3u);

    // Verify each Locule's anchor has a tight radius and matches one cluster
    std::vector<bool> matched_cluster(3, false);
    for (const auto& loc : snapshot->locules()) {
        POMAI_EXPECT_TRUE(loc != nullptr);
        const auto& anchor = loc->anchor();
        POMAI_EXPECT_EQ(anchor.centroid.size(), dim);

        // Bounding radius should be tight (< 2.0) and definitely not dataset-wide (> 10.0)
        POMAI_EXPECT_TRUE(anchor.radius > 0.0f);
        POMAI_EXPECT_TRUE(anchor.radius < 2.0f);

        // Check which cluster centroid it aligns with
        for (size_t c = 0; c < 3; ++c) {
            if (std::abs(anchor.centroid[c] - 10.0f) < 1.0f) {
                matched_cluster[c] = true;
            }
        }
    }

    // All 3 clusters must be distinctly represented by the 3 Locules
    POMAI_EXPECT_TRUE(matched_cluster[0]);
    POMAI_EXPECT_TRUE(matched_cluster[1]);
    POMAI_EXPECT_TRUE(matched_cluster[2]);

    // Test Compass Orient
    pomai::routing::Compass compass(pomai::MetricType::kL2);
    compass.UpdateLocules(snapshot->locules());

    // Query near Cluster 0
    std::vector<float> q0 = {9.9f, 0.1f, -0.1f, 0.0f};
    auto oriented = compass.Orient(q0);
    POMAI_EXPECT_EQ(oriented.size(), 3u);

    // Closest locule should be cluster 0
    POMAI_EXPECT_TRUE(oriented[0].distance_to_centroid < 1.0f);
    POMAI_EXPECT_TRUE(oriented[1].distance_to_centroid > 8.0f);
    POMAI_EXPECT_TRUE(oriented[2].distance_to_centroid > 8.0f);

    // Test Compass Peel: worst distance corresponding to a good hit in cluster 0 (e.g., dist_sq = 2.0)
    float worst_dist_sq = 2.0f;
    auto peeled = compass.Peel(oriented, worst_dist_sq);

    // Only cluster 0 should survive; distant clusters (dist - radius > sqrt(2)) are pruned!
    POMAI_EXPECT_EQ(peeled.size(), 1u);
    POMAI_EXPECT_EQ(peeled[0].locule->anchor().id, oriented[0].locule->anchor().id);

    POMAI_EXPECT_OK(rind.Close());
    (void)env->DeleteFile(test_dir);
}

} // namespace
