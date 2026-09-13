// tests/adversarial/compass_spatial_torture_test.cc
// Phase 7: Spatial Partition Torture & Phase 8: Compass Boundary Torture
// Stresses Voronoi cell boundaries, geometric bounding spheres, and Compass pruning safety.

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "tests/adversarial/golden_oracle.h"

#include <cmath>
#include <filesystem>
#include <memory>
#include <random>
#include <vector>

#include "pomegranate_engine.h"
#include "compass.h"
#include "options.h"
#include "search.h"
#include "types.h"

namespace {

using namespace pomai;
using namespace pomai::core;
using namespace pomai::routing;
using namespace pomai::adversarial;

// 1. Exact mathematical safety of Compass::Peel bounding sphere lower bounds
POMAI_TEST(Compass_GeometricBoundingSphereSafety) {
    Compass compass(MetricType::kL2);

    // Candidate when query is at [10.0, 0, 0, 0]
    // Distance to centroid is 5.0. Lower bound distance to any point in sphere is 5.0 - 2.0 = 3.0.
    // L2-squared lower bound is 3.0^2 = 9.0.
    float dist = 5.0f;
    float radius = 2.0f;
    float lower_bound = dist - radius; // 3.0
    float min_dist_sq = lower_bound * lower_bound; // 9.0

    OrientedLocule ol;
    ol.locule = nullptr; // mock
    ol.distance_to_centroid = dist;
    ol.min_possible_distance = min_dist_sq;

    std::vector<OrientedLocule> candidates = {ol};

    // Case A: worst_distance is 8.9 (better than any point in sphere can possibly achieve)
    // Pruning MUST discard this locule!
    auto pruned = compass.Peel(candidates, 8.9f);
    POMAI_EXPECT_EQ(pruned.size(), 0);

    // Case B: worst_distance is 9.1 (sphere could contain a vector with dist_sq 9.0)
    // Pruning MUST NOT discard this locule!
    auto kept = compass.Peel(candidates, 9.1f);
    POMAI_EXPECT_EQ(kept.size(), 1);

    // Case C: query is INSIDE the bounding sphere: query at [4.0, 0, 0, 0]
    // dist to centroid = 1.0 < radius (2.0) -> lower_bound = 0.0 -> min_possible_distance = 0.0
    OrientedLocule ol_inside;
    ol_inside.distance_to_centroid = 1.0f;
    ol_inside.min_possible_distance = 0.0f;
    std::vector<OrientedLocule> candidates_inside = {ol_inside};

    // Even with a very tight worst_distance of 0.01, candidate inside bounding sphere MUST NEVER be pruned!
    auto kept_inside = compass.Peel(candidates_inside, 0.01f);
    POMAI_EXPECT_EQ(kept_inside.size(), 1);
}

// 2. Adversarial Voronoi Boundary Querying & Recall Floor
POMAI_TEST(Compass_VoronoiBoundaryRecallFloor) {
    const std::string db_dir = test::TempDir("pomai-compass-boundary");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 16;
    const uint32_t N_per_cluster = 150;
    const uint32_t N_boundary = 30;

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kNever;
    opt.index_params.type = IndexType::kHnsw;
    opt.index_params.hnsw_m = 16;
    opt.index_params.hnsw_ef_construction = 100;
    opt.index_params.hnsw_ef_search = 64;

    PomegranateEngine engine(opt, MetricType::kL2);
    POMAI_EXPECT_OK(engine.Open());

    std::mt19937 rng(777);
    std::normal_distribution<float> d_noise(0.0f, 0.1f);

    GoldenOracle oracle(dim, GoldenOracle::Metric::kL2);
    uint64_t id = 1;

    // Cluster A: Centered around [-2.0, 0, ...]
    for (uint32_t i = 0; i < N_per_cluster; ++i, ++id) {
        std::vector<float> vec(dim, 0.0f);
        vec[0] = -2.0f + d_noise(rng);
        for (uint32_t d = 1; d < dim; ++d) vec[d] = d_noise(rng);
        oracle.Put(id, vec);
        POMAI_EXPECT_OK(engine.Put(id, vec));
    }

    // Cluster B: Centered around [+2.0, 0, ...]
    for (uint32_t i = 0; i < N_per_cluster; ++i, ++id) {
        std::vector<float> vec(dim, 0.0f);
        vec[0] = 2.0f + d_noise(rng);
        for (uint32_t d = 1; d < dim; ++d) vec[d] = d_noise(rng);
        oracle.Put(id, vec);
        POMAI_EXPECT_OK(engine.Put(id, vec));
    }

    // Boundary vectors: Sits right on the decision boundary x0 = 0.0
    for (uint32_t i = 0; i < N_boundary; ++i, ++id) {
        std::vector<float> vec(dim, 0.0f);
        vec[0] = 0.0f + (d_noise(rng) * 0.01f); // Extremely close to 0.0
        for (uint32_t d = 1; d < dim; ++d) vec[d] = d_noise(rng);
        oracle.Put(id, vec);
        POMAI_EXPECT_OK(engine.Put(id, vec));
    }

    POMAI_EXPECT_OK(engine.Compact());

    // Query directly on the boundary: x0 = 0.001
    const uint32_t K = 10;
    double total_recall = 0.0;
    const uint32_t num_boundary_queries = 20;

    for (uint32_t q_idx = 0; q_idx < num_boundary_queries; ++q_idx) {
        std::vector<float> q(dim, 0.0f);
        q[0] = 0.001f * static_cast<float>(static_cast<int>(q_idx) - 10);
        for (uint32_t d = 1; d < dim; ++d) q[d] = d_noise(rng);

        auto golden_hits = oracle.Search(q, K);

        SearchResult res;
        POMAI_EXPECT_OK(engine.Search(q, K, &res));

        double recall = GoldenOracle::ComputeRecall(golden_hits, res.hits);
        if (q_idx == 0) {
            std::cout << "[DEBUG Q0] recall=" << recall << " res.hits.size=" << res.hits.size() << "\n";
            std::cout << "Golden top-10 IDs: ";
            for (auto& g : golden_hits) std::cout << g.id << "(" << g.raw_dist << ") ";
            std::cout << "\nPomai top-10 IDs: ";
            for (auto& h : res.hits) std::cout << h.id << "(" << h.score << ") ";
            std::cout << "\n";
        }
        total_recall += recall;
    }

    double avg_recall = total_recall / num_boundary_queries;
    std::cout << "[COMPASS BOUNDARY AUDIT] Average Recall@" << K << " on boundary = "
              << avg_recall << std::endl;

    // Must satisfy hard recall floor contract >= 0.95 even on difficult boundaries
    POMAI_EXPECT_TRUE(avg_recall >= 0.95);

    POMAI_EXPECT_OK(engine.Close());
}

} // namespace
