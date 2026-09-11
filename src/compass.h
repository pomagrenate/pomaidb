// pomai/compass.h — Compass: Spatial routing layer over Locule anchors
//
// In the Pomegranate Engine:
// - Compass maintains spatial LoculeAnchors for active Locules.
// - Orient: calculates distances from query vector to all Locule centroids and ranks them.
// - Peel: applies geometric bounding-sphere pruning to discard Locules that cannot beat current top-k bounds.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "locule.h"
#include "types.h"

namespace pomai::routing {

struct OrientedLocule {
    std::shared_ptr<storage::Locule> locule;
    float distance_to_centroid{0.0f};
    float min_possible_distance{0.0f}; // lower bound for query to any point in the locule
};

class Compass {
public:
    explicit Compass(MetricType metric = MetricType::kL2);
    ~Compass();

    void UpdateLocules(std::vector<std::shared_ptr<storage::Locule>> locules);

    /**
     * Orient: Evaluates query proximity against all active Locule centroids.
     * Returns candidate locules sorted by distance (closest first).
     * @param nprobe Max number of locules to return (0 = return all).
     */
    std::vector<OrientedLocule> Orient(std::span<const float> query, uint32_t nprobe = 0) const;

    /**
     * Peel: Prunes candidate locules whose minimum possible distance cannot beat worst_distance.
     * @param candidates Candidates returned by Orient.
     * @param worst_distance Current distance threshold of the k-th candidate in the top-k heap.
     * @return Pruned list of candidates that can potentially contain a better vector.
     */
    std::vector<OrientedLocule> Peel(const std::vector<OrientedLocule>& candidates,
                                     float worst_distance) const;

    [[nodiscard]] size_t LoculeCount() const noexcept { return locules_.size(); }
    [[nodiscard]] MetricType metric() const noexcept { return metric_; }

private:
    MetricType metric_{MetricType::kL2};
    std::vector<std::shared_ptr<storage::Locule>> locules_;
};

} // namespace pomai::routing
