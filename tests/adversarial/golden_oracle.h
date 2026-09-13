// tests/adversarial/golden_oracle.h
// Independent Golden Reference Implementation (Phase 28)
// Completely independent from PomaiDB SIMD and distance kernels.
// Uses strict IEEE-754 double precision accumulation for ground truth calculation.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pomai::adversarial {

struct OracleHit {
    uint64_t id;
    double score; // canonical score: higher is better
    double raw_dist; // original distance (L2 or Cosine distance)
};

class GoldenOracle {
public:
    enum class Metric {
        kL2,
        kInnerProduct,
        kCosine
    };

    struct Record {
        uint64_t id;
        std::vector<float> vec;
        std::string device_id;
        std::string location_id;
        bool deleted{false};
    };

    explicit GoldenOracle(uint32_t dim, Metric metric = Metric::kL2)
        : dim_(dim), metric_(metric) {}

    void Put(uint64_t id, const std::vector<float>& vec,
             const std::string& device_id = "",
             const std::string& location_id = "") {
        records_[id] = Record{id, vec, device_id, location_id, false};
    }

    void Delete(uint64_t id) {
        auto it = records_.find(id);
        if (it != records_.end()) {
            it->second.deleted = true;
        }
    }

    bool IsDeleted(uint64_t id) const {
        auto it = records_.find(id);
        if (it == records_.end()) return true;
        return it->second.deleted;
    }

    size_t LiveCount() const {
        size_t count = 0;
        for (const auto& [id, r] : records_) {
            if (!r.deleted) count++;
        }
        return count;
    }

    // Exact Ground Truth Top-K using double precision arithmetic
    std::vector<OracleHit> Search(
        std::span<const float> query,
        uint32_t topk,
        const std::string& filter_device_id = "",
        const std::string& filter_location_id = "") const {

        std::vector<OracleHit> candidates;
        candidates.reserve(records_.size());

        for (const auto& [id, r] : records_) {
            if (r.deleted) continue;
            if (!filter_device_id.empty() && r.device_id != filter_device_id) continue;
            if (!filter_location_id.empty() && r.location_id != filter_location_id) continue;
            if (r.vec.size() != dim_) continue;

            double score = 0.0;
            double raw_dist = 0.0;

            if (metric_ == Metric::kL2) {
                double sum_sq = 0.0;
                for (size_t d = 0; d < dim_; ++d) {
                    double diff = static_cast<double>(query[d]) - static_cast<double>(r.vec[d]);
                    sum_sq += diff * diff;
                }
                raw_dist = sum_sq;
                score = -sum_sq; // Higher score is better: closer to 0
            } else if (metric_ == Metric::kInnerProduct) {
                double dot = 0.0;
                for (size_t d = 0; d < dim_; ++d) {
                    dot += static_cast<double>(query[d]) * static_cast<double>(r.vec[d]);
                }
                raw_dist = dot;
                score = dot;
            } else if (metric_ == Metric::kCosine) {
                double dot = 0.0;
                double norm_q = 0.0;
                double norm_v = 0.0;
                for (size_t d = 0; d < dim_; ++d) {
                    double qd = static_cast<double>(query[d]);
                    double vd = static_cast<double>(r.vec[d]);
                    dot += qd * vd;
                    norm_q += qd * qd;
                    norm_v += vd * vd;
                }
                double denom = std::sqrt(norm_q) * std::sqrt(norm_v);
                double sim = (denom > 1e-15) ? (dot / denom) : 0.0;
                raw_dist = 1.0 - sim;
                score = sim;
            }

            candidates.push_back(OracleHit{id, score, raw_dist});
        }

        // Deterministic Golden Sort: score descending, ID ascending
        std::sort(candidates.begin(), candidates.end(),
                  [](const OracleHit& a, const OracleHit& b) {
                      if (std::abs(a.score - b.score) > 1e-7) {
                          return a.score > b.score;
                      }
                      return a.id < b.id; // Deterministic tie-breaker
                  });

        if (candidates.size() > topk) {
            candidates.resize(topk);
        }
        return candidates;
    }

    // Exact Recall calculation comparing predicted IDs against golden truth
    static double ComputeRecall(const std::vector<uint64_t>& predicted,
                                const std::vector<OracleHit>& ground_truth,
                                uint32_t k) {
        if (ground_truth.empty() || k == 0) return 1.0;
        size_t check_k = std::min({static_cast<size_t>(k), predicted.size(), ground_truth.size()});
        if (check_k == 0) return 0.0;

        std::unordered_set<uint64_t> gt_set;
        for (size_t i = 0; i < check_k; ++i) {
            gt_set.insert(ground_truth[i].id);
        }

        size_t matches = 0;
        for (size_t i = 0; i < check_k; ++i) {
            if (gt_set.count(predicted[i])) {
                matches++;
            }
        }
        return static_cast<double>(matches) / static_cast<double>(check_k);
    }

    template <typename HitType>
    static double ComputeRecall(const std::vector<OracleHit>& ground_truth,
                                const std::vector<HitType>& predicted) {
        if (ground_truth.empty()) return 1.0;
        size_t check_k = std::min(predicted.size(), ground_truth.size());
        if (check_k == 0) return 0.0;

        std::unordered_set<uint64_t> gt_set;
        for (size_t i = 0; i < check_k; ++i) {
            gt_set.insert(ground_truth[i].id);
        }

        size_t matches = 0;
        for (size_t i = 0; i < check_k; ++i) {
            if (gt_set.count(predicted[i].id)) {
                matches++;
            }
        }
        return static_cast<double>(matches) / static_cast<double>(check_k);
    }

private:
    uint32_t dim_;
    Metric metric_;
    std::unordered_map<uint64_t, Record> records_;
};

} // namespace pomai::adversarial
