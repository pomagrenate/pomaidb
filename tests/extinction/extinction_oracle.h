// tests/extinction/extinction_oracle.h
// Extinction Event: Pure Independent FP64 Ground-Truth Reference Model
//
// Strictly decoupled from PomaiDB kernels, headers, and implementations.
// Authority for all logical states, exact metric scores, filtering, and tie-breaking.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pomai::extinction {

enum class OracleMetric : uint8_t {
    kL2 = 0,
    kInnerProduct = 1,
    kCosine = 2,
};

struct OracleMetadata {
    std::string tenant;
    std::string device_id;
    std::string location_id;
    uint64_t timestamp{0};
    uint64_t lsn{0};
    std::string payload;

    bool operator==(const OracleMetadata& o) const {
        return tenant == o.tenant && device_id == o.device_id &&
               location_id == o.location_id && timestamp == o.timestamp &&
               lsn == o.lsn && payload == o.payload;
    }
};

struct OracleFilter {
    std::string field;
    std::string value;
    uint64_t min_ts{0};
    uint64_t max_ts{0};

    bool Matches(const OracleMetadata& meta) const {
        if (field == "tenant" && meta.tenant != value) return false;
        if (field == "device_id" && meta.device_id != value) return false;
        if (field == "location_id" && meta.location_id != value) return false;
        if (field == "timestamp") {
            if (min_ts > 0 && meta.timestamp < min_ts) return false;
            if (max_ts > 0 && meta.timestamp > max_ts) return false;
        }
        return true;
    }
};

struct OracleHit {
    uint64_t id{0};
    double score{0.0};       // Canonical ranking score (higher is better)
    double raw_distance{0.0}; // Underlying distance (e.g. L2 squared or cosine dist)

    bool operator==(const OracleHit& o) const {
        return id == o.id && std::abs(score - o.score) < 1e-4;
    }
};

class ExtinctionOracle {
public:
    struct Record {
        uint64_t id{0};
        std::vector<float> vec;
        OracleMetadata meta;
        bool is_deleted{false};
        uint64_t version{0};
    };

    explicit ExtinctionOracle(uint32_t dim, OracleMetric metric = OracleMetric::kL2)
        : dim_(dim), metric_(metric) {}

    void Put(uint64_t id, const std::vector<float>& vec, const OracleMetadata& meta = {}) {
        auto& r = table_[id];
        r.id = id;
        r.vec = vec;
        r.meta = meta;
        r.is_deleted = false;
        r.version = ++version_counter_;
    }

    bool Delete(uint64_t id) {
        auto it = table_.find(id);
        if (it == table_.end() || it->second.is_deleted) {
            return false;
        }
        it->second.is_deleted = true;
        it->second.version = ++version_counter_;
        return true;
    }

    bool Get(uint64_t id, std::vector<float>* out_vec, OracleMetadata* out_meta = nullptr) const {
        auto it = table_.find(id);
        if (it == table_.end() || it->second.is_deleted) {
            return false;
        }
        if (out_vec) *out_vec = it->second.vec;
        if (out_meta) *out_meta = it->second.meta;
        return true;
    }

    bool Contains(uint64_t id) const {
        auto it = table_.find(id);
        return (it != table_.end() && !it->second.is_deleted);
    }

    size_t LiveCount() const {
        size_t count = 0;
        for (const auto& [id, r] : table_) {
            if (!r.is_deleted) count++;
        }
        return count;
    }

    size_t TombstoneCount() const {
        size_t count = 0;
        for (const auto& [id, r] : table_) {
            if (r.is_deleted) count++;
        }
        return count;
    }

    // Exhaustive Double-Precision Brute-Force Search
    std::vector<OracleHit> Search(
        std::span<const float> query,
        uint32_t topk,
        const std::vector<OracleFilter>& filters = {}) const {

        if (topk == 0 || query.size() != dim_) {
            return {};
        }

        std::vector<OracleHit> candidates;
        candidates.reserve(table_.size());

        for (const auto& [id, r] : table_) {
            if (r.is_deleted) continue;
            if (r.vec.size() != dim_) continue;

            // Apply all filters
            bool matched = true;
            for (const auto& f : filters) {
                if (!f.Matches(r.meta)) {
                    matched = false;
                    break;
                }
            }
            if (!matched) continue;

            double score = 0.0;
            double raw_dist = 0.0;

            if (metric_ == OracleMetric::kL2) {
                double l2sq = 0.0;
                for (size_t d = 0; d < dim_; ++d) {
                    double diff = static_cast<double>(query[d]) - static_cast<double>(r.vec[d]);
                    l2sq += diff * diff;
                }
                raw_dist = l2sq;
                score = -l2sq; // Higher is better (closer to 0)
            } else if (metric_ == OracleMetric::kInnerProduct) {
                double dot = 0.0;
                for (size_t d = 0; d < dim_; ++d) {
                    dot += static_cast<double>(query[d]) * static_cast<double>(r.vec[d]);
                }
                raw_dist = dot;
                score = dot;
            } else if (metric_ == OracleMetric::kCosine) {
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
                if (norm_q <= 1e-18 || norm_v <= 1e-18) {
                    score = 0.0;
                    raw_dist = 1.0;
                } else {
                    double sim = dot / (std::sqrt(norm_q) * std::sqrt(norm_v));
                    sim = std::clamp(sim, -1.0, 1.0);
                    score = sim;
                    raw_dist = 1.0 - sim;
                }
            }

            candidates.push_back({id, score, raw_dist});
        }

        // Canonical deterministic ranking:
        // In a 32-bit vector engine, scores equal in float precision must be tie-broken by ID ascending.
        std::sort(candidates.begin(), candidates.end(),
                  [](const OracleHit& a, const OracleHit& b) {
                      float fa = static_cast<float>(a.score);
                      float fb = static_cast<float>(b.score);
                      if (fa != fb) {
                          return fa > fb;
                      }
                      return a.id < b.id;
                  });

        if (candidates.size() > topk) {
            candidates.resize(topk);
        }
        return candidates;
    }

    // Exact recall calculation with tie-score equivalence
    template <typename EngineHitType>
    static double ComputeRecall(const std::vector<OracleHit>& truth,
                                const std::vector<EngineHitType>& predicted) {
        if (truth.empty()) return 1.0;
        size_t k = std::min(truth.size(), predicted.size());
        if (k == 0) return 0.0;

        std::unordered_set<uint64_t> gt_set;
        for (size_t i = 0; i < k; ++i) {
            gt_set.insert(truth[i].id);
        }

        float cutoff_score = static_cast<float>(truth[k - 1].score);
        float score_tolerance = std::max(1e-4f, 1e-5f * std::abs(cutoff_score));
        std::unordered_set<uint64_t> matched_ids;
        size_t matches = 0;
        for (size_t i = 0; i < k; ++i) {
            if (gt_set.count(predicted[i].id)) {
                if (matched_ids.insert(predicted[i].id).second) {
                    matches++;
                }
            } else if (static_cast<float>(predicted[i].score) >= cutoff_score - score_tolerance) {
                // Ties at or above the k-th threshold within float epsilon: mathematically equivalent neighbor
                if (matched_ids.insert(predicted[i].id).second) {
                    matches++;
                }
            }
        }
        return static_cast<double>(matches) / static_cast<double>(k);
    }

    void Clear() {
        table_.clear();
        version_counter_ = 0;
    }

    uint32_t dimension() const noexcept { return dim_; }
    OracleMetric metric() const noexcept { return metric_; }
    const std::unordered_map<uint64_t, Record>& table() const noexcept { return table_; }

private:
    uint32_t dim_{0};
    OracleMetric metric_{OracleMetric::kL2};
    std::unordered_map<uint64_t, Record> table_;
    uint64_t version_counter_{0};
};

} // namespace pomai::extinction
