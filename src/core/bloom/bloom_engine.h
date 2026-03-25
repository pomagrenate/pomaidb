#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pomai/status.h"

namespace pomai::core {

/**
 * BloomEngine — named persistent Bloom filters for approximate set membership.
 *
 * Each filter is identified by a uint64_t filter_id and backed by a bit array
 * of configurable size.  Uses a double-hashing scheme (Kirsch-Mitzenmacher)
 * to generate k independent hash positions from two base hashes.
 *
 * Typical use cases on edge devices:
 *   - Duplicate sensor-write prevention
 *   - Replay-attack guard
 *   - ANN candidate pre-filtering via MultiModalQuery::prefilter_bitset_id
 *
 * Thread safety: single-writer, single-reader.
 */
class BloomEngine {
public:
    /// @param max_keys_per_filter  Desired capacity per filter (used to size bit array).
    /// @param target_fpr           Target false-positive rate (default 1%).
    explicit BloomEngine(std::size_t max_keys_per_filter,
                         double target_fpr = 0.01);

    Status Open();
    Status Close();

    /// Add key to the named filter (creates filter on first use).
    Status Add(uint64_t filter_id, std::string_view key);

    /// Returns true in *out if key might be in the filter.
    /// Returns false with certainty when the key was never added.
    Status MightContain(uint64_t filter_id, std::string_view key, bool* out) const;

    /// Delete the entire filter (frees memory).
    Status Drop(uint64_t filter_id);

    /// Estimate current false-positive rate for a filter.
    Status EstimateFPR(uint64_t filter_id, double* out) const;

    /// Approximate number of distinct keys added to a filter.
    Status EstimateCount(uint64_t filter_id, std::size_t* out) const;

    void ForEach(const std::function<void(uint64_t filter_id, std::size_t num_bits, std::size_t num_elements)>& fn) const;

private:
    struct BloomFilter {
        std::vector<uint8_t> bits;   // Bit array (ceil(m/8) bytes)
        std::size_t          num_bits{0};
        std::size_t          num_hashes{0};
        std::size_t          num_elements{0};

        void SetBit(std::size_t pos);
        bool GetBit(std::size_t pos) const;
    };

    BloomFilter MakeFilter() const;
    static void HashPair(std::string_view key, uint64_t& h1, uint64_t& h2);

    std::size_t max_keys_per_filter_;
    std::size_t bits_per_filter_;   // m = ceil(-n * ln(p) / (ln2)^2)
    std::size_t num_hashes_;        // k = ceil((m/n) * ln2)

    std::unordered_map<uint64_t, BloomFilter> filters_;
};

} // namespace pomai::core