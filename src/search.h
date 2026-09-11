#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "options.h"
#include "types.h"

namespace pomai
{
    struct SearchHit
    {
        VectorId id = 0;
        float score = 0.0f; // higher is better
    };

    /**
     * @brief Interface for collecting search results without intermediate allocations.
     * Concrete implementations can write to std::vector, a pre-allocated pool, or directly to a C-buffer.
     */
    class SearchHitSink {
    public:
        virtual ~SearchHitSink() = default;
        /** Collect one search hit. Implementation may decide to keep or discard based on top-K or filters. */
        virtual void Push(VectorId id, float score) = 0;
    };

    /** @brief Priority queue comparator for min-heap (lowest score at top). */
    struct WorseHit {
        bool operator()(const SearchHit& a, const SearchHit& b) const {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            return a.id > b.id;
        }
    };

    /** @brief Check if 'a' is a better hit than 'b' (higher score, then lower ID). */
    inline bool IsBetterHit(const SearchHit& a, const SearchHit& b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.id < b.id;
    }

    struct ShardError
    {
        uint32_t shard_id;
        std::string message;
    };

    struct SemanticPointer {
        const void* raw_data_ptr = nullptr;
        uint32_t dim = 0;
        float quant_min = 0.0f;
        float quant_inv_scale = 0.0f;
        int quant_type = 0; // 0=None, 1=SQ8, 2=FP16
        uint64_t session_id = 0;
    };

    struct SearchResult
    {
        std::vector<SearchHit> hits;
        std::vector<ShardError> errors; // Partial failures
        uint32_t routed_shards_count = 0;
        uint32_t total_shards_count = 0;
        uint32_t pruned_shards_count = 0;
        uint32_t routing_probe_centroids = 0;
        uint64_t routed_buckets_count = 0; // Candidate/bucket count when routing enabled.

        std::vector<SemanticPointer> zero_copy_pointers;
        uint64_t zero_copy_session_id = 0;

        void Clear() {
            hits.clear();
            errors.clear();
            routed_shards_count = 0;
            total_shards_count = 0;
            pruned_shards_count = 0;
            routing_probe_centroids = 0;
            routed_buckets_count = 0;
            zero_copy_pointers.clear();
            zero_copy_session_id = 0;
        }
    };

} // namespace pomai

