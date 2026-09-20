#ifndef POMAIDB_HNSW_H
#define POMAIDB_HNSW_H

#include <vector>
#include <random>
#include <cmath>
#include <functional>
#include <algorithm>
#include <psync/psync.h>
#include "utils/palloc_smart_ptr.h"

namespace pomai::hnsw {

/**
 * Minimalist, high-performance HNSW implementation.
 * Decoupled from external infrastructure, backed by PomaiDB native primitives.
 */

template <typename T, typename Compare = std::less<T>>
class PriorityQueue {
public:
    PriorityQueue() = default;
    explicit PriorityQueue(const Compare& comp) : comp_(comp) {}

    bool empty() const noexcept { return c_.empty(); }
    size_t size() const noexcept { return c_.size(); }
    const T& top() const { return c_.front(); }

    void push(const T& val) {
        c_.push_back(val);
        std::push_heap(c_.begin(), c_.end(), comp_);
    }

    void push(T&& val) {
        c_.push_back(std::move(val));
        std::push_heap(c_.begin(), c_.end(), comp_);
    }

    void pop() {
        std::pop_heap(c_.begin(), c_.end(), comp_);
        c_.pop_back();
    }

    void clear() noexcept { c_.clear(); }

private:
    std::vector<T> c_;
    Compare comp_{};
};

using storage_idx_t = int32_t;

struct NodeDist {
    float dist;
    storage_idx_t id;

    bool operator<(const NodeDist& other) const {
        return dist > other.dist; // Max-heap (farther elements first)
    }
};

struct NodeDistCloser {
    float dist;
    storage_idx_t id;

    bool operator<(const NodeDistCloser& other) const {
        return dist < other.dist; // Min-heap (closer elements first)
    }
};

class HNSW {
public:
    explicit HNSW(int M = 32, int ef_construction = 200);
    ~HNSW();

    // Configuration
    int M;
    int ef_construction;
    int ef_search;

    // Distance function: (id1, id2) -> float
    using DistanceComputer = std::function<float(storage_idx_t, storage_idx_t)>;
    // Query distance function: (query_vec_id, target_id) -> float
    using QueryDistanceComputer = std::function<float(storage_idx_t)>;

    /**
     * Adds a point to the graph. 
     * @param id The internal storage ID.
     * @param level The level to insert the point at (-1 for random).
     * @param qdis Distance computer for the new point against existing points.
     */
    void add_point(storage_idx_t id, int level, DistanceComputer& qdis);

    /**
     * Searches for the nearest neighbors of a query.
     * @param qdis Distance computer for the query against existing points.
     * @param k Top-k results.
     * @param ef Search expansion factor.
     * @param out_ids Output IDs.
     * @param out_dists Output distances.
     */
    void search(QueryDistanceComputer& qdis, int k, int ef, 
                std::vector<storage_idx_t>& out_ids, 
                std::vector<float>& out_dists) const;

    // Persistence
    void save(FILE* f) const;
    void load(FILE* f);

    int get_random_level();

private:
    struct LevelData {
        std::vector<int> cum_nneighbor_per_level;
        std::vector<int> levels;           // [ntotal]
        std::vector<size_t> offsets;      // [ntotal + 1]
        std::vector<storage_idx_t> neighbors; // flat pool
    } graph;

    storage_idx_t entry_point = -1;
    int max_level = -1;
    
    std::mt19937 rng;
    double level_mult;

    // Internal helpers
    void neighbor_range(storage_idx_t id, int level, size_t& begin, size_t& end) const;
    void shrink_neighbor_list(DistanceComputer& qdis, storage_idx_t cur, 
                             PriorityQueue<NodeDist>& candidates, int max_size);
    
    mutable psync::Mutex graph_mutex;
    std::vector<alloc::UniquePtr<psync::Mutex>> node_locks;
};

} // namespace pomai::hnsw

#endif // POMAIDB_HNSW_H
