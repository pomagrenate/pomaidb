#pragma once

// ptask_parallel.h
// High-level parallel algorithms (parallel_for, parallel_reduce, parallel_invoke).
// Part of ptask - High-performance work-stealing task scheduler.

#include "ptask_pool.h"
#include <vector>
#include <algorithm>

namespace ptask {

// ============================================================================
// PARALLEL FOR
// ============================================================================

template <typename Index, typename Func>
void parallel_for(ThreadPool& pool, Index first, Index last, Func&& func, size_t chunk_size = 0) {
    if (first >= last) return;

    size_t total = static_cast<size_t>(last - first);
    if (chunk_size == 0) {
        chunk_size = std::max<size_t>(1, total / (pool.worker_count() * 4));
    }

    std::vector<TaskFuture<void>> futures;
    futures.reserve((total + chunk_size - 1) / chunk_size);

    for (Index i = first; i < last; i = static_cast<Index>(i + chunk_size)) {
        Index end = static_cast<Index>(std::min<size_t>(last, i + chunk_size));
        futures.emplace_back(pool.submit([i, end, &func]() {
            for (Index j = i; j < end; ++j) {
                func(j);
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }
}

// Overload with default global pool or convenience
template <typename Index, typename Func>
void parallel_for(Index first, Index last, Func&& func, size_t chunk_size = 0) {
    static ThreadPool default_pool;
    parallel_for(default_pool, first, last, std::forward<Func>(func), chunk_size);
}

// ============================================================================
// PARALLEL REDUCE
// ============================================================================

template <typename Index, typename T, typename ReduceOp, typename MapFunc>
T parallel_reduce(ThreadPool& pool, Index first, Index last, T init, ReduceOp&& reduce, MapFunc&& map, size_t chunk_size = 0) {
    if (first >= last) return init;

    size_t total = static_cast<size_t>(last - first);
    if (chunk_size == 0) {
        chunk_size = std::max<size_t>(1, total / (pool.worker_count() * 4));
    }

    std::vector<TaskFuture<T>> futures;
    futures.reserve((total + chunk_size - 1) / chunk_size);

    for (Index i = first; i < last; i = static_cast<Index>(i + chunk_size)) {
        Index end = static_cast<Index>(std::min<size_t>(last, i + chunk_size));
        futures.emplace_back(pool.submit([i, end, init, &reduce, &map]() -> T {
            T partial = init;
            for (Index j = i; j < end; ++j) {
                partial = reduce(partial, map(j));
            }
            return partial;
        }));
    }

    T result = init;
    for (auto& f : futures) {
        result = reduce(result, f.get());
    }
    return result;
}

// ============================================================================
// PARALLEL INVOKE
// ============================================================================

template <typename ThreadPoolType, typename F1, typename... Fs>
void parallel_invoke(ThreadPoolType& pool, F1&& f1, Fs&&... fs) {
    std::vector<TaskFuture<void>> futures;
    (futures.emplace_back(pool.submit(std::forward<Fs>(fs))), ...);
    
    // Execute first task synchronously on calling thread
    f1();

    for (auto& f : futures) {
        f.get();
    }
}

} // namespace ptask
