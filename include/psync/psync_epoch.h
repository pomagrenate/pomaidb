#pragma once

// psync_epoch.h
// Lightweight Epoch-Based Reclamation (EBR) for lock-free read-side memory safety
// Target: Linux / macOS / Windows across x86-64, ARM64, RISC-V

#include "psync_platform.h"
#include "psync_cache_padded.h"
#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>
#include <mutex>

namespace psync {

/**
 * EpochManager: Coordinates safe deferred memory reclamation.
 *
 * Solves the reader-use-after-free problem in lock-free and snapshot data structures
 * (such as deleting/unmapping old locules or nodes while reader threads are actively
 * traversing them without locks).
 */
class EpochManager {
public:
    static constexpr uint64_t kInactiveEpoch = UINT64_MAX;
    static constexpr size_t kMaxThreads = 128;

    struct RetiredNode {
        void* ptr;
        void (*deleter)(void*);
        uint64_t epoch;
        RetiredNode* next;
    };

    struct ThreadState {
        CachePadded<std::atomic<uint64_t>> active_epoch{kInactiveEpoch};
        CachePadded<std::atomic<bool>> in_use{false};
    };

    EpochManager()
        : global_epoch_(1),
          retired_head_(nullptr)
    {}

    ~EpochManager() {
        reclaim_all();
    }

    EpochManager(const EpochManager&) = delete;
    EpochManager& operator=(const EpochManager&) = delete;

    /**
     * Register the current thread to obtain an epoch slot index.
     */
    size_t register_thread() {
        for (size_t i = 0; i < kMaxThreads; ++i) {
            bool expected = false;
            if (threads_[i].in_use.value.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel
            )) {
                return i;
            }
        }
        return 0; // Fallback to slot 0
    }

    void unregister_thread(size_t slot) {
        if (slot < kMaxThreads) {
            threads_[slot].active_epoch.value.store(kInactiveEpoch, std::memory_order_release);
            threads_[slot].in_use.value.store(false, std::memory_order_release);
        }
    }

    /**
     * Enter critical read section for a registered thread slot.
     */
    void enter(size_t slot) noexcept {
        if (slot < kMaxThreads) {
            uint64_t e = global_epoch_.load(std::memory_order_relaxed);
            threads_[slot].active_epoch.value.store(e, std::memory_order_release);
            compiler_fence();
        }
    }

    /**
     * Exit critical read section.
     */
    void exit(size_t slot) noexcept {
        if (slot < kMaxThreads) {
            compiler_fence();
            threads_[slot].active_epoch.value.store(kInactiveEpoch, std::memory_order_release);
        }
    }

    /**
     * Retire an allocated pointer to be deleted once all active readers advance.
     */
    template <typename T, typename Deleter>
    void retire(T* ptr, Deleter&& deleter) {
        auto node_deleter = [](void* p) {
            Deleter()(static_cast<T*>(p));
        };
        retire_internal(ptr, node_deleter);
    }

    template <typename T>
    void retire(T* ptr) {
        retire_internal(ptr, [](void* p) {
            delete static_cast<T*>(p);
        });
    }

    /**
     * Attempt to advance global epoch and reclaim safely retired memory.
     */
    void try_reclaim() {
        uint64_t current = global_epoch_.load(std::memory_order_acquire);

        // Check if any active thread is still in an older epoch
        uint64_t min_epoch = current;
        for (size_t i = 0; i < kMaxThreads; ++i) {
            if (threads_[i].in_use.value.load(std::memory_order_relaxed)) {
                uint64_t e = threads_[i].active_epoch.value.load(std::memory_order_acquire);
                if (e != kInactiveEpoch && e < min_epoch) {
                    min_epoch = e;
                }
            }
        }

        // If all threads caught up, advance epoch
        if (min_epoch == current) {
            global_epoch_.fetch_add(1, std::memory_order_release);
        }

        // Reclaim any nodes retired strictly before min_epoch
        RetiredNode* curr = nullptr;
        {
            std::lock_guard<std::mutex> lock(reclaim_mu_);
            curr = retired_head_.load(std::memory_order_acquire);
            RetiredNode* prev = nullptr;

            while (curr) {
                if (curr->epoch < min_epoch) {
                    // Safe to delete
                    RetiredNode* to_del = curr;
                    if (prev) {
                        prev->next = curr->next;
                    } else {
                        retired_head_.store(curr->next, std::memory_order_release);
                    }
                    curr = curr->next;

                    to_del->deleter(to_del->ptr);
                    delete to_del;
                } else {
                    prev = curr;
                    curr = curr->next;
                }
            }
        }
    }

    uint64_t current_epoch() const noexcept {
        return global_epoch_.load(std::memory_order_relaxed);
    }

private:
    void retire_internal(void* ptr, void (*deleter)(void*)) {
        auto* node = new RetiredNode{
            ptr,
            deleter,
            global_epoch_.load(std::memory_order_relaxed),
            nullptr
        };

        std::lock_guard<std::mutex> lock(reclaim_mu_);
        node->next = retired_head_.load(std::memory_order_relaxed);
        retired_head_.store(node, std::memory_order_release);
    }

    void reclaim_all() {
        std::lock_guard<std::mutex> lock(reclaim_mu_);
        RetiredNode* curr = retired_head_.load(std::memory_order_relaxed);
        while (curr) {
            RetiredNode* next = curr->next;
            curr->deleter(curr->ptr);
            delete curr;
            curr = next;
        }
        retired_head_.store(nullptr, std::memory_order_relaxed);
    }

    std::atomic<uint64_t> global_epoch_;
    ThreadState threads_[kMaxThreads];
    std::atomic<RetiredNode*> retired_head_;
    std::mutex reclaim_mu_;
};

/**
 * EpochGuard: RAII guard for reading lock-free data structures safely under EBR.
 */
class EpochGuard {
public:
    EpochGuard(EpochManager& manager, size_t slot)
        : manager_(manager), slot_(slot)
    {
        manager_.enter(slot_);
    }

    ~EpochGuard() {
        manager_.exit(slot_);
    }

    EpochGuard(const EpochGuard&) = delete;
    EpochGuard& operator=(const EpochGuard&) = delete;

private:
    EpochManager& manager_;
    const size_t slot_;
};

} // namespace psync
