#pragma once

// ptask_pool.h
// Work-stealing thread pool with hardware-clamped worker threads.
// Decouples M tasks from N OS threads to eliminate over-subscription.
// Part of ptask - High-performance work-stealing task scheduler.

#include "ptask_types.h"
#include "ptask_deque.h"
#include "ptask_future.h"
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <random>
#include <chrono>

namespace ptask {

class ThreadPool {
public:
    // Construct thread pool.
    // num_threads = 0 defaults to hardware_threads().
    // Clamped strictly to [1, max_multiplier * hardware_threads()] to prevent CPU thrashing.
    explicit ThreadPool(uint32_t num_threads = 0, float max_multiplier = 1.5f)
        : worker_count_(clamp_workers(num_threads, max_multiplier)),
          stopped_(false),
          active_workers_(0) {
        
        deques_.reserve(worker_count_);
        for (uint32_t i = 0; i < worker_count_; ++i) {
            deques_.emplace_back(std::make_unique<WorkStealingDeque<Task*>>(1024));
        }

        workers_.reserve(worker_count_);
        for (uint32_t i = 0; i < worker_count_; ++i) {
            workers_.emplace_back([this, i]() {
                worker_loop(i);
            });
        }
    }

    ~ThreadPool() {
        shutdown();
    }

    // Disable copy and move
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    uint32_t worker_count() const noexcept {
        return worker_count_;
    }

    // Fire-and-forget task submission
    template <typename F>
    void spawn(F&& f) {
        Task* task = new Task(std::forward<F>(f));
        enqueue_task(task);
    }

    // Submit task returning TaskFuture<R>
    template <typename F>
    auto submit(F&& f) -> TaskFuture<std::invoke_result_t<F>> {
        using R = std::invoke_result_t<F>;
        auto state = std::make_shared<FutureState<R>>();
        
        Task* task = new Task([state, func = std::forward<F>(f)]() mutable {
            try {
                if constexpr (std::is_void_v<R>) {
                    func();
                    state->set_value();
                } else {
                    state->set_value(func());
                }
            } catch (...) {
                state->set_exception(std::current_exception());
            }
        });

        enqueue_task(task);
        return TaskFuture<R>(state, this);
    }

    // Cooperatively assist with executing tasks until a predicate becomes true.
    // Completely eliminates thread stalling/deadlocks when waiting on child tasks!
    template <typename Predicate>
    void help_work_until(Predicate&& pred) {
        while (!pred()) {
            if (!execute_one_task()) {
                cpu_pause();
            }
        }
    }

    // Wait until all queued tasks are processed
    void wait_idle() {
        help_work_until([this]() {
            if (in_flight_tasks_.load(std::memory_order_acquire) == 0) {
                return true;
            }
            return false;
        });
    }

    // Shut down workers gracefully
    void shutdown() {
        bool expected = false;
        if (stopped_.compare_exchange_strong(expected, true)) {
            wake_all();
            for (auto& w : workers_) {
                if (w.joinable()) {
                    w.join();
                }
            }
            // Drain any leftover tasks
            drain_injector();
            for (auto& dq : deques_) {
                while (auto t = dq->pop()) {
                    delete *t;
                }
            }
        }
    }

private:
    static inline thread_local uint32_t t_worker_id{UINT32_MAX};

    void enqueue_task(Task* task) {
        in_flight_tasks_.fetch_add(1, std::memory_order_relaxed);

        if (t_worker_id < worker_count_) {
            // Submitted from within worker thread: push to local deque without lock!
            deques_[t_worker_id]->push(task);
        } else {
            // Submitted from external thread: push to injector queue
            {
                std::lock_guard<std::mutex> lock(injector_mutex_);
                injector_queue_.push_back(task);
            }
        }
        wake_one();
    }

    bool execute_one_task() {
        Task* task = nullptr;

        // 1. Try local deque if called from worker
        if (t_worker_id < worker_count_) {
            if (auto opt = deques_[t_worker_id]->pop()) {
                task = *opt;
            }
        }

        // 2. Try injector queue
        if (!task) {
            std::lock_guard<std::mutex> lock(injector_mutex_);
            if (!injector_queue_.empty()) {
                task = injector_queue_.back();
                injector_queue_.pop_back();
            }
        }

        // 3. Try stealing from other workers
        if (!task && worker_count_ > 1) {
            uint32_t start_idx = (t_worker_id < worker_count_) ? t_worker_id + 1 : 0;
            for (uint32_t i = 0; i < worker_count_; ++i) {
                uint32_t victim = (start_idx + i) % worker_count_;
                if (victim == t_worker_id) continue;

                if (auto opt = deques_[victim]->steal()) {
                    task = *opt;
                    break;
                }
            }
        }

        // 4. Run task if found
        if (task) {
            (*task)();
            delete task;
            in_flight_tasks_.fetch_sub(1, std::memory_order_release);
            return true;
        }

        return false;
    }

    void worker_loop(uint32_t id) {
        t_worker_id = id;

        while (!stopped_.load(std::memory_order_relaxed)) {
            if (execute_one_task()) {
                continue;
            }

            // Adaptive backoff before sleeping
            bool found_work = false;
            for (int spin = 0; spin < 64; ++spin) {
                cpu_pause();
                if (execute_one_task()) {
                    found_work = true;
                    break;
                }
            }
            if (found_work) continue;

            // Park thread on condition variable
            std::unique_lock<std::mutex> lock(cv_mutex_);
            if (!stopped_.load(std::memory_order_relaxed) && 
                in_flight_tasks_.load(std::memory_order_relaxed) == 0) {
                cv_.wait_for(lock, std::chrono::milliseconds(5));
            }
        }
    }

    void wake_one() {
        cv_.notify_one();
    }

    void wake_all() {
        cv_.notify_all();
    }

    void drain_injector() {
        std::lock_guard<std::mutex> lock(injector_mutex_);
        while (!injector_queue_.empty()) {
            delete injector_queue_.back();
            injector_queue_.pop_back();
        }
    }

    uint32_t worker_count_;
    std::atomic<bool> stopped_;
    std::atomic<uint32_t> active_workers_;
    std::atomic<int64_t> in_flight_tasks_{0};

    std::vector<std::unique_ptr<WorkStealingDeque<Task*>>> deques_;
    std::vector<std::thread> workers_;

    std::mutex injector_mutex_;
    std::vector<Task*> injector_queue_;

    std::mutex cv_mutex_;
    std::condition_variable cv_;
};

// Out-of-line definition for TaskFuture::wait
template <typename T>
inline void TaskFuture<T>::wait() const {
    if (is_ready()) return;

    if (pool_) {
        pool_->help_work_until([this]() { return is_ready(); });
    } else {
        while (!is_ready()) {
            cpu_pause();
        }
    }
}

} // namespace ptask
