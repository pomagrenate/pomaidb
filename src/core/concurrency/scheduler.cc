#include "core/concurrency/scheduler.h"
#include <algorithm>
#include <numeric>

namespace pomai::core {

TaskScheduler::TaskScheduler() = default;
TaskScheduler::~TaskScheduler() = default;

void TaskScheduler::RegisterPeriodic(std::unique_ptr<DatabaseTask> task, std::chrono::milliseconds interval) {
    ScheduledTask st;
    st.task = std::move(task);
    st.interval = interval;
    st.next_run = std::chrono::steady_clock::now();
    tasks_.push_back(std::move(st));
}

void TaskScheduler::Poll() {
    auto now = std::chrono::steady_clock::now();
    for (auto& st : tasks_) {
        if (now >= st.next_run) {
            (void)st.task->Run();
            st.next_run = now + st.interval;
        }
    }
}

void TaskScheduler::PollBudget(uint32_t max_ops, uint32_t max_ms, bool deterministic) {
    if (tasks_.empty()) return;
    if (max_ops == 0) max_ops = 1;
    if (max_ms == 0) max_ms = 1;
    auto now = std::chrono::steady_clock::now();
    const auto deadline = now + std::chrono::milliseconds(max_ms);
    std::vector<size_t> due;
    due.reserve(tasks_.size());
    for (size_t i = 0; i < tasks_.size(); ++i) {
        if (now >= tasks_[i].next_run) due.push_back(i);
    }
    if (deterministic) {
        std::sort(due.begin(), due.end(), [&](size_t a, size_t b) {
            return tasks_[a].task->Name() < tasks_[b].task->Name();
        });
    }
    uint32_t ops = 0;
    for (size_t idx : due) {
        if (ops >= max_ops) break;
        if (std::chrono::steady_clock::now() >= deadline) break;
        auto& st = tasks_[idx];
        (void)st.task->Run();
        st.next_run = now + st.interval;
        ++ops;
    }
}

} // namespace pomai::core
