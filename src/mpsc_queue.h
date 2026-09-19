// mpsc_queue.h — Intrusive, lock-free MPSC queue backed by psync.
// Inspired by DragonflyDB's MPSCIntrusiveQueue.
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <psync/psync.h>

namespace pomai::core::concurrency {

/**
 * MPSCQueueEntry: Base class for objects that want to be queued.
 * Backed by psync::MPSCQueueEntry.
 */
using MPSCQueueEntry = psync::MPSCQueueEntry;

/**
 * MPSCIntrusiveQueue: A lock-free, many-to-one queue backed by psync.
 * Preserves PomaiDB's capitalized Push/Pop API for backward compatibility.
 */
template <typename T>
class MPSCIntrusiveQueue : public psync::MPSCIntrusiveQueue<T> {
public:
    void Push(MPSCQueueEntry* entry) noexcept { this->push(entry); }
    T* Pop() noexcept { return this->pop(); }
};

} // namespace pomai::core::concurrency
