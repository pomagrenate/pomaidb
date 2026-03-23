// runtime_stats.h — Runtime observability for edge tuning.
//
// Lightweight stats snapshot from a VectorRuntime.
// All fields are lock-free atomic reads — safe to call from any thread.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once
#include <cstdint>

namespace pomai::core {

struct RuntimeStats {
    std::uint32_t runtime_id{0};
    std::uint64_t ops_processed{0};       // Total commands processed by the worker
    std::uint64_t queue_depth{0};         // Current pending commands in mailbox
    std::uint64_t candidates_scanned{0};  // Candidates scanned in last query
    std::uint64_t memtable_entries{0};    // Current active MemTable size
    std::uint64_t mem_committed{0};       // Runtime heap committed bytes
    std::uint64_t mem_used{0};            // Runtime heap used bytes
    std::uint64_t bytes_written_wal{0};   // WAL bytes written
    std::uint64_t bytes_written_segments{0}; // Segment bytes written (freeze+compact)
    std::uint64_t bytes_written_total{0};
};

} // namespace pomai::core
