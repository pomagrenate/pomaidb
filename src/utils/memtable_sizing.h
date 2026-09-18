// pomai/memtable_sizing.h — Dynamic memtable threshold calculation
//
// Calculates dynamic memtable flush threshold based on:
// - System available RAM (if detection succeeds)
// - User-configured percentage budget
// - Vector dimension (ensure minimum vectors per segment)
// - Min/max bounds
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include "options.h"
#include <cstdint>
#include <optional>

namespace pomai::utils {

/**
 * Calculate dynamic memtable flush threshold based on system resources and dimension.
 * 
 * Formula:
 * 1. If memtable_flush_threshold_mb > 0: use as manual override (backward compatibility)
 * 2. If memtable_flush_threshold_mb == 0: use dynamic sizing
 *    a. If available RAM detected:
 *       ram_threshold = clamp(budget_pct * available_ram, min_threshold, max_threshold)
 *    b. If OS detection fails: use fallback (64MB)
 * 3. Calculate dimension requirement:
 *    dim_floor = memtable_min_vectors_per_segment * dim * sizeof(float)
 * 4. Bounded dimension floor to prevent OOM:
 *    - If dim_floor <= max_threshold: use max(ram_threshold, dim_floor)
 *    - If dim_floor > max_threshold: dynamically downscale target vectors
 * 5. Final threshold = clamp(max(ram_threshold, adjusted_dim_floor), min_threshold, max_threshold)
 * 
 * Returns threshold in bytes.
 */
uint64_t CalculateDynamicMemtableThreshold(
    const pomai::DBOptions& options,
    uint32_t dimension,
    std::optional<uint64_t> available_ram_bytes = std::nullopt
);

} // namespace pomai::utils