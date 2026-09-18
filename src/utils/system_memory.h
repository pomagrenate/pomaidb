// pomai/system_memory.h — Cross-platform system memory detection
//
// Provides system memory information for dynamic memtable sizing.
// Includes cgroup v1/v2 detection for Linux containers to prevent OOM.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstdint>
#include <optional>

namespace pomai::utils {

struct SystemMemoryInfo {
    uint64_t total_bytes;      // Total physical RAM
    uint64_t available_bytes;  // Available physical RAM (respecting cgroup limits on Linux)
};

/**
 * Get system memory information (cross-platform).
 * 
 * Platform-specific behavior:
 * - Windows: Uses GlobalMemoryStatusEx (ullAvailPhys / ullTotalPhys)
 * - Linux: Reads /proc/meminfo for host RAM, checks cgroup v1/v2 limits,
 *          returns min(host_available, cgroup_limit) to prevent container OOM
 * - macOS: Uses sysctl hw.memsize for total, estimates available
 * 
 * Returns nullopt on failure (use conservative fallback defaults: 1GB total, 512MB available).
 */
std::optional<SystemMemoryInfo> GetSystemMemoryInfo();

} // namespace pomai::utils