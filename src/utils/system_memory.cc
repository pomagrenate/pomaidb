// pomai/system_memory.cc — Cross-platform system memory detection implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "system_memory.h"

#include <fstream>
#include <sstream>

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <cstdlib>
#include <cstring>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

namespace pomai::utils {

namespace {

// Helper to read a single integer value from a file (Linux-specific)
std::optional<uint64_t> ReadUint64FromFile(const char* path) {
#if defined(__linux__)
    std::ifstream file(path);
    if (!file.is_open()) {
        return std::nullopt;
    }
    uint64_t value;
    if (!(file >> value)) {
        return std::nullopt;
    }
    return value;
#else
    (void)path;
    return std::nullopt;
#endif
}

// Linux cgroup v2 detection: /sys/fs/cgroup/memory.max
std::optional<uint64_t> GetCgroupV2MemoryLimit() {
#if defined(__linux__)
    auto limit = ReadUint64FromFile("/sys/fs/cgroup/memory.max");
    if (limit && *limit != UINT64_MAX) {  // "max" in cgroup v2 means unlimited
        return limit;
    }
    return std::nullopt;
#else
    return std::nullopt;
#endif
}

// Linux cgroup v1 detection: /sys/fs/cgroup/memory/memory.limit_in_bytes
std::optional<uint64_t> GetCgroupV1MemoryLimit() {
#if defined(__linux__)
    auto limit = ReadUint64FromFile("/sys/fs/cgroup/memory/memory.limit_in_bytes");
    if (limit && *limit != UINT64_MAX) {  // Some cgroup v1 setups use UINT64_MAX for unlimited
        return limit;
    }
    return std::nullopt;
#else
    return std::nullopt;
#endif
}

// Get Linux cgroup memory limit (v1 or v2)
std::optional<uint64_t> GetCgroupMemoryLimit() {
#if defined(__linux__)
    // Try cgroup v2 first
    auto v2_limit = GetCgroupV2MemoryLimit();
    if (v2_limit) {
        return v2_limit;
    }
    
    // Fall back to cgroup v1
    auto v1_limit = GetCgroupV1MemoryLimit();
    if (v1_limit) {
        return v1_limit;
    }
    
    return std::nullopt;
#else
    return std::nullopt;
#endif
}

// Read Linux /proc/meminfo
std::optional<SystemMemoryInfo> GetLinuxMemoryInfo() {
#if defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        return std::nullopt;
    }
    
    uint64_t mem_total = 0;
    uint64_t mem_available = 0;
    
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            std::istringstream iss(line);
            std::string label;
            uint64_t value;
            std::string unit;
            iss >> label >> value >> unit;
            if (unit == "kB") {
                mem_total = value * 1024;
            }
        } else if (line.find("MemAvailable:") == 0) {
            std::istringstream iss(line);
            std::string label;
            uint64_t value;
            std::string unit;
            iss >> label >> value >> unit;
            if (unit == "kB") {
                mem_available = value * 1024;
            }
        }
        
        if (mem_total > 0 && mem_available > 0) {
            break;
        }
    }
    
    if (mem_total == 0 || mem_available == 0) {
        return std::nullopt;
    }
    
    // Check for cgroup limits to prevent container OOM
    auto cgroup_limit = GetCgroupMemoryLimit();
    if (cgroup_limit) {
        // Use min(host_available, cgroup_limit) to respect container constraints
        uint64_t effective_available = std::min(mem_available, *cgroup_limit);
        return SystemMemoryInfo{mem_total, effective_available};
    }
    
    return SystemMemoryInfo{mem_total, mem_available};
#else
    return std::nullopt;
#endif
}

} // anonymous namespace

std::optional<SystemMemoryInfo> GetSystemMemoryInfo() {
#if defined(_WIN32) || defined(_WIN64)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return SystemMemoryInfo{
            status.ullTotalPhys,
            status.ullAvailPhys
        };
    }
    return std::nullopt;
    
#elif defined(__linux__)
    return GetLinuxMemoryInfo();
    
#elif defined(__APPLE__)
    int mib[2];
    int64_t physical_memory;
    size_t length;
    
    // Get total physical memory
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    length = sizeof(int64_t);
    if (sysctl(mib, 2, &physical_memory, &length, NULL, 0) == 0) {
        // Estimate available as 50% of total (conservative fallback)
        // macOS doesn't provide a simple "available" metric like Linux
        uint64_t available = physical_memory / 2;
        return SystemMemoryInfo{
            static_cast<uint64_t>(physical_memory),
            available
        };
    }
    return std::nullopt;
    
#else
    // Unknown platform - return conservative fallback
    return std::nullopt;
#endif
}

} // namespace pomai::utils