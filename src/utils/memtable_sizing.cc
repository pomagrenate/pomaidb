// pomai/memtable_sizing.cc — Dynamic memtable threshold calculation implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "memtable_sizing.h"
#include "system_memory.h"
#include "utils/logging.h"

#include <algorithm>
#include <cmath>

namespace pomai::utils {

uint64_t CalculateDynamicMemtableThreshold(
    const pomai::DBOptions& options,
    uint32_t dimension,
    std::optional<uint64_t> available_ram_bytes)
{
    // Manual override: if user explicitly set a threshold, use it
    if (options.memtable_flush_threshold_mb > 0) {
        uint64_t manual_threshold = static_cast<uint64_t>(options.memtable_flush_threshold_mb) * 1024 * 1024;
        POMAI_LOG_INFO("Using manual memtable threshold: {} MB (override mode)", 
                      options.memtable_flush_threshold_mb);
        return manual_threshold;
    }
    
    // Dynamic sizing mode
    constexpr uint64_t kFallbackThresholdBytes = 64 * 1024 * 1024;  // 64MB fallback
    uint64_t ram_threshold = kFallbackThresholdBytes;
    
    // Calculate RAM-based threshold if available
    if (available_ram_bytes && *available_ram_bytes > 0) {
        uint64_t min_threshold = static_cast<uint64_t>(options.memtable_min_threshold_mb) * 1024 * 1024;
        uint64_t max_threshold = static_cast<uint64_t>(options.memtable_max_threshold_mb) * 1024 * 1024;
        
        // Calculate budget-based threshold
        double budget_bytes = static_cast<double>(*available_ram_bytes) * options.memtable_budget_pct;
        uint64_t budget_threshold = static_cast<uint64_t>(std::round(budget_bytes));
        
        // Clamp to [min, max] bounds
        ram_threshold = std::clamp(budget_threshold, min_threshold, max_threshold);
        
        POMAI_LOG_INFO("RAM-based threshold: {} MB (budget: {:.1f}% of {} GB available)",
                      ram_threshold / (1024 * 1024),
                      options.memtable_budget_pct * 100.0,
                      *available_ram_bytes / (1024.0 * 1024 * 1024));
    } else {
        // Try to detect system memory if not provided
        auto mem_info = GetSystemMemoryInfo();
        if (mem_info) {
            uint64_t min_threshold = static_cast<uint64_t>(options.memtable_min_threshold_mb) * 1024 * 1024;
            uint64_t max_threshold = static_cast<uint64_t>(options.memtable_max_threshold_mb) * 1024 * 1024;
            
            double budget_bytes = static_cast<double>(mem_info->available_bytes) * options.memtable_budget_pct;
            uint64_t budget_threshold = static_cast<uint64_t>(std::round(budget_bytes));
            
            ram_threshold = std::clamp(budget_threshold, min_threshold, max_threshold);
            
            POMAI_LOG_INFO("System RAM detected: {} GB total, {} GB available. RAM-based threshold: {} MB (budget: {:.1f}%)",
                          mem_info->total_bytes / (1024.0 * 1024 * 1024),
                          mem_info->available_bytes / (1024.0 * 1024 * 1024),
                          ram_threshold / (1024 * 1024),
                          options.memtable_budget_pct * 100.0);
        } else {
            POMAI_LOG_WARN("System memory detection failed. Using fallback threshold: {} MB",
                          kFallbackThresholdBytes / (1024 * 1024));
        }
    }
    
    // Calculate dimension floor to ensure minimum vectors per segment
    constexpr size_t kFloatSize = sizeof(float);
    uint64_t dim_floor = static_cast<uint64_t>(options.memtable_min_vectors_per_segment) * 
                         dimension * kFloatSize;
    
    uint64_t min_threshold = static_cast<uint64_t>(options.memtable_min_threshold_mb) * 1024 * 1024;
    uint64_t max_threshold = static_cast<uint64_t>(options.memtable_max_threshold_mb) * 1024 * 1024;
    
    // CRITICAL: Bounded dimension floor to prevent OOM
    uint64_t adjusted_dim_floor = dim_floor;
    if (dim_floor > max_threshold) {
        // Dimension floor exceeds max threshold - dynamically downscale target vectors
        uint32_t downscaled_vectors = static_cast<uint32_t>(max_threshold / (dimension * kFloatSize));
        if (downscaled_vectors < 1024) {  // Safety: ensure at least 1024 vectors
            downscaled_vectors = 1024;
        }
        adjusted_dim_floor = static_cast<uint64_t>(downscaled_vectors) * dimension * kFloatSize;
        
        POMAI_LOG_WARN("Dimension floor requires {} MB but max threshold is {} MB. "
                      "Downscaling target vectors from {} to {} to prevent OOM.",
                      dim_floor / (1024 * 1024),
                      max_threshold / (1024 * 1024),
                      options.memtable_min_vectors_per_segment,
                      downscaled_vectors);
    }
    
    // Use max of RAM threshold and adjusted dimension floor
    uint64_t effective_threshold = std::max(ram_threshold, adjusted_dim_floor);
    
    // Final clamp to ensure we stay within bounds
    effective_threshold = std::clamp(effective_threshold, min_threshold, max_threshold);
    
    POMAI_LOG_INFO("Effective MemTable Budget: {} MB (calculated from Dim={}, SystemRAM={}, BudgetPct={:.1f}%, MinVectors={})",
                  effective_threshold / (1024 * 1024),
                  dimension,
                  available_ram_bytes ? (*available_ram_bytes / (1024.0 * 1024 * 1024)) : 0.0,
                  options.memtable_budget_pct * 100.0,
                  options.memtable_min_vectors_per_segment);
    
    return effective_threshold;
}

} // namespace pomai::utils