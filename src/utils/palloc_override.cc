// palloc_override.cc — DISABLED: Global operator new/delete override removed
//
// CRITICAL SECURITY FIX: Global operator overrides have been disabled.
// They were causing hidden dependencies on palloc initialization order
// and could cause silent failures during early startup.
//
// Use pomai::alloc::PallocAllocator<T> for explicit scoped allocation.
//
// Copyright 2026 PomaiDB authors. MIT License.

#include <palloc.h>
#include "palloc_compat.h"

namespace pomai::util {

bool EnsurePallocInitialized() {
    static const bool initialized = []() {
        return pa_version() > 0;
    }();
    return initialized;
}

} // namespace pomai::util

// ============================================================================
// GLOBAL OPERATOR NEW/DELETE OVERRIDES DISABLED
// ============================================================================
// The following global operator overrides have been intentionally disabled
// to prevent process-wide allocation capture that could cause:
// - Hidden dependencies on palloc initialization order
// - Silent failures during early startup before palloc is ready
// - Interference with third-party libraries and standard library internals
//
// For internal allocations that benefit from palloc, use:
//   pomai::alloc::PallocAllocator<T>
//   pomai::alloc::PallocVector<T>
//   pomai::alloc::PallocUnorderedMap<K, V>
//
// See src/utils/palloc_allocator.h for the scoped allocator implementation.
// ============================================================================
