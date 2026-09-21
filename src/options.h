#pragma once
#include <cstdint>
#include <string>

namespace pomai
{
    class Env;

    enum class FsyncPolicy : uint8_t
    {
        kNever = 0,
        kAlways = 1,
    };

    enum class MetricType : uint8_t
    {
        kL2 = 0,
        kInnerProduct = 1,
        kCosine = 2,
    };

    enum class MembraneKind : uint8_t
    {
        kVector = 0,
    };

    enum class IndexType : uint8_t
    {
        kIvfFlat = 0,
        kHnsw = 1,
    };

    enum class QuantizationType : uint8_t
    {
        kNone = 0,
        kSq8 = 1,
        kFp16 = 2,
        kBit = 3, // Binary Quantization (1-bit sign)
        kPq8 = 4, // Product Quantization (PQ8): M sub-quantizers × 256 centroids, M bytes per vector
    };

    /** When true, vectors are stored as SQ8 (int8) with per-vector min/max for ~4x memory reduction. */
    static constexpr bool kDefaultEnableQuantization = true;

    enum class EdgeProfile : uint8_t
    {
        // Keep user-provided settings as-is.
        kUserDefined = 0,
        // Production durability-first profile.
        kEdgeSafe = 1,
        // Balanced durability and latency defaults.
        kEdgeBalanced = 2,
        // Throughput-first profile for permissive durability environments.
        kEdgeFast = 3,
        // Backward-compatible aliases.
        kLowRam = kEdgeSafe,
        kBalanced = kEdgeBalanced,
        kThroughput = kEdgeFast,
    };

    struct IndexParams
    {
        IndexType type = IndexType::kIvfFlat;
        // IVF Params: 0 = dynamic Faiss heuristic (~4 * sqrt(N) for nlist, ~sqrt(nlist) for nprobe)
        uint32_t nlist = 0;
        uint32_t nprobe = 0;
        // HNSW Params
        uint32_t hnsw_m = 32;
        uint32_t hnsw_ef_construction = 200;
        uint32_t hnsw_ef_search = 64;
        // Adaptive dispatcher: segments with fewer vectors use brute-force SIMD
        // (guaranteeing 100% recall). Larger segments use HNSW graph traversal.
        // Default: 0 = always use HNSW when available (rely on ef_search for recall).
        uint32_t adaptive_threshold = 5000;
        QuantizationType quant_type = QuantizationType::kNone;
        // PQ8 sub-quantizer count. Must divide dim. Used only when quant_type == kPq8.
        uint32_t pq_m = 8;
        // If true, HNSW index references the segment mmap for distances instead of
        // duplicating the full vector pool — saves n×dim×4 bytes of RAM per segment.
        bool hnsw_no_vector_pool = false;

        /** Default index params (balanced quality/memory). */
        static IndexParams Default() {
            return IndexParams{};
        }

        /**
         * Low-memory preset for edge/embedded devices.
         * Fewer IVF centroids, smaller HNSW graph and ef, lower adaptive threshold
         * so more segments use brute-force (predictable, smaller index memory).
         */
        static IndexParams ForEdge() {
            IndexParams p;
            p.type = IndexType::kIvfFlat;
            p.nlist = 16;
            p.nprobe = 4;
            p.hnsw_m = 16;
            p.hnsw_ef_construction = 100;
            p.hnsw_ef_search = 32;
            p.adaptive_threshold = 2000;
            p.quant_type = QuantizationType::kNone;
            p.pq_m = 4;
            p.hnsw_no_vector_pool = true;
            return p;
        }
    };

    struct DBOptions
    {
        std::string path;
        /** VFS for file I/O; nullptr = use Env::Default(). */
        Env* env = nullptr;
        uint32_t dim = 512;
        /** If true, use SQ8 scalar quantization in storage (4x compression). Default true for edge/memory-constrained builds. */
        bool enable_quantization = kDefaultEnableQuantization;
        /** Memtable flush threshold in MiB; when exceeded, auto-freeze triggers backpressure.
         *  0 = use dynamic sizing based on system RAM and dimension (default).
         *  > 0 = manual override (use static threshold for backward compatibility).
         */
        uint32_t memtable_flush_threshold_mb = 0u;
        /** If true, when memtable exceeds threshold the vector engine will Freeze() before accepting more writes. */
        bool auto_freeze_on_pressure = true;

        // Dynamic memtable sizing (enabled by default when memtable_flush_threshold_mb == 0)
        float memtable_budget_pct = 0.15f;  // 15% of available RAM
        uint32_t memtable_min_vectors_per_segment = 8192;  // Target vectors per segment
        uint32_t memtable_min_threshold_mb = 16u;  // Minimum threshold
        uint32_t memtable_max_threshold_mb = 512u;  // Maximum threshold
        // Velocity dampening: allow temporary headroom during burst writes (0-1.0)
        float memtable_burst_dampening_factor = 0.2f;  // 20% headroom (enabled by default)
    
        /** Optional hard cap for memtable size in MiB (0 = unlimited, derive behavior from flush threshold only). */
        uint32_t max_memtable_mb = 0;
        uint32_t search_threads = 0; // 0 => auto
        FsyncPolicy fsync = FsyncPolicy::kNever;
        IndexParams index_params;
        bool routing_enabled = false;
        uint32_t routing_k = 0;
        uint32_t routing_probe = 0;
        uint32_t routing_warmup_mult = 20;
        uint32_t routing_keep_prev = 1;
        bool hybrid_partition_enabled = false;
        std::string partition_primary_key = "device_id";
        std::string partition_secondary_key = "location_id";
        MetricType metric = MetricType::kL2;

        // Edge security: encryption-at-rest.
        bool enable_encryption_at_rest = false;
        // Hex-encoded 32-byte key (64 hex chars) for AES-256-GCM.
        std::string encryption_key_hex;

        // Low-RAM controls: max entries tracked by semantic lifecycle per collection.
        uint32_t max_lifecycle_entries = 20000;

        // WAL write coalescing: buffer Put calls and flush as a single AppendBatch.
        // 0 = disabled. Recommended for kThroughput profile (500us window).
        uint32_t write_coalesce_window_us = 0;
        // Flush the coalesce buffer after this many pending writes regardless of window.
        uint32_t write_coalesce_batch_size = 256;

        // Hardware health / wear-leveling awareness.
        bool endurance_aware_maintenance = false;
        uint64_t write_budget_bytes_per_hour = 0; // 0 = disabled
        uint32_t endurance_freeze_delay_ms = 0;
        float endurance_compaction_bias = 1.0f; // >1 delays compaction under high wear.

        // Vulkan memory bridge (Phase 1 GPU prep; optional, off by default).
        bool vulkan_enable_memory_bridge = false;
        bool vulkan_prefer_unified_memory = true;
        uint32_t vulkan_staging_pool_mb = 16;
        uint64_t vulkan_zero_copy_min_bytes = 4096;

        // Optional deployment profile. Does not override index params; users keep full index freedom.
        EdgeProfile edge_profile = EdgeProfile::kUserDefined;
        // Single-thread cooperative scheduler budget per tick.
        uint32_t tick_max_ops = 8;
        uint32_t tick_max_ms = 5;
        // Opt-in deterministic mode for replay-sensitive paths.
        bool strict_deterministic = false;

        void ApplyEdgeProfile()
        {
            switch (edge_profile)
            {
                case EdgeProfile::kEdgeSafe:
                    memtable_flush_threshold_mb = 0u;  // Enable dynamic sizing
                    memtable_budget_pct = 0.10f;  // 10% RAM budget for memory-constrained
                    memtable_min_threshold_mb = 8u;
                    memtable_max_threshold_mb = 32u;
                    max_memtable_mb = 64u;
                    auto_freeze_on_pressure = true;
                    fsync = FsyncPolicy::kAlways;
                    break;
                case EdgeProfile::kEdgeBalanced:
                    memtable_flush_threshold_mb = 0u;  // Enable dynamic sizing
                    memtable_budget_pct = 0.15f;  // 15% RAM budget (balanced)
                    memtable_min_threshold_mb = 16u;
                    memtable_max_threshold_mb = 256u;
                    max_memtable_mb = 256u;
                    auto_freeze_on_pressure = true;
                    fsync = FsyncPolicy::kAlways;
                    break;
                case EdgeProfile::kEdgeFast:
                    memtable_flush_threshold_mb = 0u;  // Enable dynamic sizing
                    memtable_budget_pct = 0.20f;  // 20% RAM budget for throughput
                    memtable_min_threshold_mb = 32u;
                    memtable_max_threshold_mb = 512u;
                    max_memtable_mb = 512u;
                    auto_freeze_on_pressure = true;
                    fsync = FsyncPolicy::kNever;
                    write_coalesce_window_us = 500u;
                    write_coalesce_batch_size = 256u;
                    break;
                case EdgeProfile::kUserDefined:
                default:
                    break;
            }
        }
    };

    // One membrane = one logical collection.
    struct MembraneSpec
    {
        std::string name;
        uint32_t dim = 0;         // 0 => inherit DBOptions.dim
        MetricType metric = MetricType::kL2;
        IndexParams index_params;
        MembraneKind kind = MembraneKind::kVector;
        uint64_t sync_lsn = 0;
        // Optional retention policy (primarily used by kMeta and kKeyValue membranes).
        // 0 means disabled for each field.
        uint32_t ttl_sec = 0;
        uint32_t retention_max_count = 0;
        uint64_t retention_max_bytes = 0;
    };

} // namespace pomai
