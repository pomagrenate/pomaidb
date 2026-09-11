// pomai/pomai_format.h — Binary on-disk specification for PomaiDB (.pom)
//
// The Pomegranate Engine Storage Format:
// - Locule Container (.pom): Contains PomaiFileHeader, ArilDirectory, and multiple Arils.
// - Aril: Immutable bounded vector block containing Pulp (SQ8), Seed Kernel (FP32),
//   Seed Scar (tombstones), Seed Directory, and optional Aril Graph.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pomai::format {

// Magic constants
constexpr uint32_t kPomaiMagic = 0x504F4D41;  // 'POMA' in little-endian
constexpr uint32_t kArilMagic  = 0x4152494C;  // 'ARIL' in little-endian
constexpr uint16_t kPomaiFormatVersion = 1;

// Kind of container
constexpr uint16_t kContainerKindLocule = 1;

#pragma pack(push, 1)

/**
 * PomaiFileHeader: Header for every .pom container file.
 */
struct PomaiFileHeader {
    uint32_t magic{kPomaiMagic};
    uint16_t version{kPomaiFormatVersion};
    uint16_t kind{kContainerKindLocule};

    uint64_t generation{0};

    uint32_t dimension{0};
    uint32_t aril_count{0};

    uint64_t directory_offset{0};
    uint64_t directory_size{0};

    uint64_t footer_offset{0};

    uint32_t checksum{0};
};

/**
 * ArilDirectoryEntry: Entry in the ArilDirectory table of a Locule container.
 */
struct ArilDirectoryEntry {
    uint32_t aril_id{0};
    uint32_t vector_count{0};
    uint64_t aril_offset{0};
    uint64_t aril_size{0};
    uint32_t checksum{0};
    uint32_t reserved{0};
};

/**
 * ArilHeader: Header for an individual Aril within the container.
 * Separates Pulp (approximate) from Seed Kernel (exact).
 */
struct ArilHeader {
    uint32_t magic{kArilMagic};
    uint16_t version{kPomaiFormatVersion};
    uint16_t flags{0};

    uint32_t vector_count{0};
    uint32_t dimension{0};

    // Pulp: Quantized representation (SQ8 / BQ / FP16)
    uint64_t pulp_offset{0};
    uint64_t pulp_size{0};
    float    pulp_quant_min{0.0f};
    float    pulp_quant_inv_scale{0.0f};
    uint8_t  pulp_quant_type{1}; // 1 = SQ8, 2 = FP16, 3 = Bit
    uint8_t  reserved1[3]{0, 0, 0};

    // Seed Kernel: Exact FP32 vectors
    uint64_t kernel_offset{0};
    uint64_t kernel_size{0};

    // Aril Graph: Local HNSW graph block
    uint64_t graph_offset{0};
    uint64_t graph_size{0};

    // Seed Scar: Tombstone bitset
    uint64_t scar_offset{0};
    uint64_t scar_size{0};

    // Seed Directory: VectorId -> slot index
    uint64_t dir_offset{0};
    uint64_t dir_size{0};

    // Metadata Block: Offsets + string blob
    uint64_t metadata_offset{0};
    uint64_t metadata_size{0};

    uint32_t checksum{0};
};

/**
 * SeedDirectoryEntry: Maps user VectorId to local slot [0, vector_count).
 */
struct SeedDirectoryEntry {
    uint64_t id{0};
    uint32_t slot{0};
    uint32_t flags{0}; // 1 = tombstone
};

constexpr uint32_t kLoculeFooterMagic = 0x464F4F54; // 'FOOT'

/**
 * LoculeFooter: Stored at footer_offset in a Locule container.
 * Followed immediately by centroid_dim * sizeof(float) floats.
 */
struct LoculeFooter {
    uint32_t magic{kLoculeFooterMagic};
    uint32_t locule_id{0};
    float radius{0.0f};
    uint32_t centroid_dim{0};
    uint32_t checksum{0};
    uint32_t reserved{0};
};

#pragma pack(pop)

/**
 * LoculeAnchor: Spatial anchor representing a Locule's region in vector space.
 */
struct LoculeAnchor {
    uint32_t id{0};
    float radius{0.0f};
    std::vector<float> centroid;
};

} // namespace pomai::format
