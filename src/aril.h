// pomai/aril.h — Fundamental immutable vector storage unit (Aril)
//
// In the Pomegranate Engine:
// - An Aril is a bounded physical/logical vector group.
// - Separates fast approximate Pulp from exact Seed Kernel.
// - Contains Seed Directory, Seed Scar (tombstones), and optional Aril Graph.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "pomai_format.h"
#include "pulp.h"
#include "seed_kernel.h"
#include "seed_scar.h"
#include "status.h"
#include "metadata.h"
#include "options.h"
#include "utils/palloc_smart_ptr.h"

namespace pomai::index { class HnswIndex; }

namespace pomai::storage {

// Forward declarations (defined in seed_kernel.h and pulp.h)
struct VectorMetadata;
struct PulpMetadata;

/**
 * ArilReader: Zero-copy read-only access to an immutable Aril.
 */
class ArilReader {
public:
    static pomai::Status OpenFromMemory(const uint8_t* base_addr, size_t max_size,
                                        uint32_t aril_id,
                                        alloc::SharedPtr<ArilReader>* out);

    ~ArilReader();

    [[nodiscard]] uint32_t aril_id() const noexcept { return aril_id_; }
    [[nodiscard]] uint32_t vector_count() const noexcept { return header_.vector_count; }
    [[nodiscard]] uint32_t dimension() const noexcept { return header_.dimension; }

    [[nodiscard]] const PulpView& pulp() const noexcept { return pulp_; }
    [[nodiscard]] const SeedKernelView& kernel() const noexcept { return kernel_; }
    [[nodiscard]] const SeedScarView& scar() const noexcept { return scar_; }
    [[nodiscard]] const pomai::index::HnswIndex* local_graph() const noexcept { return graph_.get(); }
    [[nodiscard]] bool HasGraph() const noexcept { return graph_ != nullptr; }

    [[nodiscard]] bool FindSlot(pomai::VectorId id, uint32_t* out_slot) const noexcept;

    pomai::Status GetVector(pomai::VectorId id, std::vector<float>* out) const;
    pomai::Status GetMetadata(uint32_t slot, pomai::Metadata* out) const;

    [[nodiscard]] std::span<const format::SeedDirectoryEntry> directory() const noexcept { return directory_; }
    [[nodiscard]] std::span<const float> GetVectorSpan(uint32_t slot) const noexcept { return kernel_.GetVector(slot); }

    [[nodiscard]] const format::ArilHeader& header() const noexcept { return header_; }

private:
    ArilReader() = default;

    uint32_t aril_id_{0};
    format::ArilHeader header_{};
    const uint8_t* aril_base_{nullptr};
    size_t aril_size_{0};

    PulpView pulp_;
    SeedKernelView kernel_;
    SeedScarView scar_;
    std::span<const format::SeedDirectoryEntry> directory_;

    const uint8_t* meta_base_{nullptr};
    size_t meta_size_{0};

    // HNSW graph loaded from memory uses Adopt() because HNSW library allocates with new
    alloc::UniquePtr<pomai::index::HnswIndex> graph_;
};

/**
 * ArilBuilder: Assembles live entries into a serialized immutable Aril block.
 */
class ArilBuilder {
public:
    struct Entry {
        pomai::VectorId id{0};
        std::vector<float> vec;
        bool is_deleted{false};
        pomai::Metadata meta;
    };

    ArilBuilder(uint32_t aril_id, uint32_t dim,
                pomai::IndexParams index_params = {},
                pomai::MetricType metric = pomai::MetricType::kL2);

    pomai::Status Add(pomai::VectorId id, std::span<const float> vec,
                      bool is_deleted, const pomai::Metadata& meta);

    pomai::Status Build(std::vector<uint8_t>* out_bytes);

    [[nodiscard]] uint32_t aril_id() const noexcept { return aril_id_; }
    [[nodiscard]] uint32_t count() const noexcept { return static_cast<uint32_t>(entries_.size()); }

private:
    uint32_t aril_id_{0};
    uint32_t dim_{0};
    pomai::IndexParams index_params_;
    pomai::MetricType metric_;
    std::vector<Entry> entries_;
};

} // namespace pomai::storage
