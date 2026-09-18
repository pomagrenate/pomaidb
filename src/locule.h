// pomai/locule.h — Locule: Spatial compartment container (.pom) holding immutable Arils
//
// In the Pomegranate Engine:
// - A Locule represents a spatial region in vector space.
// - Bounded by a LoculeAnchor (centroid and radius).
// - Backed by a single .pom container file on disk, mapped via zero-copy mmap.
// - Contains one or more immutable Arils.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "aril.h"
#include "pomai_format.h"
#include "status.h"
#include "types.h"
#include "metadata.h"
#include "storage/palloc_io.h"
#include "utils/palloc_smart_ptr.h"
#include "utils/palloc_allocator.h"

namespace pomai::storage {

class Locule {
public:
    static Status Open(const std::string& filepath, alloc::SharedPtr<Locule>* out);
    static Status OpenFromMemory(void* data, size_t size, alloc::SharedPtr<Locule>* out);

    static Status Write(const std::string& filepath,
                        uint32_t locule_id, uint64_t generation,
                        uint32_t dim,
                        const format::LoculeAnchor& anchor,
                        const std::vector<std::vector<uint8_t>>& serialized_arils);

    ~Locule();

    void CloseMapping();

    [[nodiscard]] uint32_t locule_id() const noexcept { return locule_id_; }
    [[nodiscard]] uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] uint32_t dimension() const noexcept { return dimension_; }
    [[nodiscard]] const format::LoculeAnchor& anchor() const noexcept { return anchor_; }
    [[nodiscard]] const std::string& filepath() const noexcept { return filepath_; }

    [[nodiscard]] size_t aril_count() const noexcept { return arils_.size(); }
    [[nodiscard]] const std::vector<alloc::SharedPtr<ArilReader>>& arils() const noexcept { return arils_; }
    [[nodiscard]] alloc::SharedPtr<ArilReader> get_aril(size_t index) const {
        if (index < arils_.size()) return arils_[index];
        return nullptr;
    }

    [[nodiscard]] size_t TotalVectorCount() const noexcept;
    [[nodiscard]] size_t ActiveVectorCount() const noexcept;

    Status Get(VectorId id, std::vector<float>* out, Metadata* meta = nullptr) const;
    [[nodiscard]] bool Contains(VectorId id) const;
    [[nodiscard]] bool IsDeleted(VectorId id) const;

private:
    Locule() = default;

    uint32_t locule_id_{0};
    uint64_t generation_{0};
    uint32_t dimension_{0};
    format::LoculeAnchor anchor_;
    std::string filepath_;

    alloc::UniquePtr<storage::PallocFileMapping> mapping_;
    uint8_t* memory_buffer_{nullptr};  // palloc-backed with 64-byte alignment
    const uint8_t* base_addr_{nullptr};
    size_t file_size_{0};

    std::vector<alloc::SharedPtr<ArilReader>> arils_;
};

} // namespace pomai::storage
