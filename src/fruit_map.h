// pomai/fruit_map.h — FruitMap: Atomic manifest and snapshot manager for Locules
//
// In the Pomegranate Engine:
// - FruitMap manages active Locules, their on-disk .pom container paths, and generations.
// - Provides thread-safe immutable FruitSnapshots for readers.
// - Persists fruit.manifest atomically with checksums.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstdint>
#include <psync/psync.h>
#include <span>
#include <string>
#include <vector>

#include "compass.h"
#include "locule.h"
#include "status.h"
#include "types.h"
#include "metadata.h"
#include "utils/palloc_smart_ptr.h"

namespace pomai::manifest {

class FruitSnapshot {
public:
    FruitSnapshot(uint64_t generation,
                  uint32_t dimension,
                  MetricType metric,
                  std::vector<alloc::SharedPtr<storage::Locule>> locules);

    [[nodiscard]] uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] uint32_t dimension() const noexcept { return dimension_; }
    [[nodiscard]] MetricType metric() const noexcept { return metric_; }

    [[nodiscard]] const std::vector<alloc::SharedPtr<storage::Locule>>& locules() const noexcept { return locules_; }
    [[nodiscard]] const routing::Compass& compass() const noexcept { return compass_; }

    [[nodiscard]] size_t TotalVectorCount() const noexcept;
    [[nodiscard]] size_t ActiveVectorCount() const noexcept;

    Status Get(VectorId id, std::vector<float>* out, Metadata* meta = nullptr) const;
    [[nodiscard]] bool Contains(VectorId id) const;
    [[nodiscard]] bool IsDeleted(VectorId id) const;

private:
    uint64_t generation_{0};
    uint32_t dimension_{0};
    MetricType metric_{MetricType::kL2};
    std::vector<alloc::SharedPtr<storage::Locule>> locules_;
    routing::Compass compass_;
};

class FruitMap {
public:
    FruitMap(std::string db_dir, uint32_t dim, MetricType metric);
    ~FruitMap();

    Status Open();
    Status SaveManifest(uint64_t generation, const std::vector<std::string>& locule_files);
    Status InstallSnapshot(alloc::SharedPtr<FruitSnapshot> snapshot);

    [[nodiscard]] alloc::SharedPtr<FruitSnapshot> CurrentSnapshot() const;

    [[nodiscard]] uint64_t NextGeneration() noexcept {
        return ++current_generation_;
    }

private:
    std::string db_dir_;
    std::string manifest_path_;
    uint32_t dimension_{0};
    MetricType metric_{MetricType::kL2};

    uint64_t current_generation_{0};
    mutable psync::Mutex snapshot_mu_;
    alloc::SharedPtr<FruitSnapshot> current_snapshot_;
};

} // namespace pomai::manifest
