// pomai/press.h — Press: Drying, compaction, and reseeding of Arils into Locules
//
// In the Pomegranate Engine:
// - Press performs:
//   1. Drying: removes tombstones identified by SeedScar and memtable deletes.
//   2. Reseeding: computes spatial LoculeAnchors (centroids and bounding radii).
//   3. Pressing: quantizes into Pulp (SQ8), packs into SeedKernel, builds Arils, and writes .pom Locules.
//   4. Atomic update of FruitMap manifest.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "aril.h"
#include "fruit_map.h"
#include "locule.h"
#include "options.h"
#include "rind.h"
#include "status.h"
#include "utils/env.h"

namespace pomai::compact {

struct PressOptions {
    size_t target_aril_vector_count{10000};
    size_t target_locule_aril_count{5};
    IndexParams index_params;
};

class Press {
public:
    Press(Env* env,
          std::string db_dir,
          uint32_t dim,
          MetricType metric,
          PressOptions options = {});

    ~Press();

    /**
     * Compact: Performs full drying, compaction, and reseeding of active Locules and frozen Rind memtables.
     * Produces new .pom files, saves new manifest, updates FruitMap, and purges old files.
     */
    Status Compact(ingest::Rind* rind, manifest::FruitMap* fruit_map);

    /**
     * PressFrozenRindOnly: Quickly packs frozen memtables from Rind into an additional Locule without
     * rewriting existing Locules.
     */
    Status PressFrozenRindOnly(ingest::Rind* rind, manifest::FruitMap* fruit_map);

private:
    Env* env_;
    std::string db_dir_;
    uint32_t dim_;
    MetricType metric_;
    PressOptions options_;
};

} // namespace pomai::compact
