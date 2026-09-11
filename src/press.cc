// pomai/press.cc — Press: Compaction, drying, and reseeding implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "press.h"

#include <algorithm>
#include <cmath>
#include <map>

#include "distance.h"

namespace pomai::compact {

Press::Press(Env* env,
             std::string db_dir,
             uint32_t dim,
             MetricType metric,
             PressOptions options)
    : env_(env ? env : Env::Default()),
      db_dir_(std::move(db_dir)),
      dim_(dim),
      metric_(metric),
      options_(std::move(options)) {}

Press::~Press() = default;

Status Press::Compact(ingest::Rind* rind, manifest::FruitMap* fruit_map) {
    if (!fruit_map) return Status::InvalidArgument("null fruit_map");

    auto snapshot = fruit_map->CurrentSnapshot();
    std::vector<std::shared_ptr<table::MemTable>> frozen;
    if (rind) {
        frozen = rind->TakeFrozenMemtables();
    }

    if ((!snapshot || snapshot->locules().empty()) && frozen.empty()) {
        return Status::Ok();
    }

    struct Item {
        VectorId id{0};
        std::vector<float> vec;
        Metadata meta;
    };
    std::map<VectorId, Item> live_items;

    // 1. Drying: extract live vectors from existing Locules in the snapshot
    if (snapshot) {
        for (const auto& loc : snapshot->locules()) {
            if (!loc) continue;
            for (const auto& aril : loc->arils()) {
                if (!aril) continue;
                for (const auto& entry : aril->directory()) {
                    if (aril->scar().IsDeleted(entry.slot)) continue;
                    auto span = aril->GetVectorSpan(entry.slot);
                    if (span.empty() || span.size() != dim_) continue;

                    Item item;
                    item.id = entry.id;
                    item.vec.assign(span.begin(), span.end());
                    (void)aril->GetMetadata(entry.slot, &item.meta);
                    live_items[item.id] = std::move(item);
                }
            }
        }
    }

    // 2. Merge frozen memtables (overwrites older records, applies deletes)
    for (const auto& table : frozen) {
        if (!table) continue;
        auto cursor = table->CreateCursor();
        table::MemTable::CursorEntry entry;
        while (cursor.Next(&entry)) {
            if (entry.is_deleted) {
                live_items.erase(entry.id);
            } else if (entry.vec.size() == dim_) {
                Item item;
                item.id = entry.id;
                item.vec.assign(entry.vec.begin(), entry.vec.end());
                if (entry.meta) item.meta = *entry.meta;
                live_items[item.id] = std::move(item);
            }
        }
    }

    uint64_t next_gen = fruit_map->NextGeneration();

    // If no live items survive
    if (live_items.empty()) {
        Status s = fruit_map->SaveManifest(next_gen, {});
        if (!s.ok()) return s;

        auto empty_snap = std::make_shared<manifest::FruitSnapshot>(
            next_gen, dim_, metric_, std::vector<std::shared_ptr<storage::Locule>>{});
        (void)fruit_map->InstallSnapshot(empty_snap);

        if (snapshot) {
            for (const auto& old_loc : snapshot->locules()) {
                if (old_loc) (void)env_->DeleteFile(old_loc->filepath());
            }
        }
        return Status::Ok();
    }

    // Convert map to contiguous vector
    std::vector<Item> all_items;
    all_items.reserve(live_items.size());
    for (auto& kv : live_items) {
        all_items.push_back(std::move(kv.second));
    }

    // 3. Reseeding: Partition into Locules and compute anchors
    size_t locule_capacity = options_.target_aril_vector_count * options_.target_locule_aril_count;
    if (locule_capacity == 0) locule_capacity = 50000;
    size_t num_locules = (all_items.size() + locule_capacity - 1) / locule_capacity;
    if (num_locules == 0) num_locules = 1;

    std::vector<std::string> new_locule_files;
    std::vector<std::shared_ptr<storage::Locule>> new_locules;
    new_locules.reserve(num_locules);
    new_locule_files.reserve(num_locules);

    for (size_t l_idx = 0; l_idx < num_locules; ++l_idx) {
        size_t start = l_idx * locule_capacity;
        size_t end = std::min(all_items.size(), start + locule_capacity);
        size_t locule_count = end - start;

        format::LoculeAnchor anchor;
        anchor.id = static_cast<uint32_t>(l_idx + 1);
        anchor.centroid.assign(dim_, 0.0f);

        for (size_t i = start; i < end; ++i) {
            const auto& v = all_items[i].vec;
            for (size_t d = 0; d < dim_; ++d) {
                anchor.centroid[d] += v[d];
            }
        }

        if (locule_count > 0) {
            float inv = 1.0f / static_cast<float>(locule_count);
            for (size_t d = 0; d < dim_; ++d) {
                anchor.centroid[d] *= inv;
            }
        }

        float max_dsq = 0.0f;
        for (size_t i = start; i < end; ++i) {
            float dsq = core::L2Sq(all_items[i].vec, anchor.centroid);
            if (dsq > max_dsq) max_dsq = dsq;
        }
        anchor.radius = std::sqrt(std::max(0.0f, max_dsq));

        // 4. Pressing: Break into Arils and build
        std::vector<std::vector<uint8_t>> serialized_arils;
        size_t aril_cap = options_.target_aril_vector_count;
        if (aril_cap == 0) aril_cap = 10000;
        size_t num_arils = (locule_count + aril_cap - 1) / aril_cap;
        if (num_arils == 0) num_arils = 1;

        for (size_t a_idx = 0; a_idx < num_arils; ++a_idx) {
            size_t a_start = start + a_idx * aril_cap;
            size_t a_end = std::min(end, a_start + aril_cap);

            storage::ArilBuilder builder(static_cast<uint32_t>(a_idx + 1), dim_, options_.index_params, metric_);
            for (size_t i = a_start; i < a_end; ++i) {
                (void)builder.Add(all_items[i].id, all_items[i].vec, false, all_items[i].meta);
            }

            std::vector<uint8_t> aril_bytes;
            Status s = builder.Build(&aril_bytes);
            if (!s.ok()) return s;
            serialized_arils.push_back(std::move(aril_bytes));
        }

        std::string filename = "locule_" + std::to_string(anchor.id) + "_gen_" + std::to_string(next_gen) + ".pom";
        std::string full_path = db_dir_ + "/" + filename;

        Status s = storage::Locule::Write(env_, full_path, anchor.id, next_gen, dim_, anchor, serialized_arils);
        if (!s.ok()) return s;

        std::shared_ptr<storage::Locule> opened_loc;
        s = storage::Locule::Open(env_, full_path, &opened_loc);
        if (!s.ok()) return s;

        new_locules.push_back(std::move(opened_loc));
        new_locule_files.push_back(filename);
    }

    // 5. Atomic installation
    Status s = fruit_map->SaveManifest(next_gen, new_locule_files);
    if (!s.ok()) return s;

    auto new_snap = std::make_shared<manifest::FruitSnapshot>(
        next_gen, dim_, metric_, std::move(new_locules));
    (void)fruit_map->InstallSnapshot(new_snap);

    // 6. Purge old locules
    if (snapshot) {
        for (const auto& old_loc : snapshot->locules()) {
            if (old_loc) {
                (void)env_->DeleteFile(old_loc->filepath());
            }
        }
    }

    return Status::Ok();
}

Status Press::PressFrozenRindOnly(ingest::Rind* rind, manifest::FruitMap* fruit_map) {
    if (!fruit_map) return Status::InvalidArgument("null fruit_map");
    if (!rind) return Status::Ok();

    auto frozen = rind->TakeFrozenMemtables();
    if (frozen.empty()) return Status::Ok();

    struct Item {
        VectorId id{0};
        std::vector<float> vec;
        Metadata meta;
    };
    std::vector<Item> items;

    for (const auto& table : frozen) {
        if (!table) continue;
        auto cursor = table->CreateCursor();
        table::MemTable::CursorEntry entry;
        while (cursor.Next(&entry)) {
            if (entry.is_deleted) continue;
            if (entry.vec.size() == dim_) {
                Item item;
                item.id = entry.id;
                item.vec.assign(entry.vec.begin(), entry.vec.end());
                if (entry.meta) item.meta = *entry.meta;
                items.push_back(std::move(item));
            }
        }
    }

    if (items.empty()) return Status::Ok();

    auto snapshot = fruit_map->CurrentSnapshot();
    uint64_t next_gen = fruit_map->NextGeneration();

    uint32_t base_id = 0;
    std::vector<std::shared_ptr<storage::Locule>> all_locules;
    std::vector<std::string> all_locule_files;
    if (snapshot) {
        base_id = static_cast<uint32_t>(snapshot->locules().size());
        for (const auto& loc : snapshot->locules()) {
            if (loc) {
                all_locules.push_back(loc);
                std::string fname = loc->filepath();
                size_t slash = fname.find_last_of("/\\");
                if (slash != std::string::npos) {
                    fname = fname.substr(slash + 1);
                }
                all_locule_files.push_back(std::move(fname));
            }
        }
    }

    size_t locule_capacity = options_.target_aril_vector_count * options_.target_locule_aril_count;
    if (locule_capacity == 0) locule_capacity = 50000;
    size_t num_locules = (items.size() + locule_capacity - 1) / locule_capacity;
    if (num_locules == 0) num_locules = 1;

    for (size_t l_idx = 0; l_idx < num_locules; ++l_idx) {
        size_t start = l_idx * locule_capacity;
        size_t end = std::min(items.size(), start + locule_capacity);
        size_t locule_count = end - start;

        format::LoculeAnchor anchor;
        anchor.id = base_id + static_cast<uint32_t>(l_idx + 1);
        anchor.centroid.assign(dim_, 0.0f);

        for (size_t i = start; i < end; ++i) {
            const auto& v = items[i].vec;
            for (size_t d = 0; d < dim_; ++d) {
                anchor.centroid[d] += v[d];
            }
        }

        if (locule_count > 0) {
            float inv = 1.0f / static_cast<float>(locule_count);
            for (size_t d = 0; d < dim_; ++d) {
                anchor.centroid[d] *= inv;
            }
        }

        float max_dsq = 0.0f;
        for (size_t i = start; i < end; ++i) {
            float dsq = core::L2Sq(items[i].vec, anchor.centroid);
            if (dsq > max_dsq) max_dsq = dsq;
        }
        anchor.radius = std::sqrt(std::max(0.0f, max_dsq));

        std::vector<std::vector<uint8_t>> serialized_arils;
        size_t aril_cap = options_.target_aril_vector_count;
        if (aril_cap == 0) aril_cap = 10000;
        size_t num_arils = (locule_count + aril_cap - 1) / aril_cap;
        if (num_arils == 0) num_arils = 1;

        for (size_t a_idx = 0; a_idx < num_arils; ++a_idx) {
            size_t a_start = start + a_idx * aril_cap;
            size_t a_end = std::min(end, a_start + aril_cap);

            storage::ArilBuilder builder(static_cast<uint32_t>(a_idx + 1), dim_, options_.index_params, metric_);
            for (size_t i = a_start; i < a_end; ++i) {
                (void)builder.Add(items[i].id, items[i].vec, false, items[i].meta);
            }

            std::vector<uint8_t> aril_bytes;
            Status s = builder.Build(&aril_bytes);
            if (!s.ok()) return s;
            serialized_arils.push_back(std::move(aril_bytes));
        }

        std::string filename = "locule_" + std::to_string(anchor.id) + "_gen_" + std::to_string(next_gen) + ".pom";
        std::string full_path = db_dir_ + "/" + filename;

        Status s = storage::Locule::Write(env_, full_path, anchor.id, next_gen, dim_, anchor, serialized_arils);
        if (!s.ok()) return s;

        std::shared_ptr<storage::Locule> opened_loc;
        s = storage::Locule::Open(env_, full_path, &opened_loc);
        if (!s.ok()) return s;

        all_locules.push_back(std::move(opened_loc));
        all_locule_files.push_back(filename);
    }

    Status s = fruit_map->SaveManifest(next_gen, all_locule_files);
    if (!s.ok()) return s;

    auto new_snap = std::make_shared<manifest::FruitSnapshot>(
        next_gen, dim_, metric_, std::move(all_locules));
    (void)fruit_map->InstallSnapshot(new_snap);

    return Status::Ok();
}

} // namespace pomai::compact
