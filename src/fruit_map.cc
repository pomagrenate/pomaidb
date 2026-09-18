// pomai/fruit_map.cc — FruitMap: Atomic manifest and snapshot manager implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "fruit_map.h"

#include <algorithm>
#include <sstream>

#include "utils/crc32c.h"
#include "storage/palloc_io.h"

namespace pomai::manifest {

FruitSnapshot::FruitSnapshot(uint64_t generation,
                             uint32_t dimension,
                             MetricType metric,
                             std::vector<alloc::SharedPtr<storage::Locule>> locules)
    : generation_(generation),
      dimension_(dimension),
      metric_(metric),
      locules_(std::move(locules)),
      compass_(metric) {
    compass_.UpdateLocules(locules_);
}

size_t FruitSnapshot::TotalVectorCount() const noexcept {
    size_t count = 0;
    for (const auto& loc : locules_) {
        if (loc) count += loc->TotalVectorCount();
    }
    return count;
}

size_t FruitSnapshot::ActiveVectorCount() const noexcept {
    size_t count = 0;
    for (const auto& loc : locules_) {
        if (loc) count += loc->ActiveVectorCount();
    }
    return count;
}

Status FruitSnapshot::Get(VectorId id, std::vector<float>* out, Metadata* meta) const {
    for (auto it = locules_.rbegin(); it != locules_.rend(); ++it) {
        const auto& loc = *it;
        if (!loc) continue;

        if (loc->IsDeleted(id)) {
            return Status::NotFound("vector is tombstoned");
        }

        Status s = loc->Get(id, out, meta);
        if (s.ok()) {
            return Status::Ok();
        }
    }
    return Status::NotFound("vector not found in snapshot");
}

bool FruitSnapshot::Contains(VectorId id) const {
    for (auto it = locules_.rbegin(); it != locules_.rend(); ++it) {
        const auto& loc = *it;
        if (!loc) continue;
        if (loc->IsDeleted(id)) return false;
        if (loc->Contains(id)) return true;
    }
    return false;
}

bool FruitSnapshot::IsDeleted(VectorId id) const {
    for (auto it = locules_.rbegin(); it != locules_.rend(); ++it) {
        const auto& loc = *it;
        if (!loc) continue;
        if (loc->IsDeleted(id)) return true;
        if (loc->Contains(id)) return false;
    }
    return false;
}

// -----------------------------------------------------------------------------
// FruitMap
// -----------------------------------------------------------------------------

FruitMap::FruitMap(std::string db_dir, uint32_t dim, MetricType metric)
    : db_dir_(std::move(db_dir)),
      dimension_(dim),
      metric_(metric) {
    manifest_path_ = db_dir_ + "/fruit.manifest";
}

FruitMap::~FruitMap() = default;

Status FruitMap::Open() {
    Status s = storage::PallocFilesystem::CreateDir(db_dir_.c_str());
    if (!s.ok()) return s;

    if (storage::PallocFilesystem::FileExists(manifest_path_.c_str()).ok()) {
        alloc::UniquePtr<storage::PallocSequentialFile> file;
        s = storage::PallocSequentialFile::Open(manifest_path_.c_str(), &file);
        if (!s.ok()) return s;

        std::string content;
        char buf[4096];
        Slice slice;
        while (file->Read(sizeof(buf), &slice).ok() && slice.size() > 0) {
            content.append(reinterpret_cast<const char*>(slice.data()), slice.size());
        }

        std::istringstream stream(content);
        std::string line;
        if (!std::getline(stream, line)) {
            return Status::Corruption("empty manifest");
        }
        // Trim carriage return if any
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("POMAI_FRUIT_MANIFEST_v1", 0) != 0) {
            return Status::Corruption("invalid manifest header: " + line);
        }

        uint64_t gen = 0;
        uint32_t dim = dimension_;
        std::vector<std::string> locule_files;

        size_t crc_pos = content.rfind("crc32=");
        if (crc_pos != std::string::npos) {
            std::string payload = content.substr(0, crc_pos);
            uint32_t expected_crc = 0;
            try {
                expected_crc = static_cast<uint32_t>(std::stoul(content.substr(crc_pos + 6)));
            } catch (...) {
                return Status::Corruption("invalid manifest crc32 format");
            }
            uint32_t actual_crc = pomai::util::Crc32c(payload.data(), payload.size());
            if (actual_crc != expected_crc) {
                return Status::Corruption("manifest checksum mismatch");
            }
        }

        while (std::getline(stream, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;

            if (line.rfind("generation=", 0) == 0) {
                gen = std::stoull(line.substr(11));
            } else if (line.rfind("dim=", 0) == 0) {
                dim = static_cast<uint32_t>(std::stoul(line.substr(4)));
            } else if (line.rfind("metric=", 0) == 0) {
                metric_ = static_cast<MetricType>(std::stoul(line.substr(7)));
            } else if (line.rfind("locule_count=", 0) == 0) {
                // Locule count header
            } else if (line.rfind("crc32=", 0) == 0) {
                // Already verified above
            } else {
                locule_files.push_back(line);
            }
        }

        current_generation_ = gen;
        dimension_ = dim;

        std::vector<alloc::SharedPtr<storage::Locule>> loaded_locules;
        loaded_locules.reserve(locule_files.size());

        for (const auto& path : locule_files) {
            std::string full_path = path;
            if (!path.empty() && path[0] != '/' && path.find(':') == std::string::npos) {
                full_path = db_dir_ + "/" + path;
            }

            alloc::SharedPtr<storage::Locule> loc;
            s = storage::Locule::Open(full_path, &loc);
            if (!s.ok()) {
                return Status::Corruption("failed to open locule " + full_path + ": " + s.ToString());
            }
            loaded_locules.push_back(loc);
        }

        void* raw = palloc_malloc_aligned(sizeof(FruitSnapshot), alignof(FruitSnapshot));
        if (!raw) return Status::IOError("FruitSnapshot allocation failed");
        auto snap = alloc::SharedPtr<FruitSnapshot>::AdoptPalloc(new (raw) FruitSnapshot(current_generation_, dimension_, metric_, std::move(loaded_locules)));
        (void)InstallSnapshot(snap);
        return Status::Ok();
    }

    // No manifest exists yet; initialize empty snapshot
    void* raw = palloc_malloc_aligned(sizeof(FruitSnapshot), alignof(FruitSnapshot));
    if (!raw) return Status::IOError("FruitSnapshot allocation failed");
    auto snap = alloc::SharedPtr<FruitSnapshot>::AdoptPalloc(new (raw) FruitSnapshot(0, dimension_, metric_, std::vector<alloc::SharedPtr<storage::Locule>>{}));
    (void)InstallSnapshot(snap);
    return Status::Ok();
}

Status FruitMap::SaveManifest(uint64_t generation, const std::vector<std::string>& locule_files) {
    std::ostringstream ss;
    ss << "POMAI_FRUIT_MANIFEST_v1\n";
    ss << "generation=" << generation << "\n";
    ss << "dim=" << dimension_ << "\n";
    ss << "metric=" << static_cast<int>(metric_) << "\n";
    ss << "locule_count=" << locule_files.size() << "\n";

    for (const auto& path : locule_files) {
        ss << path << "\n";
    }

    std::string payload = ss.str();
    uint32_t crc = pomai::util::Crc32c(payload.data(), payload.size());
    ss << "crc32=" << crc << "\n";

    std::string final_content = ss.str();
    std::string tmp_manifest = manifest_path_ + ".tmp";

    alloc::UniquePtr<storage::PallocWritableFile> file;
    Status s = storage::PallocWritableFile::Create(tmp_manifest.c_str(), &file);
    if (!s.ok()) return s;

    s = file->Append(Slice(final_content.data(), final_content.size()));
    if (!s.ok()) return s;
    s = file->Flush();
    if (!s.ok()) return s;
    s = file->Sync();
    if (!s.ok()) return s;
    s = file->Close();
    if (!s.ok()) return s;

    // Atomic rename simulation
    s = storage::PallocFilesystem::RemoveFile(manifest_path_.c_str());
    if (!s.ok() && s.code() != ErrorCode::kNotFound) return s;
    s = storage::PallocFilesystem::RemoveFile(tmp_manifest.c_str());
    if (!s.ok()) return s;
    s = storage::PallocWritableFile::Create(manifest_path_.c_str(), &file);
    if (!s.ok()) return s;
    s = file->Append(Slice(final_content.data(), final_content.size()));
    if (!s.ok()) return s;
    s = file->Flush();
    if (!s.ok()) return s;
    s = file->Sync();
    if (!s.ok()) return s;
    s = file->Close();
    if (!s.ok()) return s;

    return Status::Ok();
}

Status FruitMap::InstallSnapshot(alloc::SharedPtr<FruitSnapshot> snapshot) {
    psync::LockGuard<psync::Mutex> lock(snapshot_mu_);
    current_snapshot_ = snapshot;
    return Status::Ok();
}

alloc::SharedPtr<FruitSnapshot> FruitMap::CurrentSnapshot() const {
    psync::LockGuard<psync::Mutex> lock(snapshot_mu_);
    return current_snapshot_;
}

} // namespace pomai::manifest
