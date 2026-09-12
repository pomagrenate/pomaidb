// pomai/locule.cc — Locule: Spatial compartment container (.pom) implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "locule.h"

#include <algorithm>
#include <cstring>

#include "utils/crc32c.h"

namespace pomai::storage {

Locule::~Locule() = default;

void Locule::CloseMapping() {
    arils_.clear();
    mapping_.reset();
    memory_buffer_.reset();
    base_addr_ = nullptr;
}

Status Locule::Open(Env* env, const std::string& filepath, std::shared_ptr<Locule>* out) {
    if (!env || !out) {
        return Status::InvalidArgument("null env or output pointer");
    }

    std::unique_ptr<FileMapping> mapping;
    Status s = env->NewFileMapping(filepath, &mapping);
    if (!s.ok()) {
        return s;
    }

    if (!mapping || !mapping->Data() || mapping->Size() < sizeof(format::PomaiFileHeader)) {
        return Status::Corruption("locule file too small for PomaiFileHeader: " + filepath);
    }

    auto locule = std::shared_ptr<Locule>(new Locule());
    locule->filepath_ = filepath;
    locule->file_size_ = mapping->Size();
    locule->base_addr_ = static_cast<const uint8_t*>(mapping->Data());
    locule->mapping_ = std::move(mapping);

    // Read and validate header
    format::PomaiFileHeader hdr{};
    std::memcpy(&hdr, locule->base_addr_, sizeof(hdr));

    if (hdr.magic != format::kPomaiMagic) {
        return Status::Corruption("invalid pomai magic in " + filepath);
    }
    if (hdr.version != format::kPomaiFormatVersion) {
        return Status::Corruption("unsupported pomai version in " + filepath);
    }

    locule->generation_ = hdr.generation;
    locule->dimension_ = hdr.dimension;

    // Read footer if present
    if (hdr.footer_offset != 0 && hdr.footer_offset + sizeof(format::LoculeFooter) <= locule->file_size_) {
        format::LoculeFooter footer{};
        std::memcpy(&footer, locule->base_addr_ + hdr.footer_offset, sizeof(footer));
        if (footer.magic == format::kLoculeFooterMagic) {
            locule->locule_id_ = footer.locule_id;
            locule->anchor_.id = footer.locule_id;
            locule->anchor_.radius = footer.radius;
            if (footer.centroid_dim > 0 &&
                hdr.footer_offset + sizeof(format::LoculeFooter) + footer.centroid_dim * sizeof(float) <= locule->file_size_) {
                locule->anchor_.centroid.resize(footer.centroid_dim);
                std::memcpy(locule->anchor_.centroid.data(),
                            locule->base_addr_ + hdr.footer_offset + sizeof(format::LoculeFooter),
                            footer.centroid_dim * sizeof(float));
            }
        }
    }

    // Read Aril directory
    if (hdr.aril_count > 0) {
        size_t required_dir_size = hdr.aril_count * sizeof(format::ArilDirectoryEntry);
        if (hdr.directory_offset + required_dir_size > locule->file_size_) {
            return Status::Corruption("aril directory truncated in " + filepath);
        }

        const auto* entries = reinterpret_cast<const format::ArilDirectoryEntry*>(
            locule->base_addr_ + hdr.directory_offset);

        locule->arils_.reserve(hdr.aril_count);
        for (uint32_t i = 0; i < hdr.aril_count; ++i) {
            const auto& entry = entries[i];
            if (entry.aril_offset + entry.aril_size > locule->file_size_) {
                return Status::Corruption("aril block extends beyond locule file size: " + filepath);
            }

            std::shared_ptr<ArilReader> aril;
            s = ArilReader::OpenFromMemory(locule->base_addr_ + entry.aril_offset,
                                           entry.aril_size,
                                           entry.aril_id,
                                           &aril);
            if (!s.ok()) {
                return s;
            }
            locule->arils_.push_back(std::move(aril));
        }
    }

    *out = std::move(locule);
    return Status::Ok();
}

Status Locule::OpenFromMemory(std::unique_ptr<uint8_t[]> data, size_t size, std::shared_ptr<Locule>* out) {
    if (!data || size < sizeof(format::PomaiFileHeader) || !out) {
        return Status::InvalidArgument("invalid in-memory locule buffer");
    }

    auto locule = std::shared_ptr<Locule>(new Locule());
    locule->filepath_ = ":memory:";
    locule->file_size_ = size;
    locule->base_addr_ = data.get();
    locule->memory_buffer_ = std::move(data);

    format::PomaiFileHeader hdr{};
    std::memcpy(&hdr, locule->base_addr_, sizeof(hdr));

    if (hdr.magic != format::kPomaiMagic) {
        return Status::Corruption("invalid pomai magic in memory buffer");
    }

    locule->generation_ = hdr.generation;
    locule->dimension_ = hdr.dimension;

    if (hdr.footer_offset != 0 && hdr.footer_offset + sizeof(format::LoculeFooter) <= locule->file_size_) {
        format::LoculeFooter footer{};
        std::memcpy(&footer, locule->base_addr_ + hdr.footer_offset, sizeof(footer));
        if (footer.magic == format::kLoculeFooterMagic) {
            locule->locule_id_ = footer.locule_id;
            locule->anchor_.id = footer.locule_id;
            locule->anchor_.radius = footer.radius;
            if (footer.centroid_dim > 0 &&
                hdr.footer_offset + sizeof(format::LoculeFooter) + footer.centroid_dim * sizeof(float) <= locule->file_size_) {
                locule->anchor_.centroid.resize(footer.centroid_dim);
                std::memcpy(locule->anchor_.centroid.data(),
                            locule->base_addr_ + hdr.footer_offset + sizeof(format::LoculeFooter),
                            footer.centroid_dim * sizeof(float));
            }
        }
    }

    if (hdr.aril_count > 0) {
        size_t required_dir_size = hdr.aril_count * sizeof(format::ArilDirectoryEntry);
        if (hdr.directory_offset + required_dir_size > locule->file_size_) {
            return Status::Corruption("aril directory truncated in memory buffer");
        }

        const auto* entries = reinterpret_cast<const format::ArilDirectoryEntry*>(
            locule->base_addr_ + hdr.directory_offset);

        locule->arils_.reserve(hdr.aril_count);
        for (uint32_t i = 0; i < hdr.aril_count; ++i) {
            const auto& entry = entries[i];
            if (entry.aril_offset + entry.aril_size > locule->file_size_) {
                return Status::Corruption("aril block extends beyond locule memory buffer");
            }

            std::shared_ptr<ArilReader> aril;
            Status s = ArilReader::OpenFromMemory(locule->base_addr_ + entry.aril_offset,
                                                  entry.aril_size,
                                                  entry.aril_id,
                                                  &aril);
            if (!s.ok()) {
                return s;
            }
            locule->arils_.push_back(std::move(aril));
        }
    }

    *out = std::move(locule);
    return Status::Ok();
}

Status Locule::Write(Env* env, const std::string& filepath,
                     uint32_t locule_id, uint64_t generation,
                     uint32_t dim,
                     const format::LoculeAnchor& anchor,
                     const std::vector<std::vector<uint8_t>>& serialized_arils) {
    if (!env) return Status::InvalidArgument("null env");

    constexpr size_t kAlign = 64;
    auto align_up = [](size_t n) -> size_t {
        return (n + (kAlign - 1)) & ~(kAlign - 1);
    };

    size_t header_size = align_up(sizeof(format::PomaiFileHeader));
    size_t dir_offset = header_size;
    size_t dir_size = serialized_arils.size() * sizeof(format::ArilDirectoryEntry);
    size_t current_offset = align_up(dir_offset + dir_size);

    std::vector<format::ArilDirectoryEntry> entries(serialized_arils.size());
    for (size_t i = 0; i < serialized_arils.size(); ++i) {
        current_offset = align_up(current_offset);
        entries[i].aril_id = static_cast<uint32_t>(i + 1);
        entries[i].aril_offset = current_offset;
        entries[i].aril_size = serialized_arils[i].size();
        entries[i].checksum = pomai::util::Crc32c(serialized_arils[i].data(), serialized_arils[i].size());
        
        // Extract vector_count from aril header
        if (serialized_arils[i].size() >= sizeof(format::ArilHeader)) {
            const auto* ahdr = reinterpret_cast<const format::ArilHeader*>(serialized_arils[i].data());
            entries[i].vector_count = ahdr->vector_count;
        }
        current_offset += serialized_arils[i].size();
    }

    size_t footer_offset = align_up(current_offset);
    size_t centroid_bytes = anchor.centroid.size() * sizeof(float);
    size_t footer_size = sizeof(format::LoculeFooter) + centroid_bytes;
    size_t total_size = align_up(footer_offset + footer_size);

    std::vector<uint8_t> buffer(total_size, 0);

    // Build header
    format::PomaiFileHeader hdr{};
    hdr.magic = format::kPomaiMagic;
    hdr.version = format::kPomaiFormatVersion;
    hdr.kind = format::kContainerKindLocule;
    hdr.generation = generation;
    hdr.dimension = dim;
    hdr.aril_count = static_cast<uint32_t>(serialized_arils.size());
    hdr.directory_offset = dir_offset;
    hdr.directory_size = dir_size;
    hdr.footer_offset = footer_offset;
    hdr.checksum = 0;

    std::memcpy(buffer.data(), &hdr, sizeof(hdr));

    // Copy directory entries
    if (!entries.empty()) {
        std::memcpy(buffer.data() + dir_offset, entries.data(), dir_size);
    }

    // Copy Aril blocks
    for (size_t i = 0; i < serialized_arils.size(); ++i) {
        std::memcpy(buffer.data() + entries[i].aril_offset,
                    serialized_arils[i].data(),
                    serialized_arils[i].size());
    }

    // Build footer
    format::LoculeFooter footer{};
    footer.magic = format::kLoculeFooterMagic;
    footer.locule_id = locule_id;
    footer.radius = anchor.radius;
    footer.centroid_dim = static_cast<uint32_t>(anchor.centroid.size());
    footer.checksum = 0;
    std::memcpy(buffer.data() + footer_offset, &footer, sizeof(footer));
    if (!anchor.centroid.empty()) {
        std::memcpy(buffer.data() + footer_offset + sizeof(footer),
                    anchor.centroid.data(),
                    centroid_bytes);
    }

    // Compute checksum over content excluding header checksum field
    hdr.checksum = pomai::util::Crc32c(buffer.data() + 16, total_size - 16);
    std::memcpy(buffer.data(), &hdr, sizeof(hdr));

    // Write file atomically (.tmp -> rename)
    std::string tmp_path = filepath + ".tmp";
    std::unique_ptr<WritableFile> file;
    Status s = env->NewWritableFile(tmp_path, &file);
    if (!s.ok()) return s;

    s = file->Append(Slice(reinterpret_cast<const char*>(buffer.data()), buffer.size()));
    if (!s.ok()) return s;
    s = file->Flush();
    if (!s.ok()) return s;
    s = file->Sync();
    if (!s.ok()) return s;
    s = file->Close();
    if (!s.ok()) return s;

    return env->RenameFile(tmp_path, filepath);
}

size_t Locule::TotalVectorCount() const noexcept {
    size_t total = 0;
    for (const auto& a : arils_) {
        total += a->vector_count();
    }
    return total;
}

size_t Locule::ActiveVectorCount() const noexcept {
    size_t total = 0;
    for (const auto& a : arils_) {
        total += a->vector_count() - a->scar().deleted_count();
    }
    return total;
}

Status Locule::Get(VectorId id, std::vector<float>* out, Metadata* meta) const {
    // Search arils in reverse (newest Aril wins)
    for (auto it = arils_.rbegin(); it != arils_.rend(); ++it) {
        const auto& a = *it;
        uint32_t slot = 0;
        if (a->FindSlot(id, &slot)) {
            if (a->scar().IsDeleted(slot)) {
                return Status::NotFound("vector is tombstoned");
            }
            Status s = a->GetVector(id, out);
            if (!s.ok()) return s;
            if (meta) {
                (void)a->GetMetadata(slot, meta);
            }
            return Status::Ok();
        }
    }
    return Status::NotFound("vector not found in locule");
}

bool Locule::Contains(VectorId id) const {
    for (auto it = arils_.rbegin(); it != arils_.rend(); ++it) {
        const auto& a = *it;
        uint32_t slot = 0;
        if (a->FindSlot(id, &slot)) {
            return !a->scar().IsDeleted(slot);
        }
    }
    return false;
}

bool Locule::IsDeleted(VectorId id) const {
    for (auto it = arils_.rbegin(); it != arils_.rend(); ++it) {
        const auto& a = *it;
        uint32_t slot = 0;
        if (a->FindSlot(id, &slot)) {
            return a->scar().IsDeleted(slot);
        }
    }
    return false;
}

} // namespace pomai::storage
