// pomai/aril.cc — Fundamental immutable vector storage unit (Aril) implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "aril.h"

#include <algorithm>
#include <cstring>

#include "crc32c.h"
#include "hnsw_index.h"
#include "palloc_compat.h"
#include "utils/palloc_smart_ptr.h"

namespace pomai::storage {

ArilReader::~ArilReader() = default;

pomai::Status ArilReader::OpenFromMemory(const uint8_t* base_addr, size_t max_size,
                                        uint32_t aril_id,
                                        alloc::SharedPtr<ArilReader>* out) {
    if (!base_addr || max_size < sizeof(format::ArilHeader)) {
        return pomai::Status::Corruption("aril data too small for header");
    }

    format::ArilHeader hdr{};
    std::memcpy(&hdr, base_addr, sizeof(hdr));

    if (hdr.magic != format::kArilMagic) {
        return pomai::Status::Corruption("invalid aril magic");
    }
    if (hdr.version != format::kPomaiFormatVersion) {
        return pomai::Status::Corruption("unsupported aril version");
    }

    // Strict bounds validation with overflow protection
    auto validate_bounds = [&](uint64_t off, uint64_t size, const char* name) -> pomai::Status {
        if (size > 0) {
            if (off < sizeof(format::ArilHeader) || off >= max_size || size > max_size ||
                off + size > max_size || off + size < off) {
                return pomai::Status::Corruption(std::string("aril section out of bounds: ") + name);
            }
        }
        return pomai::Status::Ok();
    };

    auto st = validate_bounds(hdr.pulp_offset, hdr.pulp_size, "pulp");
    if (!st.ok()) return st;
    st = validate_bounds(hdr.kernel_offset, hdr.kernel_size, "kernel");
    if (!st.ok()) return st;
    st = validate_bounds(hdr.graph_offset, hdr.graph_size, "graph");
    if (!st.ok()) return st;
    st = validate_bounds(hdr.scar_offset, hdr.scar_size, "scar");
    if (!st.ok()) return st;
    st = validate_bounds(hdr.dir_offset, hdr.dir_size, "directory");
    if (!st.ok()) return st;
    st = validate_bounds(hdr.metadata_offset, hdr.metadata_size, "metadata");
    if (!st.ok()) return st;
    if (hdr.dimension == 0 || hdr.dimension > 65536) {
        return pomai::Status::Corruption("invalid aril dimension");
    }

    // Section size consistency against declared vector count
    if (hdr.vector_count > 0) {
        if (hdr.vector_count > max_size) {
            return pomai::Status::Corruption("aril declared vector_count out of bounds");
        }
        if (hdr.pulp_size > 0) {
            uint64_t min_pulp = static_cast<uint64_t>(hdr.vector_count) * hdr.dimension;
            if (hdr.pulp_size < min_pulp) {
                return pomai::Status::Corruption("aril pulp section truncated for declared vector_count");
            }
        }
        if (hdr.kernel_size > 0) {
            uint64_t min_kernel = static_cast<uint64_t>(hdr.vector_count) * hdr.dimension * sizeof(float);
            if (hdr.kernel_size < min_kernel) {
                return pomai::Status::Corruption("aril kernel section truncated for declared vector_count");
            }
        }
        if (hdr.dir_size > 0) {
            uint64_t min_dir = static_cast<uint64_t>(hdr.vector_count) * sizeof(format::SeedDirectoryEntry);
            if (hdr.dir_size < min_dir) {
                return pomai::Status::Corruption("aril directory section truncated for declared vector_count");
            }
        }
        if (hdr.scar_size > 0) {
            uint64_t min_scar = (static_cast<uint64_t>(hdr.vector_count) + 7) / 8;
            if (hdr.scar_size < min_scar) {
                return pomai::Status::Corruption("aril scar section truncated for declared vector_count");
            }
        }
    }

    // Allocate ArilReader with palloc with 64-byte alignment for SIMD
    void* raw = palloc_malloc_aligned(sizeof(ArilReader), 64);
    if (!raw) return pomai::Status::IOError("ArilReader allocation failed");
    auto reader = alloc::SharedPtr<ArilReader>::AdoptPalloc(new (raw) ArilReader());
    reader->aril_id_ = aril_id;
    reader->header_ = hdr;
    reader->aril_base_ = base_addr;
    reader->aril_size_ = max_size;

    // Resolve Pulp view
    if (hdr.pulp_size > 0) {
        reader->pulp_ = PulpView(base_addr + hdr.pulp_offset, hdr.vector_count,
                                 hdr.dimension, hdr.pulp_quant_min,
                                 hdr.pulp_quant_inv_scale, hdr.pulp_quant_type);
    }

    // Resolve Kernel view
    if (hdr.kernel_size > 0) {
        const float* k_ptr = reinterpret_cast<const float*>(base_addr + hdr.kernel_offset);
        reader->kernel_ = SeedKernelView(k_ptr, hdr.vector_count, hdr.dimension);
    }

    // Resolve Scar view
    if (hdr.scar_size > 0) {
        reader->scar_ = SeedScarView(base_addr + hdr.scar_offset, hdr.vector_count);
    }

    // Resolve Directory
    if (hdr.dir_size > 0) {
        const auto* dir_ptr = reinterpret_cast<const format::SeedDirectoryEntry*>(base_addr + hdr.dir_offset);
        reader->directory_ = std::span<const format::SeedDirectoryEntry>(dir_ptr, hdr.vector_count);
    }

    // Resolve Metadata
    if (hdr.metadata_size > 0) {
        reader->meta_base_ = base_addr + hdr.metadata_offset;
        reader->meta_size_ = hdr.metadata_size;
    }

    // Resolve Graph (with graceful fallback on corruption)
    if (hdr.graph_size > 0) {
        auto graph_idx = alloc::UniquePtr<index::HnswIndex>::Make(nullptr, hdr.dimension);
        if (graph_idx) {
            auto load_st = graph_idx->LoadFromBuffer(base_addr + hdr.graph_offset, hdr.graph_size);
            if (load_st.ok()) {
                reader->graph_ = std::move(graph_idx);
            } else {
                reader->graph_ = nullptr; // Fallback to Pulp SQ8 flat scan
            }
        }
    }

    *out = reader;
    return pomai::Status::Ok();
}

bool ArilReader::FindSlot(pomai::VectorId id, uint32_t* out_slot) const noexcept {
    if (directory_.empty()) return false;

    // Binary search by VectorId
    auto it = std::lower_bound(directory_.begin(), directory_.end(), id,
        [](const format::SeedDirectoryEntry& entry, pomai::VectorId target) {
            return entry.id < target;
        });

    if (it != directory_.end() && it->id == id) {
        if (out_slot) *out_slot = it->slot;
        return true;
    }
    return false;
}

pomai::Status ArilReader::GetVector(pomai::VectorId id, std::vector<float>* out) const {
    if (!out) return pomai::Status::InvalidArgument("out is null");

    uint32_t slot = 0;
    if (!FindSlot(id, &slot)) {
        return pomai::Status::NotFound("vector not found in aril");
    }

    if (scar_.IsDeleted(slot)) {
        return pomai::Status::NotFound("tombstone");
    }

    auto vec = kernel_.GetVector(slot);
    if (vec.empty()) {
        return pomai::Status::Corruption("kernel vector empty");
    }

    out->assign(vec.begin(), vec.end());
    return pomai::Status::Ok();
}

pomai::Status ArilReader::GetMetadata(uint32_t slot, pomai::Metadata* out) const {
    if (!out) return pomai::Status::InvalidArgument("out is null");
    if (!meta_base_ || slot >= header_.vector_count) {
        return pomai::Status::Ok(); // Empty metadata
    }

    const uint32_t count = header_.vector_count;
    const uint64_t* offsets = reinterpret_cast<const uint64_t*>(meta_base_);
    const uint64_t blob_start_offset = (count + 1) * sizeof(uint64_t);

    if (blob_start_offset > meta_size_) {
        return pomai::Status::Corruption("meta blob start out of bounds");
    }

    const uint64_t start = offsets[slot];
    const uint64_t end = offsets[slot + 1];

    if (start >= end || (blob_start_offset + end) > meta_size_) {
        return pomai::Status::Ok();
    }

    const uint8_t* blob_ptr = meta_base_ + blob_start_offset + start;
    std::memcpy(&out->timestamp, blob_ptr, sizeof(out->timestamp));
    std::memcpy(&out->lsn, blob_ptr + 8, sizeof(out->lsn));

    size_t cursor = 16;
    auto read_str = [&](std::string* s) {
        if (cursor + 4 > (end - start)) return;
        uint32_t len = 0;
        std::memcpy(&len, blob_ptr + cursor, 4);
        cursor += 4;
        if (len > 0 && cursor + len <= (end - start)) {
            s->assign(reinterpret_cast<const char*>(blob_ptr + cursor), len);
            cursor += len;
        } else {
            s->clear();
        }
    };

    read_str(&out->device_id);
    read_str(&out->location_id);
    read_str(&out->tenant);
    read_str(&out->payload);

    return pomai::Status::Ok();
}

// -----------------------------------------------------------------------------
// ArilBuilder Implementation
// -----------------------------------------------------------------------------

ArilBuilder::ArilBuilder(uint32_t aril_id, uint32_t dim,
                         pomai::IndexParams index_params,
                         pomai::MetricType metric)
    : aril_id_(aril_id), dim_(dim),
      index_params_(std::move(index_params)), metric_(metric) {}

pomai::Status ArilBuilder::Add(pomai::VectorId id, std::span<const float> vec,
                              bool is_deleted, const pomai::Metadata& meta) {
    if (!is_deleted && vec.size() != dim_) {
        return pomai::Status::InvalidArgument("dimension mismatch");
    }
    Entry e;
    e.id = id;
    e.is_deleted = is_deleted;
    e.meta = meta;
    if (!is_deleted) {
        e.vec.assign(vec.begin(), vec.end());
    } else {
        e.vec.assign(dim_, 0.0f);
    }
    entries_.push_back(std::move(e));
    return pomai::Status::Ok();
}

pomai::Status ArilBuilder::Build(std::vector<uint8_t>* out_bytes) {
    if (!out_bytes) return pomai::Status::InvalidArgument("out_bytes is null");

    // 1. Sort entries by VectorId for efficient binary search
    std::sort(entries_.begin(), entries_.end(),
              [](const Entry& a, const Entry& b) { return a.id < b.id; });

    const uint32_t count = static_cast<uint32_t>(entries_.size());

    // 2. Prepare Pulp (SQ8)
    PulpBuilder pulp_builder(dim_, 1);
    std::vector<std::span<const float>> active_vecs;
    active_vecs.reserve(count);
    for (const auto& e : entries_) {
        if (!e.is_deleted) {
            active_vecs.emplace_back(e.vec);
        }
    }
    pulp_builder.Train(active_vecs);

    for (const auto& e : entries_) {
        pulp_builder.EncodeAppend(e.vec);
    }

    // 3. Prepare Seed Kernel (FP32)
    SeedKernelBuilder kernel_builder(dim_);
    for (const auto& e : entries_) {
        kernel_builder.Append(e.vec);
    }

    // 4. Prepare Seed Scar (Tombstone Bitset)
    SeedScarBuilder scar_builder(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (entries_[i].is_deleted) {
            scar_builder.MarkDeleted(i);
        }
    }

    // 5. Prepare Seed Directory
    std::vector<format::SeedDirectoryEntry> directory(count);
    for (uint32_t i = 0; i < count; ++i) {
        directory[i].id = entries_[i].id;
        directory[i].slot = i;
        directory[i].flags = entries_[i].is_deleted ? 1 : 0;
    }

    // 6. Prepare Metadata Block
    std::vector<uint64_t> meta_offsets;
    std::vector<uint8_t> meta_blob;
    meta_offsets.reserve(count + 1);
    meta_offsets.push_back(0);

    for (uint32_t i = 0; i < count; ++i) {
        const auto& m = entries_[i].meta;
        const bool has_meta = !m.tenant.empty() || !m.device_id.empty() ||
                              !m.location_id.empty() || m.timestamp > 0 ||
                              m.lsn > 0 || !m.payload.empty();
        if (has_meta) {
            auto append_bytes = [&](const void* p, size_t n) {
                const auto* b = static_cast<const uint8_t*>(p);
                meta_blob.insert(meta_blob.end(), b, b + n);
            };
            auto append_str = [&](const std::string& s) {
                uint32_t len = static_cast<uint32_t>(s.size());
                append_bytes(&len, sizeof(len));
                if (len > 0) append_bytes(s.data(), len);
            };

            append_bytes(&m.timestamp, sizeof(m.timestamp));
            append_bytes(&m.lsn, sizeof(m.lsn));
            append_str(m.device_id);
            append_str(m.location_id);
            append_str(m.tenant);
            append_str(m.payload);

            meta_offsets.push_back(meta_blob.size());
        } else {
            meta_offsets.push_back(meta_blob.size());
        }
    }

    // 7. Calculate aligned offsets
    auto align64 = [](size_t n) -> size_t {
        return (n + 63) & ~size_t(63);
    };

    size_t current_offset = sizeof(format::ArilHeader);

    // Pulp offset
    current_offset = align64(current_offset);
    const uint64_t pulp_off = current_offset;
    const uint64_t pulp_sz = pulp_builder.buffer().size();
    current_offset += pulp_sz;

    // Kernel offset
    current_offset = align64(current_offset);
    const uint64_t kernel_off = current_offset;
    const uint64_t kernel_sz = kernel_builder.size_bytes();
    current_offset += kernel_sz;

    // Graph Block
    std::vector<uint8_t> graph_bytes;
    if (index_params_.type == IndexType::kHnsw && count > 0) {
        index::HnswOptions opts;
        opts.M = (index_params_.hnsw_m > 0) ? index_params_.hnsw_m : 16;
        opts.ef_construction = (index_params_.hnsw_ef_construction > 0) ? index_params_.hnsw_ef_construction : 200;
        opts.ef_search = (index_params_.hnsw_ef_search > 0) ? index_params_.hnsw_ef_search : 64;
        opts.initial_max_elements = count;

        index::HnswIndex hnsw(dim_, opts, metric_);
        for (uint32_t i = 0; i < count; ++i) {
            (void)hnsw.Add(static_cast<VectorId>(i), entries_[i].vec);
        }
        (void)hnsw.SaveToBuffer(&graph_bytes);
    }

    // Graph offset
    current_offset = align64(current_offset);
    const uint64_t graph_off = current_offset;
    const uint64_t graph_sz = graph_bytes.size();
    current_offset += graph_sz;

    // Scar offset
    current_offset = align64(current_offset);
    const uint64_t scar_off = current_offset;
    const uint64_t scar_sz = scar_builder.bytes().size();
    current_offset += scar_sz;

    // Directory offset
    current_offset = align64(current_offset);
    const uint64_t dir_off = current_offset;
    const uint64_t dir_sz = directory.size() * sizeof(format::SeedDirectoryEntry);
    current_offset += dir_sz;

    // Metadata offset
    current_offset = align64(current_offset);
    const uint64_t meta_off = current_offset;
    const uint64_t meta_sz = meta_offsets.size() * sizeof(uint64_t) + meta_blob.size();
    current_offset += meta_sz;

    // 8. Construct ArilHeader
    format::ArilHeader hdr{};
    hdr.magic = format::kArilMagic;
    hdr.version = format::kPomaiFormatVersion;
    hdr.flags = 0;
    hdr.vector_count = count;
    hdr.dimension = dim_;

    hdr.pulp_offset = pulp_off;
    hdr.pulp_size = pulp_sz;
    hdr.pulp_quant_min = pulp_builder.min_val();
    hdr.pulp_quant_inv_scale = pulp_builder.inv_scale();
    hdr.pulp_quant_type = pulp_builder.quant_type();

    hdr.kernel_offset = kernel_off;
    hdr.kernel_size = kernel_sz;

    hdr.graph_offset = (graph_sz > 0) ? graph_off : 0;
    hdr.graph_size = graph_sz;

    hdr.scar_offset = scar_off;
    hdr.scar_size = scar_sz;

    hdr.dir_offset = dir_off;
    hdr.dir_size = dir_sz;

    hdr.metadata_offset = meta_off;
    hdr.metadata_size = meta_sz;

    // Allocate final output
    out_bytes->assign(current_offset, 0);

    // Copy header
    std::memcpy(out_bytes->data(), &hdr, sizeof(hdr));

    // Copy Pulp
    if (pulp_sz > 0) {
        std::memcpy(out_bytes->data() + pulp_off, pulp_builder.buffer().data(), pulp_sz);
    }

    // Copy Kernel
    if (kernel_sz > 0) {
        std::memcpy(out_bytes->data() + kernel_off, kernel_builder.data().data(), kernel_sz);
    }

    // Copy Graph
    if (graph_sz > 0) {
        std::memcpy(out_bytes->data() + graph_off, graph_bytes.data(), graph_sz);
    }

    // Copy Scar
    if (scar_sz > 0) {
        std::memcpy(out_bytes->data() + scar_off, scar_builder.bytes().data(), scar_sz);
    }

    // Copy Directory
    if (dir_sz > 0) {
        std::memcpy(out_bytes->data() + dir_off, directory.data(), dir_sz);
    }

    // Copy Metadata
    if (meta_sz > 0) {
        const size_t offsets_bytes = meta_offsets.size() * sizeof(uint64_t);
        std::memcpy(out_bytes->data() + meta_off, meta_offsets.data(), offsets_bytes);
        if (!meta_blob.empty()) {
            std::memcpy(out_bytes->data() + meta_off + offsets_bytes, meta_blob.data(), meta_blob.size());
        }
    }

    // 9. Compute CRC32C over payload (excluding checksum field itself)
    const uint32_t crc = pomai::util::Crc32c(out_bytes->data() + sizeof(uint32_t),
                                            out_bytes->size() - sizeof(uint32_t));
    format::ArilHeader* final_hdr = reinterpret_cast<format::ArilHeader*>(out_bytes->data());
    final_hdr->checksum = crc;

    return pomai::Status::Ok();
}

} // namespace pomai::storage
