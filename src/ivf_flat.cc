#include "ivf_flat.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>

#include "distance.h"
#include "storage/palloc_io.h"

namespace pomai::index {

namespace {
    // File Format Constants
    constexpr char kMagic[] = "POMAI_IVF_V1";
    constexpr size_t kMagicLen = 12; // including null or fixed size
    
    // Helper to write POD
    template<typename T>
    Status WritePod(storage::PallocWritableFile* out, const T& val) {
        return out->Append(Slice(reinterpret_cast<const char*>(&val), sizeof(T)));
    }
    
    template<typename T>
    Status ReadPod(storage::PallocSequentialFile* in, T& val) {
        Slice s;
        Status st = in->Read(sizeof(T), &s);
        if (!st.ok()) return st;
        if (s.size() != sizeof(T)) return Status::Corruption("Short read in ReadPod");
        std::memcpy(&val, s.data(), sizeof(T));
        return Status::Ok();
    }
}

IvfFlatIndex::IvfFlatIndex(uint32_t dim, Options opt) 
    : dim_(dim), opt_(opt) {
    if (opt_.nlist == 0) opt_.nlist = 1;
    lists_.resize(opt_.nlist);
}

IvfFlatIndex::~IvfFlatIndex() = default;

uint32_t IvfFlatIndex::FindNearestCentroid(std::span<const float> vec) const {
    float best_score = -std::numeric_limits<float>::infinity();
    uint32_t best_idx = 0;
    
    for (uint32_t i = 0; i < opt_.nlist; ++i) {
        std::span<const float> c(&centroids_[i * dim_], dim_);
        float score = pomai::core::Dot(vec, c);
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }
    return best_idx;
}

pomai::Status IvfFlatIndex::Train(std::span<const float> data, size_t num_vectors) {
    if (num_vectors == 0) return pomai::Status::Ok();
    if (data.size() < num_vectors * dim_) 
        return pomai::Status::InvalidArgument("Data buffer too small");

    // Standard KMeans
    // Initialize centroids (Random sample)
    centroids_.resize(opt_.nlist * dim_);
    
    std::mt19937 rng(42); // Fixed seed for determinism
    std::uniform_int_distribution<size_t> dist(0, num_vectors - 1);
    
    for (uint32_t i = 0; i < opt_.nlist; ++i) {
        size_t idx = dist(rng);
        const float* src = &data[idx * dim_];
        float* dst = &centroids_[i * dim_];
        std::copy(src, src + dim_, dst);
    }

    // Iterations (Lloyd's)
    // Use L2 for clustering stability, even if assignment is Dot?
    // Let's use L2 for clustering to be safe standard KMeans.
    // Wait, if we use L2 for training but Dot for assignment, buckets might be suboptimal.
    // But Spherical KMeans is tricky without normalization.
    // Let's use L2.
    
    std::vector<uint32_t> assignments(num_vectors);
    std::vector<float> new_centroids(opt_.nlist * dim_);
    std::vector<uint32_t> counts(opt_.nlist);
    
    for (int iter = 0; iter < 10; ++iter) {
        // E-Step
        bool changed = false;
        for (size_t i = 0; i < num_vectors; ++i) {
            std::span<const float> vec(&data[i * dim_], dim_);
            
            float max_score = -std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            
            for (uint32_t c = 0; c < opt_.nlist; ++c) {
                std::span<const float> cen(&centroids_[c * dim_], dim_);
                float s = pomai::core::Dot(vec, cen);
                if (s > max_score) {
                    max_score = s;
                    best_c = c;
                }
            }
            
            if (assignments[i] != best_c) changed = true;
            assignments[i] = best_c;
        }
        
        if (!changed && iter > 0) break;
        
        // M-Step
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        
        for (size_t i = 0; i < num_vectors; ++i) {
            uint32_t c = assignments[i];
            counts[c]++;
            const float* src = &data[i * dim_];
            float* dst = &new_centroids[c * dim_];
            for (uint32_t k = 0; k < dim_; ++k) {
                dst[k] += src[k];
            }
        }
        
        for (uint32_t c = 0; c < opt_.nlist; ++c) {
            if (counts[c] > 0) {
                float inv = 1.0f / static_cast<float>(counts[c]);
                float* dst = &new_centroids[c * dim_];
                for (uint32_t k = 0; k < dim_; ++k) dst[k] *= inv;
            } else {
                // Re-init empty cluster? Keep old.
                std::copy(&centroids_[c * dim_], &centroids_[c * dim_] + dim_, &new_centroids[c * dim_]);
            }
        }
        centroids_.resize(new_centroids.size());
        std::memcpy(centroids_.data(), new_centroids.data(), new_centroids.size() * sizeof(float));
    }
    
    trained_ = true;
    return pomai::Status::Ok();
}

pomai::Status IvfFlatIndex::Add(uint32_t entry_index, std::span<const float> vec) {
    if (!trained_) return pomai::Status::Aborted("Index not trained");
    if (vec.size() != dim_) return pomai::Status::InvalidArgument("Dim mismatch");
    
    uint32_t c = FindNearestCentroid(vec);
    lists_[c].push_back(entry_index);
    total_count_++;
    return pomai::Status::Ok();
}

pomai::Status IvfFlatIndex::Search(std::span<const float> query, uint32_t nprobe, 
                                   std::vector<uint32_t>* out) const {
    if (!trained_) {
        // Fallback or empty? 
        // If not trained, we can't search via index.
        // Return Ok with empty lists -> caller must brute force (or handle fallback).
        return pomai::Status::Ok(); 
        // NOTE: In our design, "not trained" means index shouldn't exist.
    }

    out->clear();
    out->reserve(std::min(total_count_, static_cast<size_t>(1u << 20))); // cap reserve for pathological nlist

    const uint32_t K = std::min(nprobe, opt_.nlist);

    // Fast path: probing every list yields the full posting union (each vector lives in exactly one list).
    // Skip centroid dot products + partial_sort — hot in configs that set nprobe == nlist (e.g. max-recall).
    if (K >= opt_.nlist) {
        for (uint32_t c = 0; c < opt_.nlist; ++c) {
            const auto& lst = lists_[c];
            out->insert(out->end(), lst.begin(), lst.end());
        }
        return pomai::Status::Ok();
    }

    // 1. Score Centroids
    thread_local std::vector<std::pair<float, uint32_t>> scores_reuse;
    scores_reuse.clear();
    scores_reuse.reserve(opt_.nlist);

    for (uint32_t c = 0; c < opt_.nlist; ++c) {
        std::span<const float> cen(&centroids_[c * dim_], dim_);
        float s = pomai::core::Dot(query, cen);
        scores_reuse.push_back({s, c});
    }

    // 2. Select Top nprobe
    std::partial_sort(scores_reuse.begin(), scores_reuse.begin() + K, scores_reuse.end(), std::greater<>());

    // 3. Gather
    for (uint32_t k = 0; k < K; ++k) {
        uint32_t c = scores_reuse[k].second;
        const auto& lst = lists_[c];
        out->insert(out->end(), lst.begin(), lst.end());
    }
    
    return pomai::Status::Ok();
}

pomai::Status IvfFlatIndex::Save(const std::string& path) const {
    alloc::UniquePtr<storage::PallocWritableFile> out;
    auto st = storage::PallocWritableFile::Create(path.c_str(), &out);
    if (!st.ok()) return pomai::Status::Internal("Failed to open file for writing");
    
    // Header
    st = out->Append(Slice(kMagic, kMagicLen));
    if (!st.ok()) return st;
    uint32_t version = 1;
    st = WritePod(out.get(), version);
    if (!st.ok()) return st;
    st = WritePod(out.get(), dim_);
    if (!st.ok()) return st;
    st = WritePod(out.get(), opt_.nlist);
    if (!st.ok()) return st;
    
    size_t tc = total_count_;
    st = WritePod(out.get(), tc);
    if (!st.ok()) return st;
    
    // Centroids
    st = out->Append(Slice(reinterpret_cast<const char*>(centroids_.data()), centroids_.size() * sizeof(float)));
    if (!st.ok()) return st;
    
    // Lists
    // Format: [ListSize(u32)] [IDs...] for each list
    for (const auto& lst : lists_) {
        uint32_t sz = static_cast<uint32_t>(lst.size());
        st = WritePod(out.get(), sz);
        if (!st.ok()) return st;
        if (sz > 0) {
            st = out->Append(Slice(reinterpret_cast<const char*>(lst.data()), lst.size() * sizeof(uint32_t)));
            if (!st.ok()) return st;
        }
    }
    
    st = out->Flush();
    if (!st.ok()) return st;
    return out->Close();
}

pomai::Status IvfFlatIndex::Load(const std::string& path, std::unique_ptr<IvfFlatIndex>* out) {
    if (!out) return pomai::Status::InvalidArgument("out is null");

    uint64_t file_size = 0;
    auto st = storage::PallocFilesystem::GetFileSize(path.c_str(), &file_size);
    if (!st.ok()) return pomai::Status::NotFound("Index file not found");

    alloc::UniquePtr<storage::PallocSequentialFile> in;
    st = storage::PallocSequentialFile::Open(path.c_str(), &in);
    if (!st.ok()) return st;

    // Header
    Slice magic_slice;
    st = in->Read(kMagicLen, &magic_slice);
    if (!st.ok() || magic_slice.size() != kMagicLen ||
        std::memcmp(magic_slice.data(), kMagic, kMagicLen) != 0) {
        return pomai::Status::Corruption("Invalid index magic");
    }

    uint32_t version = 0;
    st = ReadPod(in.get(), version);
    if (!st.ok() || version != 1) return pomai::Status::Corruption("Unsupported version");

    uint32_t dim = 0, nlist = 0;
    st = ReadPod(in.get(), dim);
    if (!st.ok()) return st;
    st = ReadPod(in.get(), nlist);
    if (!st.ok()) return st;

    size_t total_count = 0;
    st = ReadPod(in.get(), total_count);
    if (!st.ok()) return pomai::Status::Corruption("Truncated index header");
    if (dim == 0 || nlist == 0) {
        return pomai::Status::Corruption("Invalid index dimensions");
    }

    const uint64_t centroid_floats = static_cast<uint64_t>(dim) * static_cast<uint64_t>(nlist);
    const uint64_t centroid_bytes = centroid_floats * sizeof(float);
    const uint64_t fixed_header = kMagicLen + sizeof(uint32_t) * 3 + sizeof(size_t);
    const uint64_t min_bytes_needed = fixed_header + centroid_bytes + static_cast<uint64_t>(nlist) * sizeof(uint32_t);
    if (centroid_floats > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        min_bytes_needed > file_size) {
        return pomai::Status::Corruption("Index header claims oversized payload");
    }

    Options opt;
    opt.nlist = nlist;
    auto idx = std::make_unique<IvfFlatIndex>(dim, opt);
    idx->total_count_ = total_count;
    idx->trained_ = true;

    // Centroids: read in streaming chunks into idx->centroids_
    idx->centroids_.resize(static_cast<size_t>(centroid_floats));
    {
        char* dst = reinterpret_cast<char*>(idx->centroids_.data());
        size_t remaining = static_cast<size_t>(centroid_bytes);
        while (remaining > 0) {
            size_t to_read = std::min(static_cast<size_t>(storage::kDefaultIoBufferSize), remaining);
            Slice chunk_slice;
            st = in->Read(to_read, &chunk_slice);
            if (!st.ok() || chunk_slice.size() != to_read) {
                return pomai::Status::Corruption("Truncated centroids");
            }
            std::memcpy(dst, chunk_slice.data(), to_read);
            dst += to_read;
            remaining -= to_read;
        }
    }

    // Lists
    uint64_t ids_seen = 0;
    const uint64_t max_possible_ids = (file_size - fixed_header - centroid_bytes - static_cast<uint64_t>(nlist) * sizeof(uint32_t)) / sizeof(uint32_t);
    for (uint32_t i = 0; i < nlist; ++i) {
        uint32_t sz;
        st = ReadPod(in.get(), sz);
        if (!st.ok()) return st;
        ids_seen += static_cast<uint64_t>(sz);
        if (ids_seen > max_possible_ids) {
            return pomai::Status::Corruption("Index list size exceeds file size");
        }

        idx->lists_[i].resize(sz);
        if (sz > 0) {
            size_t list_bytes = static_cast<size_t>(sz) * sizeof(uint32_t);
            char* dst = reinterpret_cast<char*>(idx->lists_[i].data());
            size_t remaining = list_bytes;
            while (remaining > 0) {
                size_t to_read = std::min(static_cast<size_t>(storage::kDefaultIoBufferSize), remaining);
                Slice list_slice;
                st = in->Read(to_read, &list_slice);
                if (!st.ok() || list_slice.size() != to_read) {
                    return pomai::Status::Corruption("Read failed/Truncated list");
                }
                std::memcpy(dst, list_slice.data(), to_read);
                dst += to_read;
                remaining -= to_read;
            }
        }
    }

    *out = std::move(idx);
    return pomai::Status::Ok();
}

} // namespace pomai::index
