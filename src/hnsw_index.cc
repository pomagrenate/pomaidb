// src/hnsw_index.cc — PomaiDB production wrapper implementation around nmslib/hnswlib
//
// Algorithmic Authority: nmslib/hnswlib (upstream commit 058d7a866e462c00f0a6dea8660969379d5916bd)
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <psync/psync.h>
#include <sstream>
#include <streambuf>
#include "storage/palloc_io.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>  // For _mm_prefetch
#endif

#include "distance.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif
#include "hnswlib/hnswlib.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace pomai::index {

namespace {

constexpr uint32_t kPomaiHnswMagic = 0x504D4832u; // "PMH2"
constexpr uint32_t kPomaiHnswVersion = 2u;

#pragma pack(push, 1)
struct PomaiHnswHeader {
    uint32_t magic{kPomaiHnswMagic};
    uint32_t version{kPomaiHnswVersion};
    uint32_t dim{0};
    uint32_t metric{0};
    uint32_t M{0};
    uint32_t ef_construction{0};
    uint32_t ef_search{0};
    uint32_t reserved{0};
};
#pragma pack(pop)

// ── In-Memory Seekable Stream Buffer for Zero-Copy Deserialization ────────────
struct membuf : std::streambuf {
    membuf(const char* base, size_t size) {
        char* p = const_cast<char*>(base);
        this->setg(p, p, p + size);
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                     std::ios_base::openmode /*which*/ = std::ios_base::in) override {
        char* cur = gptr();
        if (dir == std::ios_base::beg) {
            cur = eback() + off;
        } else if (dir == std::ios_base::cur) {
            cur += off;
        } else if (dir == std::ios_base::end) {
            cur = egptr() + off;
        }
        if (cur < eback() || cur > egptr()) {
            return pos_type(off_type(-1));
        }
        setg(eback(), cur, egptr());
        return pos_type(cur - eback());
    }

    pos_type seekpos(pos_type sp,
                     std::ios_base::openmode which = std::ios_base::in) override {
        return seekoff(off_type(sp), std::ios_base::beg, which);
    }
};

struct imemstream : virtual membuf, std::istream {
    imemstream(const char* base, size_t size)
        : membuf(base, size), std::istream(static_cast<std::streambuf*>(this)) {}
};

// ── PomaiDistanceSpace: Adapter from hnswlib to PomaiDB AVX2 SIMD Kernels ──────
class PomaiDistanceSpace : public hnswlib::SpaceInterface<float> {
public:
    PomaiDistanceSpace(size_t dim, pomai::MetricType metric)
        : dim_(dim), metric_(metric), data_size_(dim * sizeof(float)) {
        SetMetric(metric_);
    }

    void SetMetric(pomai::MetricType metric) {
        metric_ = metric;
        switch (metric_) {
            case pomai::MetricType::kL2:
                dist_func_ = L2SqDist;
                break;
            case pomai::MetricType::kInnerProduct:
                dist_func_ = IPDist;
                break;
            case pomai::MetricType::kCosine:
                dist_func_ = CosineDist;
                break;
        }
    }

    size_t get_data_size() override { return data_size_; }
    hnswlib::DISTFUNC<float> get_dist_func() override { return dist_func_; }
    void* get_dist_func_param() override { return &dim_; }

private:
    static float L2SqDist(const void* a, const void* b, const void* param) {
        const size_t d = *reinterpret_cast<const size_t*>(param);
        return pomai::core::L2Sq(
            std::span<const float>(reinterpret_cast<const float*>(a), d),
            std::span<const float>(reinterpret_cast<const float*>(b), d));
    }

    static float IPDist(const void* a, const void* b, const void* param) {
        const size_t d = *reinterpret_cast<const size_t*>(param);
        float dot = pomai::core::Dot(
            std::span<const float>(reinterpret_cast<const float*>(a), d),
            std::span<const float>(reinterpret_cast<const float*>(b), d));
        return 1.0f - dot; // Minimizing 1 - dot maximizes dot
    }

    static float CosineDist(const void* a, const void* b, const void* param) {
        const size_t d = *reinterpret_cast<const size_t*>(param);
        return pomai::core::CosineDistance(
            std::span<const float>(reinterpret_cast<const float*>(a), d),
            std::span<const float>(reinterpret_cast<const float*>(b), d));
    }

    size_t dim_;
    pomai::MetricType metric_;
    size_t data_size_;
    hnswlib::DISTFUNC<float> dist_func_{nullptr};
};

// ── In-Graph Filter Adapter ───────────────────────────────────────────────────
class FilterAdapter : public hnswlib::BaseFilterFunctor {
public:
    explicit FilterAdapter(IdFilter* filter) : filter_(filter) {}
    bool operator()(hnswlib::labeltype label) override {
        if (!filter_) return true;
        return filter_->IsAllowed(static_cast<VectorId>(label));
    }
private:
    IdFilter* filter_;
};

} // namespace

// ── HnswIndex::Impl ───────────────────────────────────────────────────────────
class HnswIndex::Impl {
public:
    Impl(uint32_t dim, HnswOptions opts, pomai::MetricType metric)
        : dim_(dim), opts_(opts), metric_(metric), space_(dim, metric) {
        size_t initial_cap = std::max<size_t>(opts_.initial_max_elements, 16);
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            &space_, initial_cap, opts_.M, opts_.ef_construction, opts_.random_seed);
    }

    pomai::Status Add(VectorId id, std::span<const float> vec) {
        if (vec.size() != dim_) {
            return pomai::Status::InvalidArgument("vector dimension mismatch");
        }

        psync::UniqueLock<psync::Mutex> lock(write_mu_);
        size_t cur = index_->cur_element_count;
        if (cur >= index_->max_elements_) {
            size_t new_cap = std::max<size_t>(index_->max_elements_ * 2, cur + 1024);
            auto st = index_->resizeIndexNoExceptions(new_cap);
            if (!st.ok()) {
                return pomai::Status::ResourceExhausted(st.message());
            }
        }

        auto st = index_->addPointNoExceptions(vec.data(), static_cast<hnswlib::labeltype>(id));
        if (!st.ok()) {
            return pomai::Status::Internal(st.message());
        }
        return pomai::Status::Ok();
    }

    pomai::Status Search(std::span<const float> query,
                         uint32_t topk,
                         int ef_search,
                         std::vector<VectorId>* out_ids,
                         std::vector<float>* out_dists,
                         IdFilter* filter) const {
        if (query.size() != dim_) {
            return pomai::Status::InvalidArgument("query dimension mismatch");
        }
        if (!out_ids || !out_dists) {
            return pomai::Status::InvalidArgument("out_ids/out_dists cannot be null");
        }

        out_ids->clear();
        out_dists->clear();

        if (topk == 0 || index_->cur_element_count == 0) {
            return pomai::Status::Ok();
        }

        // CRITICAL FIX: Add prefetching for query vector to improve cache performance
        // Prefetch the query vector into L1 cache before search
        const float* query_ptr = query.data();
#if defined(__x86_64__) || defined(_M_X64)
        for (size_t i = 0; i < dim_; i += 64 / sizeof(float)) {
            _mm_prefetch((const char*)(query_ptr + i), _MM_HINT_T0);
        }
#endif

        // Hard Invariant: ef_search >= topk
        size_t eff_ef = (ef_search > 0) ? static_cast<size_t>(ef_search) : opts_.ef_search;
        eff_ef = std::max<size_t>(eff_ef, topk);
        index_->setEf(eff_ef);

        FilterAdapter adapter(filter);
        hnswlib::BaseFilterFunctor* p_filter = filter ? &adapter : nullptr;

        auto result = index_->searchKnnNoExceptions(query.data(), topk, p_filter);
        if (!result.ok()) {
            return pomai::Status::Internal(result.status().message());
        }

        auto pq = std::move(result.value());
        size_t result_sz = pq.size();
        out_ids->resize(result_sz);
        out_dists->resize(result_sz);

        // Priority queue is max-heap (furthest element first).
        // Fill arrays backwards so index 0 is the closest neighbor.
        size_t idx = result_sz;
        while (!pq.empty()) {
            --idx;
            const auto& top = pq.top();
            (*out_dists)[idx] = top.first;
            (*out_ids)[idx] = static_cast<VectorId>(top.second);
            pq.pop();
        }

        return pomai::Status::Ok();
    }

    size_t count() const {
        return index_ ? static_cast<size_t>(index_->cur_element_count.load()) : size_t{0};
    }

    const HnswOptions& opts() const noexcept { return opts_; }
    pomai::MetricType metric() const noexcept { return metric_; }

    pomai::Status SaveToStream(std::ostream& out) const {
        auto st = index_->saveIndexNoExceptions(out);
        if (!st.ok()) {
            return pomai::Status::IOError(st.message());
        }
        return pomai::Status::Ok();
    }

    pomai::Status LoadFromStream(std::istream& in) {
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space_);
        auto st = index_->loadIndexNoExceptions(in, &space_, 0);
        if (!st.ok()) {
            return pomai::Status::Corruption(st.message());
        }
        return pomai::Status::Ok();
    }

    pomai::Status SaveToBuffer(std::vector<uint8_t>* out) const {
        if (!out) return pomai::Status::InvalidArgument("out buffer is null");

        std::ostringstream oss(std::ios::binary);
        auto st = index_->saveIndexNoExceptions(oss);
        if (!st.ok()) {
            return pomai::Status::IOError(st.message());
        }
        std::string graph_bytes = oss.str();

        PomaiHnswHeader hdr;
        hdr.magic = kPomaiHnswMagic;
        hdr.version = kPomaiHnswVersion;
        hdr.dim = dim_;
        hdr.metric = static_cast<uint32_t>(metric_);
        hdr.M = static_cast<uint32_t>(opts_.M);
        hdr.ef_construction = static_cast<uint32_t>(opts_.ef_construction);
        hdr.ef_search = static_cast<uint32_t>(opts_.ef_search);
        hdr.reserved = 0;

        out->resize(sizeof(hdr) + graph_bytes.size());
        std::memcpy(out->data(), &hdr, sizeof(hdr));
        std::memcpy(out->data() + sizeof(hdr), graph_bytes.data(), graph_bytes.size());
        return pomai::Status::Ok();
    }

    pomai::Status LoadFromBuffer(const uint8_t* data, size_t len) {
        if (!data || len < sizeof(PomaiHnswHeader)) {
            return pomai::Status::Corruption("buffer too small for Pomai HNSW header");
        }

        PomaiHnswHeader hdr;
        std::memcpy(&hdr, data, sizeof(hdr));

        if (hdr.magic != kPomaiHnswMagic || hdr.version != kPomaiHnswVersion) {
            return pomai::Status::Corruption("Invalid Pomai HNSW magic or version");
        }

        if (hdr.dim != dim_) {
            return pomai::Status::Corruption("HNSW index dimension mismatch");
        }

        metric_ = static_cast<pomai::MetricType>(hdr.metric);
        space_.SetMetric(metric_);
        opts_.M = hdr.M;
        opts_.ef_construction = hdr.ef_construction;
        opts_.ef_search = hdr.ef_search;

        const char* graph_base = reinterpret_cast<const char*>(data + sizeof(hdr));
        size_t graph_len = len - sizeof(hdr);

        imemstream in(graph_base, graph_len);
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space_);
        auto st = index_->loadIndexNoExceptions(in, &space_, 0);
        if (!st.ok()) {
            return pomai::Status::Corruption(st.message());
        }
        return pomai::Status::Ok();
    }

private:
    uint32_t dim_;
    HnswOptions opts_;
    pomai::MetricType metric_;
    PomaiDistanceSpace space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    mutable psync::Mutex write_mu_;
};

// ── Public HnswIndex Implementation ───────────────────────────────────────────

HnswIndex::HnswIndex(uint32_t dim, HnswOptions opts, pomai::MetricType metric)
    : dim_(dim), opts_(opts), metric_(metric),
      impl_(alloc::UniquePtr<Impl>::Make(nullptr, dim, opts, metric)) {}

HnswIndex::~HnswIndex() = default;

HnswIndex::HnswIndex(HnswIndex&&) noexcept = default;
HnswIndex& HnswIndex::operator=(HnswIndex&&) noexcept = default;

pomai::Status HnswIndex::Add(VectorId id, std::span<const float> vec) {
    return impl_->Add(id, vec);
}

pomai::Status HnswIndex::AddBatch(const VectorId* ids, const float* vecs, std::size_t n) {
    if (!ids || !vecs) {
        return pomai::Status::InvalidArgument("ids or vecs pointer is null");
    }
    for (size_t i = 0; i < n; ++i) {
        auto st = Add(ids[i], std::span<const float>(vecs + i * dim_, dim_));
        if (!st.ok()) return st;
    }
    return pomai::Status::Ok();
}

pomai::Status HnswIndex::Search(std::span<const float> query,
                               uint32_t topk,
                               int ef_search,
                               std::vector<VectorId>* out_ids,
                               std::vector<float>* out_dists,
                               IdFilter* filter) const {
    return impl_->Search(query, topk, ef_search, out_ids, out_dists, filter);
}

std::size_t HnswIndex::count() const {
    return impl_->count();
}

HnswOptions HnswIndex::opts() const noexcept {
    return impl_ ? impl_->opts() : opts_;
}

pomai::MetricType HnswIndex::metric() const noexcept {
    return impl_ ? impl_->metric() : metric_;
}

pomai::Status HnswIndex::SaveToStream(std::ostream& out) const {
    return impl_->SaveToStream(out);
}

pomai::Status HnswIndex::LoadFromStream(std::istream& in) {
    auto st = impl_->LoadFromStream(in);
    if (st.ok()) {
        opts_ = impl_->opts();
        metric_ = impl_->metric();
    }
    return st;
}

pomai::Status HnswIndex::SaveToBuffer(std::vector<uint8_t>* out) const {
    return impl_->SaveToBuffer(out);
}

pomai::Status HnswIndex::LoadFromBuffer(const uint8_t* data, size_t len) {
    auto st = impl_->LoadFromBuffer(data, len);
    if (st.ok()) {
        opts_ = impl_->opts();
        metric_ = impl_->metric();
    }
    return st;
}

pomai::Status HnswIndex::Save(const std::string& path) const {
    std::vector<uint8_t> buf;
    auto st = SaveToBuffer(&buf);
    if (!st.ok()) return st;

    alloc::UniquePtr<storage::PallocWritableFile> file;
    st = storage::PallocWritableFile::Create(path.c_str(), &file);
    if (!st.ok()) return st;

    st = file->Append(Slice(reinterpret_cast<const char*>(buf.data()), buf.size()));
    if (!st.ok()) return st;
    st = file->Flush();
    if (!st.ok()) return st;
    return file->Close();
}

pomai::Status HnswIndex::Load(const std::string& path,
                             alloc::UniquePtr<HnswIndex>* out) {
    if (!out) return pomai::Status::InvalidArgument("out pointer is null");

    uint64_t file_size = 0;
    auto st = storage::PallocFilesystem::GetFileSize(path.c_str(), &file_size);
    if (!st.ok()) return st;

    if (file_size < sizeof(PomaiHnswHeader)) {
        return pomai::Status::Corruption("file too small for HNSW index");
    }

    alloc::UniquePtr<storage::PallocRandomAccessFile> raf;
    st = storage::PallocRandomAccessFile::Open(path.c_str(), &raf);
    if (!st.ok()) return st;

    Slice read_slice;
    st = raf->Read(0, file_size, &read_slice);
    if (!st.ok()) return st;
    if (read_slice.size() != file_size) {
        return pomai::Status::IOError("short read in HnswIndex::Load");
    }

    PomaiHnswHeader hdr;
    std::memcpy(&hdr, read_slice.data(), sizeof(hdr));
    if (hdr.magic != kPomaiHnswMagic || hdr.version != kPomaiHnswVersion) {
        return pomai::Status::Corruption("Invalid Pomai HNSW magic or version");
    }

    HnswOptions opts;
    opts.M = hdr.M;
    opts.ef_construction = hdr.ef_construction;
    opts.ef_search = hdr.ef_search;

    auto index = alloc::UniquePtr<HnswIndex>::Make(nullptr, hdr.dim, opts, static_cast<pomai::MetricType>(hdr.metric));
    st = index->LoadFromBuffer(reinterpret_cast<const uint8_t*>(read_slice.data()), read_slice.size());
    if (!st.ok()) return st;

    *out = std::move(index);
    return pomai::Status::Ok();
}

pomai::Status HnswIndex::Load(const std::string& path,
                             std::unique_ptr<HnswIndex>* out) {
    if (!out) return pomai::Status::InvalidArgument("out pointer is null");

    uint64_t file_size = 0;
    auto st = storage::PallocFilesystem::GetFileSize(path.c_str(), &file_size);
    if (!st.ok()) return st;

    if (file_size < sizeof(PomaiHnswHeader)) {
        return pomai::Status::Corruption("file too small for HNSW index");
    }

    alloc::UniquePtr<storage::PallocRandomAccessFile> raf;
    st = storage::PallocRandomAccessFile::Open(path.c_str(), &raf);
    if (!st.ok()) return st;

    Slice read_slice;
    st = raf->Read(0, file_size, &read_slice);
    if (!st.ok()) return st;
    if (read_slice.size() != file_size) {
        return pomai::Status::IOError("short read in HnswIndex::Load");
    }

    PomaiHnswHeader hdr;
    std::memcpy(&hdr, read_slice.data(), sizeof(hdr));
    if (hdr.magic != kPomaiHnswMagic || hdr.version != kPomaiHnswVersion) {
        return pomai::Status::Corruption("Invalid Pomai HNSW magic or version");
    }

    HnswOptions opts;
    opts.M = hdr.M;
    opts.ef_construction = hdr.ef_construction;
    opts.ef_search = hdr.ef_search;

    auto index = std::make_unique<HnswIndex>(hdr.dim, opts, static_cast<pomai::MetricType>(hdr.metric));
    st = index->LoadFromBuffer(reinterpret_cast<const uint8_t*>(read_slice.data()), read_slice.size());
    if (!st.ok()) return st;

    *out = std::move(index);
    return pomai::Status::Ok();
}

pomai::Status HnswIndex::Load(const std::string& path,
                             uint32_t dim,
                             pomai::MetricType metric,
                             alloc::UniquePtr<HnswIndex>* out) {
    if (!out) return pomai::Status::InvalidArgument("out pointer is null");
    auto st = Load(path, out);
    if (!st.ok()) return st;
    if ((*out)->dim() != dim) return pomai::Status::Corruption("HNSW dim mismatch");
    if ((*out)->metric() != metric) return pomai::Status::Corruption("HNSW metric mismatch");
    return pomai::Status::Ok();
}

pomai::Status HnswIndex::Load(const std::string& path,
                             uint32_t dim,
                             pomai::MetricType metric,
                             std::unique_ptr<HnswIndex>* out) {
    if (!out) return pomai::Status::InvalidArgument("out pointer is null");
    auto st = Load(path, out);
    if (!st.ok()) return st;
    if ((*out)->dim() != dim) return pomai::Status::Corruption("HNSW dim mismatch");
    if ((*out)->metric() != metric) return pomai::Status::Corruption("HNSW metric mismatch");
    return pomai::Status::Ok();
}

} // namespace pomai::index
