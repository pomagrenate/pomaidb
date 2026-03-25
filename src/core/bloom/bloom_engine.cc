#include "core/bloom/bloom_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>

namespace pomai::core {

// ---- BloomFilter helpers ---------------------------------------------------

void BloomEngine::BloomFilter::SetBit(std::size_t pos)
{
    bits[pos / 8] |= static_cast<uint8_t>(1u << (pos % 8));
}

bool BloomEngine::BloomFilter::GetBit(std::size_t pos) const
{
    return (bits[pos / 8] >> (pos % 8)) & 1u;
}

// ---- Construction ----------------------------------------------------------

BloomEngine::BloomEngine(std::size_t max_keys_per_filter, double target_fpr)
    : max_keys_per_filter_(max_keys_per_filter)
{
    // Optimal bit array size: m = ceil(-n * ln(p) / (ln2)^2)
    const double ln2   = 0.693147180559945;
    const double ln2sq = ln2 * ln2;
    const double n     = static_cast<double>(std::max(max_keys_per_filter, std::size_t{1}));
    const double p     = std::max(target_fpr, 1e-10);
    bits_per_filter_   = static_cast<std::size_t>(std::ceil(-n * std::log(p) / ln2sq));
    bits_per_filter_   = std::max(bits_per_filter_, std::size_t{64});

    // Optimal number of hashes: k = ceil((m/n) * ln2)
    num_hashes_ = static_cast<std::size_t>(
        std::ceil((static_cast<double>(bits_per_filter_) / n) * ln2));
    num_hashes_ = std::clamp(num_hashes_, std::size_t{1}, std::size_t{16});
}

// ---- Hashing (Kirsch-Mitzenmacher double hashing) --------------------------

void BloomEngine::HashPair(std::string_view key, uint64_t& h1, uint64_t& h2)
{
    // FNV-1a 64-bit for h1, then rotate for h2.
    h1 = 14695981039346656037ULL;
    for (unsigned char c : key) {
        h1 ^= c;
        h1 *= 1099511628211ULL;
    }
    // h2: same body with a different seed
    h2 = 2166136261ULL;
    for (unsigned char c : key) {
        h2 ^= c;
        h2 *= 16777619ULL;
    }
    h2 = (h2 << 33) | (h2 >> 31); // rotate to decorrelate
}

BloomEngine::BloomFilter BloomEngine::MakeFilter() const
{
    BloomFilter f;
    f.num_bits    = bits_per_filter_;
    f.num_hashes  = num_hashes_;
    f.num_elements = 0;
    f.bits.assign((bits_per_filter_ + 7) / 8, 0u);
    return f;
}

// ---- API -------------------------------------------------------------------

Status BloomEngine::Open()  { return Status::Ok(); }
Status BloomEngine::Close() { return Status::Ok(); }

Status BloomEngine::Add(uint64_t filter_id, std::string_view key)
{
    auto it = filters_.find(filter_id);
    if (it == filters_.end()) {
        filters_.emplace(filter_id, MakeFilter());
        it = filters_.find(filter_id);
    }
    BloomFilter& f = it->second;

    uint64_t h1, h2;
    HashPair(key, h1, h2);
    for (std::size_t i = 0; i < f.num_hashes; ++i) {
        std::size_t pos = static_cast<std::size_t>((h1 + i * h2) % f.num_bits);
        f.SetBit(pos);
    }
    ++f.num_elements;
    return Status::Ok();
}

Status BloomEngine::MightContain(uint64_t filter_id, std::string_view key, bool* out) const
{
    if (!out) return Status::InvalidArgument("out is null");
    auto it = filters_.find(filter_id);
    if (it == filters_.end()) { *out = false; return Status::Ok(); }

    const BloomFilter& f = it->second;
    uint64_t h1, h2;
    HashPair(key, h1, h2);
    for (std::size_t i = 0; i < f.num_hashes; ++i) {
        std::size_t pos = static_cast<std::size_t>((h1 + i * h2) % f.num_bits);
        if (!f.GetBit(pos)) { *out = false; return Status::Ok(); }
    }
    *out = true;
    return Status::Ok();
}

Status BloomEngine::Drop(uint64_t filter_id)
{
    auto it = filters_.find(filter_id);
    if (it == filters_.end()) return Status::NotFound("bloom filter not found");
    filters_.erase(it);
    return Status::Ok();
}

Status BloomEngine::EstimateFPR(uint64_t filter_id, double* out) const
{
    if (!out) return Status::InvalidArgument("out is null");
    auto it = filters_.find(filter_id);
    if (it == filters_.end()) return Status::NotFound("bloom filter not found");

    const BloomFilter& f = it->second;
    if (f.num_elements == 0) { *out = 0.0; return Status::Ok(); }

    const double k = static_cast<double>(f.num_hashes);
    const double n = static_cast<double>(f.num_elements);
    const double m = static_cast<double>(f.num_bits);
    // FPR ≈ (1 - e^(-k*n/m))^k
    *out = std::pow(1.0 - std::exp(-k * n / m), k);
    return Status::Ok();
}

Status BloomEngine::EstimateCount(uint64_t filter_id, std::size_t* out) const
{
    if (!out) return Status::InvalidArgument("out is null");
    auto it = filters_.find(filter_id);
    if (it == filters_.end()) return Status::NotFound("bloom filter not found");
    *out = it->second.num_elements;
    return Status::Ok();
}

void BloomEngine::ForEach(
    const std::function<void(uint64_t, std::size_t, std::size_t)>& fn) const
{
    for (const auto& [fid, f] : filters_)
        fn(fid, f.num_bits, f.num_elements);
}

} // namespace pomai::core