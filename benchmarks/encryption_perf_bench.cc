#include "pomai/pomai.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct Metrics {
  double ingest_qps = 0.0;
  double p50_us = 0.0;
  double p95_us = 0.0;
  double p99_us = 0.0;
};

std::vector<float> MakeVec(std::uint64_t seed, std::uint32_t dim) {
  std::vector<float> v(dim, 0.0f);
  for (std::uint32_t i = 0; i < dim; ++i) {
    const std::uint64_t x = seed * 6364136223846793005ULL + (i + 1) * 1442695040888963407ULL;
    v[i] = static_cast<float>((x % 2001ULL) - 1000ULL) / 1000.0f;
  }
  return v;
}

double Percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t idx = static_cast<std::size_t>(std::floor(p * static_cast<double>(v.size() - 1)));
  return v[idx];
}

Metrics OneRun(const std::string& path, std::uint32_t dim, std::uint32_t nvec, std::uint32_t nquery, bool encrypted_mode) {
  fs::remove_all(path);

  pomai::DBOptions opts;
  opts.path = path;
  opts.dim = dim;
  opts.shard_count = 1;
  opts.fsync = pomai::FsyncPolicy::kNever;

  if (encrypted_mode) {
    opts.enable_encryption_at_rest = true;
    opts.encryption_key_hex = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
  }

  std::unique_ptr<pomai::DB> db;
  auto st = pomai::DB::Open(opts, &db);
  if (!st.ok()) {
    std::cerr << "open failed: " << st.message() << "\n";
    std::exit(2);
  }

  const auto t0 = Clock::now();
  for (std::uint32_t i = 0; i < nvec; ++i) {
    auto vec = MakeVec(i + 1, dim);
    st = db->Put(i + 1, vec);
    if (!st.ok()) {
      std::cerr << "put failed: " << st.message() << "\n";
      std::exit(2);
    }
  }
  const auto t1 = Clock::now();

  std::vector<double> lat_us;
  lat_us.reserve(nquery);
  pomai::SearchResult out;
  for (std::uint32_t q = 0; q < nquery; ++q) {
    auto query = MakeVec(100000 + q, dim);
    const auto qs = Clock::now();
    st = db->Search(query, 10, &out);
    const auto qe = Clock::now();
    if (!st.ok()) {
      std::cerr << "search failed: " << st.message() << "\n";
      std::exit(2);
    }
    lat_us.push_back(std::chrono::duration<double, std::micro>(qe - qs).count());
  }

  const double ingest_sec = std::chrono::duration<double>(t1 - t0).count();

  Metrics m;
  m.ingest_qps = ingest_sec > 0.0 ? static_cast<double>(nvec) / ingest_sec : 0.0;
  m.p50_us = Percentile(lat_us, 0.50);
  m.p95_us = Percentile(lat_us, 0.95);
  m.p99_us = Percentile(lat_us, 0.99);
  return m;
}

void PrintMetrics(const std::string& name, const Metrics& m) {
  std::cout << name << ": ingest_qps=" << m.ingest_qps << " p50_us=" << m.p50_us
            << " p95_us=" << m.p95_us << " p99_us=" << m.p99_us << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t dim = 64;
  std::uint32_t nvec = 10000;
  std::uint32_t nquery = 1000;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--dim" && i + 1 < argc) dim = static_cast<std::uint32_t>(std::stoul(argv[++i]));
    if (arg == "--vectors" && i + 1 < argc) nvec = static_cast<std::uint32_t>(std::stoul(argv[++i]));
    if (arg == "--queries" && i + 1 < argc) nquery = static_cast<std::uint32_t>(std::stoul(argv[++i]));
  }

  const Metrics plain = OneRun("/tmp/pomai_enc_plain", dim, nvec, nquery, false);
  const Metrics enc = OneRun("/tmp/pomai_enc_enc", dim, nvec, nquery, true);

  PrintMetrics("plain", plain);
  PrintMetrics("encrypted", enc);

  const double ingest_overhead = (plain.ingest_qps > 0.0)
      ? ((plain.ingest_qps - enc.ingest_qps) / plain.ingest_qps) * 100.0
      : 0.0;
  const double p99_overhead = (plain.p99_us > 0.0)
      ? ((enc.p99_us - plain.p99_us) / plain.p99_us) * 100.0
      : 0.0;

  std::cout << "overhead_percent: ingest=" << ingest_overhead
            << " search_p99=" << p99_overhead << "\n";
  std::cout << "NOTE: encrypted mode uses WAL AES-256-GCM path.\n";
  return 0;
}
