#include "pomai/pomai.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<float> MakeVec(std::uint64_t seed, std::uint32_t dim) {
  std::vector<float> v(dim, 0.0f);
  for (std::uint32_t i = 0; i < dim; ++i) {
    v[i] = static_cast<float>((seed + i) % 251) / 251.0f;
  }
  return v;
}

std::size_t ReadRssKb() {
  std::ifstream in("/proc/self/status");
  std::string key;
  while (in >> key) {
    if (key == "VmRSS:") {
      std::size_t kb = 0;
      in >> kb;
      return kb;
    }
    std::string rest;
    std::getline(in, rest);
  }
  return 0;
}

} // namespace

int main() {
  pomai::DBOptions opt;
  opt.path = "/tmp/pomai_low_ram_profile";
  opt.dim = 32;
  opt.shard_count = 1;
  opt.max_lifecycle_entries = 2000;
  opt.max_text_docs = 5000;
  opt.max_query_frontier = 512;

  std::unique_ptr<pomai::DB> db;
  auto st = pomai::DB::Open(opt, &db);
  if (!st.ok()) {
    std::cerr << "open failed: " << st.message() << "\n";
    return 2;
  }

  std::size_t peak_rss_kb = ReadRssKb();
  for (std::uint64_t i = 1; i <= 50000; ++i) {
    auto v = MakeVec(i, opt.dim);
    st = db->Put(i, v);
    if (!st.ok()) {
      std::cerr << "put failed: " << st.message() << "\n";
      return 2;
    }
    if ((i % 1000) == 0) {
      std::size_t rss = ReadRssKb();
      if (rss > peak_rss_kb) peak_rss_kb = rss;
    }
  }
  std::cout << "low_ram_peak_rss_kb=" << peak_rss_kb << "\n";
  return 0;
}

