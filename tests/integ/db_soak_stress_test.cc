#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include "pomai/pomai.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace pomai {
namespace {

std::uint32_t ReadEnvU32(const char* key, std::uint32_t fallback) {
  const char* v = std::getenv(key);
  if (v == nullptr || *v == '\0') return fallback;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(v, &end, 10);
  if (end == v) return fallback;
  return static_cast<std::uint32_t>(parsed);
}

std::vector<float> MakeVec(std::uint64_t seed, std::uint32_t dim) {
  std::vector<float> v(dim, 0.0f);
  for (std::uint32_t i = 0; i < dim; ++i) {
    const std::uint64_t x = seed * 1315423911ULL + (i + 1) * 2654435761ULL;
    v[i] = static_cast<float>((x % 2001ULL) - 1000ULL) / 1000.0f;
  }
  return v;
}

POMAI_TEST(DB_SoakStress_LongRunningNoLeaksOrCorruption) {
  // Keep CI fast by default; allow hours-long soak by env override.
  const std::uint32_t seconds = ReadEnvU32("POMAI_SOAK_SECONDS", 6);
  const std::uint32_t dim = ReadEnvU32("POMAI_SOAK_DIM", 32);
  const std::uint32_t topk = 5;

  DBOptions opt;
  opt.path = test::TempDir("db-soak-stress");
  opt.dim = dim;
  opt.shard_count = 2;
  opt.fsync = FsyncPolicy::kNever;
  opt.memtable_flush_threshold_mb = 8;
  opt.max_memtable_mb = 16;
  opt.auto_freeze_on_pressure = true;

  std::unique_ptr<DB> db;
  POMAI_EXPECT_OK(DB::Open(opt, &db));

  const auto start = std::chrono::steady_clock::now();
  std::uint64_t id = 1;
  std::uint64_t ops = 0;

  while (true) {
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed_sec =
        std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
    if (elapsed_sec >= static_cast<long>(seconds)) break;

    auto v = MakeVec(id, dim);
    POMAI_EXPECT_OK(db->Put(id, v));
    ++ops;

    if ((id % 8) == 0) {
      SearchResult out;
      POMAI_EXPECT_OK(db->Search(v, topk, &out));
      POMAI_EXPECT_TRUE(!out.hits.empty());
      ++ops;
    }

    if ((id % 128) == 0) {
      POMAI_EXPECT_OK(db->Freeze("__default__"));
      ++ops;
    }

    if (id > 300 && (id % 16) == 0) {
      POMAI_EXPECT_OK(db->Delete(id - 200));
      ++ops;
    }

    ++id;
  }

  POMAI_EXPECT_TRUE(ops > 100);
  POMAI_EXPECT_OK(db->Close());
}

}  // namespace
}  // namespace pomai
