#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include "pomai/pomai.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace pomai {
namespace {
namespace fs = std::filesystem;

std::vector<float> MakeVec(std::uint64_t seed, std::uint32_t dim) {
  std::vector<float> v(dim, 0.0f);
  for (std::uint32_t i = 0; i < dim; ++i) {
    const std::uint64_t x = seed * 11400714819323198485ULL + i * 2654435761ULL;
    v[i] = static_cast<float>((x % 1024ULL)) / 1024.0f;
  }
  return v;
}

void FlipByte(const std::string& path, std::uint64_t offset) {
  std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
  POMAI_EXPECT_TRUE(f.is_open());
  f.seekg(0, std::ios::end);
  const std::uint64_t sz = static_cast<std::uint64_t>(f.tellg());
  if (sz == 0) return;
  const std::uint64_t pos = offset % sz;
  f.seekg(static_cast<std::streamoff>(pos), std::ios::beg);
  char b = 0;
  f.read(&b, 1);
  b = static_cast<char>(b ^ 0x7F);
  f.seekp(static_cast<std::streamoff>(pos), std::ios::beg);
  f.write(&b, 1);
}

POMAI_TEST(SdFaultInjection_CorruptedWalAndSegment_NoCrashOnReopen) {
  DBOptions opt;
  opt.path = test::TempDir("sd-fault-injection");
  opt.dim = 16;
  opt.shard_count = 1;
  opt.fsync = FsyncPolicy::kNever;

  {
    std::unique_ptr<DB> db;
    POMAI_EXPECT_OK(DB::Open(opt, &db));
    for (std::uint64_t i = 1; i <= 3000; ++i) {
      auto v = MakeVec(i, opt.dim);
      POMAI_EXPECT_OK(db->Put(i, v));
      if ((i % 500) == 0) {
        POMAI_EXPECT_OK(db->Freeze("__default__"));
      }
    }
    POMAI_EXPECT_OK(db->Close());
  }

  std::string wal_path;
  std::string seg_path;
  for (const auto& it : fs::recursive_directory_iterator(opt.path)) {
    if (!it.is_regular_file()) continue;
    const auto name = it.path().filename().string();
    if (wal_path.empty() && name.rfind("wal_", 0) == 0) wal_path = it.path().string();
    if (seg_path.empty() && it.path().extension() == ".dat" &&
        name.rfind("manifest.", 0) != 0) {
      seg_path = it.path().string();
    }
  }

  POMAI_EXPECT_TRUE(!wal_path.empty());
  POMAI_EXPECT_TRUE(!seg_path.empty());

  // Simulate SD-bitrot by corrupting one byte in WAL and one byte in a segment.
  FlipByte(wal_path, 17);
  FlipByte(seg_path, 257);

  std::unique_ptr<DB> reopened;
  Status st = DB::Open(opt, &reopened);
  // Accept either graceful open or explicit corruption error, but never crash.
  if (st.ok()) {
    std::vector<float> query(opt.dim, 0.2f);
    SearchResult out;
    (void)reopened->Search(query, 5, &out);
    POMAI_EXPECT_OK(reopened->Close());
  } else {
    POMAI_EXPECT_TRUE(st.code() == ErrorCode::kCorruption ||
                      st.code() == ErrorCode::kAborted ||
                      st.code() == ErrorCode::kIO);
  }
}

}  // namespace
}  // namespace pomai
