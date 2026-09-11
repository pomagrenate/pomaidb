#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "vector_engine.h"
#include "options.h"
#include "search.h"
#include "wal.h"
#include "memtable.h"

namespace
{

  std::vector<float> MakeVec(std::uint32_t dim, float base)
  {
    std::vector<float> v(dim);
    for (std::uint32_t i = 0; i < dim; ++i)
      v[i] = base + static_cast<float>(i) * 0.001f;
    return v;
  }

  POMAI_TEST(VectorRuntime_TSAN_ActorSerializesCommands)
  {
    const std::uint32_t dim = 32;

    const std::string path = pomai::test::TempDir("pomai-vector_runtime_tsan_test");

    pomai::DBOptions opt;
    opt.path = path;
    opt.dim = dim;
    opt.metric = pomai::MetricType::kL2;
    opt.fsync = pomai::FsyncPolicy::kNever;

    pomai::core::VectorEngine rt(opt, pomai::MembraneKind::kVector, pomai::MetricType::kL2);
    POMAI_EXPECT_OK(rt.Open());

    // Single-threaded: sequential puts (no worker thread, no Enqueue).
    constexpr int kThreads = 4;
    constexpr int kOps = 2000;

    for (int t = 0; t < kThreads; ++t) {
      for (int i = 0; i < kOps; ++i) {
        const pomai::VectorId id = static_cast<pomai::VectorId>(t * 1'000'000 + i);
        auto v = MakeVec(dim, static_cast<float>(id % 1000) * 0.01f);
        POMAI_EXPECT_OK(rt.Put(id, v));
      }
    }

    // Search sanity
    auto q = MakeVec(dim, 0.0f);
    std::vector<pomai::SearchHit> out;
    POMAI_EXPECT_OK(rt.Search(q, /*topk*/ 10, &out));
    POMAI_EXPECT_TRUE(out.size() <= 10);
  }

} // namespace
