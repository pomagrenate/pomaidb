#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include "core/membrane/manager.h"

namespace pomai {
namespace {

POMAI_TEST(TextMembrane_BasicKeywordSearch) {
  DBOptions opt;
  opt.path = test::TempDir("text-membrane");
  opt.dim = 4;
  opt.shard_count = 1;
  core::MembraneManager mgr(opt);
  POMAI_EXPECT_OK(mgr.Open());

  MembraneSpec spec;
  spec.name = "docs";
  spec.kind = MembraneKind::kText;
  spec.dim = 4;
  spec.shard_count = 1;
  POMAI_EXPECT_OK(mgr.CreateMembrane(spec));
  POMAI_EXPECT_OK(mgr.OpenMembrane("docs"));

  std::vector<float> dummy{0, 0, 0, 0};
  Metadata m1;
  m1.text = "device abc123 reached gate north";
  POMAI_EXPECT_OK(mgr.PutVector("docs", 10, dummy, m1));

  Metadata m2;
  m2.text = "camera xyz seen in lobby";
  POMAI_EXPECT_OK(mgr.PutVector("docs", 11, dummy, m2));

  std::vector<core::LexicalHit> hits;
  POMAI_EXPECT_OK(mgr.SearchLexical("docs", "abc123 gate", 5, &hits));
  POMAI_EXPECT_TRUE(!hits.empty());
  POMAI_EXPECT_EQ(hits[0].id, static_cast<VectorId>(10));
}

}  // namespace
}  // namespace pomai

