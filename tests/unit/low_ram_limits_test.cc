#include "tests/common/test_main.h"

#include "core/lifecycle/semantic_lifecycle.h"
#include "core/text/text_membrane.h"

namespace pomai {
namespace {

POMAI_TEST(LowRam_SemanticLifecycle_CapBounded) {
  core::SemanticLifecycle lc(128);
  for (std::uint64_t i = 0; i < 5000; ++i) {
    lc.OnWrite(i);
    lc.OnRead(i);
  }
  const std::size_t total = lc.CountHot() + lc.CountWarm() + lc.CountCold();
  POMAI_EXPECT_TRUE(total <= 128);
}

POMAI_TEST(LowRam_TextMembrane_CapBounded) {
  core::TextMembrane tm(64);
  for (std::uint64_t i = 0; i < 1000; ++i) {
    POMAI_EXPECT_OK(tm.Put(i, "device-" + std::to_string(i)));
  }
  std::vector<core::LexicalHit> out;
  POMAI_EXPECT_OK(tm.Search("device", 1000, &out));
  POMAI_EXPECT_TRUE(out.size() <= 64);
}

} // namespace
} // namespace pomai

