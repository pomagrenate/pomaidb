#include "tests/common/test_main.h"

#include "semantic_lifecycle.h"

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

} // namespace
} // namespace pomai

