#include "tests/common/test_main.h"
#include <cstdint>
#include <vector>

#include "core/shard/mailbox.h"

// Single-threaded: push then close then pop (no producer/consumer threads).
POMAI_TEST(Mailbox_BasicMpsc)
{
  using Q = pomai::core::BoundedMpscQueue<std::uint64_t>;
  Q q(/*cap*/ 8192);  // Must fit kProducers * kPer

  std::uint64_t sum = 0;

  // Simulate multiple "producers" by pushing in sequence (same total as before).
  constexpr int kProducers = 4;
  constexpr int kPer = 2000;

  for (int p = 0; p < kProducers; ++p) {
    for (int i = 1; i <= kPer; ++i) {
      POMAI_EXPECT_TRUE(q.TryPush(static_cast<std::uint64_t>(i)));
    }
  }

  q.Close();

  // Drain in same thread.
  for (;;) {
    auto v = q.PopBlocking();
    if (!v.has_value()) break;
    sum += *v;
  }

  const std::uint64_t expected_one = (static_cast<std::uint64_t>(kPer) * (kPer + 1)) / 2;
  const std::uint64_t expected = expected_one * kProducers;
  POMAI_EXPECT_EQ(sum, expected);
  POMAI_EXPECT_EQ(q.Size(), 0);
}
