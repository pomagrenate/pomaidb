#include "core/membrane/manager.h"
#include "tests/common/test_tmpdir.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

std::vector<float> MakeVec(std::uint64_t seed, std::uint32_t dim) {
  std::vector<float> v(dim, 0.0f);
  for (std::uint32_t i = 0; i < dim; ++i) {
    v[i] = static_cast<float>((seed + i) % 97) / 97.0f;
  }
  return v;
}

}  // namespace

int main() {
  pomai::DBOptions opt;
  opt.path = pomai::test::TempDir("bench-hybrid-orchestrator");
  opt.dim = 32;
  opt.shard_count = 1;

  pomai::core::MembraneManager mgr(opt);
  auto st = mgr.Open();
  if (!st.ok()) {
    std::cerr << "open failed: " << st.message() << "\n";
    return 2;
  }

  pomai::MembraneSpec docs;
  docs.name = "docs";
  docs.kind = pomai::MembraneKind::kText;
  docs.dim = 32;
  docs.shard_count = 1;
  (void)mgr.CreateMembrane(docs);
  (void)mgr.OpenMembrane("docs");

  for (std::uint64_t i = 1; i <= 20000; ++i) {
    auto v = MakeVec(i, 32);
    (void)mgr.PutVector("__default__", i, v);
    pomai::Metadata m;
    m.text = "device-" + std::to_string(i) + " camera edge";
    (void)mgr.PutVector("docs", i, v, m);
  }

  pomai::MultiModalQuery q;
  q.vector = MakeVec(7, 32);
  q.keywords = "device-123 camera";
  q.top_k = 20;
  q.text_membrane = "docs";
  q.execution_order = pomai::QueryExecutionOrder::kAuto;

  pomai::SearchResult out;
  const auto t0 = Clock::now();
  for (int i = 0; i < 1000; ++i) {
    (void)mgr.SearchMultiModal("__default__", q, &out);
  }
  const auto t1 = Clock::now();
  const double qps = 1000.0 / std::chrono::duration<double>(t1 - t0).count();
  std::cout << "hybrid_orchestrator_qps=" << qps << "\n";
  return 0;
}

