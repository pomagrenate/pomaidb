#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include "core/membrane/manager.h"
#include "core/security/key_manager.h"

#include <array>

namespace pomai {
namespace {

POMAI_TEST(QueryOrchestrator_CrossMembrane_VectorThenGraph) {
  DBOptions opt;
  opt.path = test::TempDir("orchestrator-cross");
  opt.dim = 4;
  opt.shard_count = 1;
  opt.fsync = FsyncPolicy::kNever;

  core::MembraneManager mgr(opt);
  POMAI_EXPECT_OK(mgr.Open());

  MembraneSpec vector_spec;
  vector_spec.name = "faces";
  vector_spec.kind = MembraneKind::kVector;
  vector_spec.dim = 4;
  vector_spec.shard_count = 1;
  POMAI_EXPECT_OK(mgr.CreateMembrane(vector_spec));
  POMAI_EXPECT_OK(mgr.OpenMembrane("faces"));

  MembraneSpec graph_spec;
  graph_spec.name = "access";
  graph_spec.kind = MembraneKind::kGraph;
  graph_spec.dim = 4;
  graph_spec.shard_count = 1;
  POMAI_EXPECT_OK(mgr.CreateMembrane(graph_spec));
  POMAI_EXPECT_OK(mgr.OpenMembrane("access"));

  std::vector<float> v{1.0f, 0.0f, 0.0f, 0.0f};
  POMAI_EXPECT_OK(mgr.PutVector("faces", 42, v));

  POMAI_EXPECT_OK(mgr.AddVertex("access", 42, 1, Metadata{}));
  POMAI_EXPECT_OK(mgr.AddVertex("access", 900, 1, Metadata{}));
  POMAI_EXPECT_OK(mgr.AddEdge("access", 42, 900, 1, 1, Metadata{}));

  MultiModalQuery q;
  q.vector = v;
  q.top_k = 1;
  q.graph_hops = 1;
  q.vector_membrane = "faces";
  q.graph_membrane = "access";

  SearchResult out;
  POMAI_EXPECT_OK(mgr.SearchMultiModal("faces", q, &out));
  POMAI_EXPECT_EQ(out.hits.size(), 1u);
  POMAI_EXPECT_EQ(out.hits[0].id, static_cast<VectorId>(42));
  POMAI_EXPECT_TRUE(!out.hits[0].related_ids.empty());
}

POMAI_TEST(QueryOrchestrator_AnomalyWipesKeys) {
  std::array<std::uint8_t, 32> key{};
  for (std::size_t i = 0; i < key.size(); ++i) key[i] = static_cast<std::uint8_t>(i + 1);
  core::KeyManager::Global().SetKey(key);
  POMAI_EXPECT_TRUE(core::KeyManager::Global().IsArmed());

  DBOptions opt;
  opt.path = test::TempDir("orchestrator-anomaly");
  opt.dim = 4;
  opt.shard_count = 1;
  core::MembraneManager mgr(opt);
  POMAI_EXPECT_OK(mgr.Open());

  MultiModalQuery q;
  q.vector.assign(5000, 0.1f); // triggers anomaly heuristic
  q.top_k = 10;
  SearchResult out;
  Status st = mgr.SearchMultiModal("__default__", q, &out);
  POMAI_EXPECT_TRUE(!st.ok());
  POMAI_EXPECT_TRUE(!core::KeyManager::Global().IsArmed());
}

}  // namespace
}  // namespace pomai

