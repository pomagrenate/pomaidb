#include "tests/common/test_main.h"

#include "core/query/query_planner.h"
#include "core/storage/compaction_manager.h"
#include "pomai/graph.h"
#include "pomai/search.h"

namespace pomai::core {

namespace {
class MockQueryEngine final : public IQueryEngine {
public:
    Status Search(std::string_view, std::span<const float>, uint32_t, const pomai::SearchOptions&, pomai::SearchResult* out) override {
        out->hits = {{1, 0.8f, {}}, {2, 0.6f, {}}, {3, 0.2f, {}}};
        return Status::Ok();
    }
    Status SearchLexical(std::string_view, const std::string&, uint32_t, std::vector<LexicalHit>* out) override {
        out->clear();
        return Status::Ok();
    }
    Status GetNeighbors(std::string_view, VertexId, std::vector<pomai::Neighbor>* out) override {
        out->clear();
        return Status::Ok();
    }
    Status GetNeighbors(std::string_view, VertexId, EdgeType, std::vector<pomai::Neighbor>* out) override {
        out->clear();
        return Status::Ok();
    }
};
} // namespace

POMAI_TEST(EdgeAICore_MetadataAsOfAndPartitionFilters) {
    pomai::Metadata m;
    m.timestamp = 100;
    m.lsn = 55;
    m.device_id = "dev-A";
    m.location_id = "zone-1";

    pomai::SearchOptions opts;
    opts.as_of_ts = 120;
    opts.as_of_lsn = 60;
    opts.partition_device_id = "dev-A";
    opts.partition_location_id = "zone-1";
    POMAI_EXPECT_TRUE(opts.Matches(m));

    opts.as_of_ts = 80;
    POMAI_EXPECT_TRUE(!opts.Matches(m));
}

POMAI_TEST(EdgeAICore_QueryPlannerAggregatesOnHits) {
    MockQueryEngine engine;
    QueryPlanner planner(&engine);
    pomai::MultiModalQuery q;
    q.vector = {0.1f, 0.2f, 0.3f};
    q.top_k = 3;
    q.aggregates.push_back({pomai::AggregateOp::kCount, "score", 0});
    q.aggregates.push_back({pomai::AggregateOp::kSum, "score", 0});
    q.aggregates.push_back({pomai::AggregateOp::kTopK, "score", 2});

    pomai::SearchResult out;
    POMAI_EXPECT_OK(planner.Execute("__default__", q, &out));
    POMAI_EXPECT_EQ(out.aggregates.size(), static_cast<size_t>(3));
    POMAI_EXPECT_TRUE(out.aggregates[0].value >= 3.0);
    POMAI_EXPECT_TRUE(out.aggregates[1].value > 1.5);
    POMAI_EXPECT_EQ(out.aggregates[2].topk_hits.size(), static_cast<size_t>(2));
}

POMAI_TEST(EdgeAICore_CompactionBiasInfluencesPriority) {
    storage::CompactionManager mgr;
    std::vector<storage::CompactionManager::LevelStats> stats = {
        {0, 128 * 1024 * 1024, 8, 2.0, 10.0}, // high score but heavily biased
        {1, 64 * 1024 * 1024, 4, 1.2, 1.0},
    };
    auto task = mgr.PickCompaction(stats);
    POMAI_EXPECT_TRUE(task.valid);
    POMAI_EXPECT_EQ(task.input_level, 1);
}

} // namespace pomai::core
