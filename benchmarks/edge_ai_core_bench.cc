#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include "core/query/query_planner.h"
#include "pomai/graph.h"

namespace pomai::core {
namespace {
class BenchEngine final : public IQueryEngine {
public:
    Status Search(std::string_view, std::span<const float>, uint32_t topk, const pomai::SearchOptions&, pomai::SearchResult* out) override {
        out->hits.clear();
        out->hits.reserve(topk);
        for (uint32_t i = 0; i < topk; ++i) {
            out->hits.push_back({static_cast<VectorId>(i + 1), 1.0f / static_cast<float>(i + 1), {}});
        }
        return Status::Ok();
    }
    Status SearchLexical(std::string_view, const std::string&, uint32_t, std::vector<LexicalHit>* out) override { out->clear(); return Status::Ok(); }
    Status GetNeighbors(std::string_view, VertexId, std::vector<pomai::Neighbor>* out) override { out->clear(); return Status::Ok(); }
    Status GetNeighbors(std::string_view, VertexId, EdgeType, std::vector<pomai::Neighbor>* out) override { out->clear(); return Status::Ok(); }
};
} // namespace
} // namespace pomai::core

int main() {
    pomai::core::BenchEngine engine;
    pomai::core::QueryPlanner planner(&engine);
    pomai::MultiModalQuery q;
    q.vector = {0.01f, 0.02f, 0.03f, 0.04f};
    q.top_k = 256;
    q.aggregates = {
        {pomai::AggregateOp::kSum, "score", 0},
        {pomai::AggregateOp::kAvg, "score", 0},
        {pomai::AggregateOp::kTopK, "score", 16},
    };

    constexpr int kIters = 20000;
    pomai::SearchResult out;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        auto st = planner.Execute("__default__", q, &out);
        if (!st.ok()) {
            std::cerr << "planner failed: " << st.message() << "\n";
            return 1;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "edge_ai_core_bench\n";
    std::cout << "iterations: " << kIters << "\n";
    std::cout << "seconds: " << sec << "\n";
    std::cout << "queries_per_sec: " << (static_cast<double>(kIters) / sec) << "\n";
    return 0;
}
