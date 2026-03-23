#pragma once

#include <string_view>

#include "core/query/query_planner.h"

namespace pomai::core {

class QueryOrchestrator {
public:
    explicit QueryOrchestrator(IQueryEngine* engine, std::size_t max_frontier = 2048)
        : planner_(engine), engine_(engine), max_frontier_(max_frontier) {}
    Status Execute(std::string_view default_membrane, const pomai::MultiModalQuery& query, pomai::SearchResult* out);

private:
    QueryPlanner planner_;
    IQueryEngine* engine_;
    std::size_t max_frontier_ = 2048;
};

} // namespace pomai::core

