#pragma once

#include <memory>
#include "core/kernel/pod.h"
#include "core/query/query_planner.h"

namespace pomai::core {

    /**
     * QueryPod: Orchestrates complex searches.
     * Wraps the QueryPlanner logic.
     */
    class QueryPod : public Pod {
    public:
        explicit QueryPod(std::unique_ptr<QueryPlanner> planner)
            : planner_(std::move(planner)) {}

        void Handle(Message&& msg) override;

        PodId Id() const override { return PodId::kQuery; }
        std::string Name() const override { return "QueryService"; }

        MemoryQuota GetQuota() const override { return {0, 0}; }

    private:
        std::unique_ptr<QueryPlanner> planner_;
    };

} // namespace pomai::core
