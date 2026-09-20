// pomai/pomegranate_query.h — Pomegranate 5-Stage Query Orchestrator
//
// In the Pomegranate Engine:
// 1. Orient: Compass locates closest Locule centroids.
// 2. Peel: Bounding-sphere pruning eliminates distant Locules.
// 3. Taste: Approximate SIMD SQ8 search on Aril Pulp + Rind MemTable scan.
// 4. Pick: Candidate selection into bounded priority pool.
// 5. Rerank: Exact FP32 metric calculation on Seed Kernel for final top-K.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "fruit_map.h"
#include "options.h"
#include "rind.h"
#include "search.h"
#include "status.h"
#include "types.h"
#include <ptask/ptask.h>

namespace pomai::query {

class PomegranateQuery {
public:
    static Status Execute(std::span<const float> query,
                          uint32_t topk,
                          const SearchOptions& opts,
                          MetricType metric,
                          const manifest::FruitSnapshot* snapshot,
                          const ingest::Rind* rind,
                          SearchHitSink& sink,
                          ptask::ThreadPool* thread_pool = nullptr);

    static Status Execute(std::span<const float> query,
                          uint32_t topk,
                          const SearchOptions& opts,
                          MetricType metric,
                          const manifest::FruitSnapshot* snapshot,
                          const ingest::Rind* rind,
                          SearchResult* out,
                          ptask::ThreadPool* thread_pool = nullptr);
};

} // namespace pomai::query
