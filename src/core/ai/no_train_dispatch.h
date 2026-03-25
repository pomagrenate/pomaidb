#pragma once

#include <span>

#include "pomai/options.h"
#include "pomai/status.h"

namespace pomai::core::ai {

struct InferenceSummary {
    float score = 0.0f;
    bool action_required = false;
    const char* label = "stable";
    const char* explanation = "No-train heuristic baseline";
};

Status InferNoTrainForKind(MembraneKind kind, std::span<const float> features, InferenceSummary* out);

} // namespace pomai::core::ai
