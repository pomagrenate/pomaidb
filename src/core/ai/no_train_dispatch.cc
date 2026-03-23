#include "core/ai/no_train_dispatch.h"

#include <algorithm>
#include <numeric>

namespace pomai::core::ai {

Status InferNoTrainForKind(MembraneKind kind, std::span<const float> features, InferenceSummary* out) {
    if (!out) return Status::InvalidArgument("out is null");
    if (features.empty()) return Status::InvalidArgument("features must not be empty");

    const float sum = std::accumulate(features.begin(), features.end(), 0.0f);
    const float mean = sum / static_cast<float>(features.size());
    const float maxv = *std::max_element(features.begin(), features.end());
    const float minv = *std::min_element(features.begin(), features.end());
    const float range = maxv - minv;
    const float normalized = std::clamp((mean + range) * 0.25f, 0.0f, 1.0f);

    out->score = normalized;
    out->action_required = false;
    out->label = "stable";
    out->explanation = "No-train heuristic baseline";

    switch (kind) {
        case MembraneKind::kVector:
            out->label = normalized > 0.7f ? "high_similarity" : "normal_similarity";
            out->action_required = normalized > 0.92f;
            out->explanation = "Vector confidence from centroid/spread";
            break;
        case MembraneKind::kRag:
            out->label = normalized > 0.6f ? "retrieval_confident" : "retrieval_weak";
            out->action_required = normalized < 0.25f;
            out->explanation = "RAG relevance from retrieval compactness";
            break;
        case MembraneKind::kGraph:
            out->label = range > 0.8f ? "graph_anomaly_risk" : "graph_normal";
            out->action_required = range > 1.25f;
            out->explanation = "Graph risk from neighborhood variance";
            break;
        case MembraneKind::kText:
            out->label = normalized > 0.5f ? "lexical_precise" : "lexical_ambiguous";
            out->action_required = normalized < 0.2f;
            out->explanation = "Text certainty from lexical density";
            break;
        case MembraneKind::kTimeSeries:
            out->label = range > 0.9f ? "timeseries_spike" : "timeseries_stable";
            out->action_required = range > 1.4f;
            out->explanation = "Time-series anomaly from window spread";
            break;
        case MembraneKind::kKeyValue:
            out->label = normalized > 0.7f ? "config_drift" : "config_stable";
            out->action_required = normalized > 0.85f;
            out->explanation = "KV drift from deterministic stats";
            break;
        case MembraneKind::kSketch:
            out->label = maxv > 0.8f ? "heavy_hitter_detected" : "sketch_nominal";
            out->action_required = maxv > 0.95f;
            out->explanation = "Sketch pressure from estimate features";
            break;
        case MembraneKind::kBlob:
            out->label = range > 0.7f ? "blob_entropy_high" : "blob_entropy_normal";
            out->action_required = mean < 0.1f;
            out->explanation = "Blob entropy proxy from byte stats";
            break;
        case MembraneKind::kSpatial:
            out->label = normalized > 0.6f ? "inside_safe_zone" : "outside_safe_zone";
            out->action_required = normalized < 0.35f;
            out->explanation = "Spatial safety from geofence distance features";
            break;
        case MembraneKind::kMesh:
            out->label = range > 0.6f ? "mesh_collision_risk" : "mesh_clearance_ok";
            out->action_required = range > 1.1f;
            out->explanation = "Mesh risk from geometric spread";
            break;
        case MembraneKind::kSparse:
            out->label = mean < 0.25f ? "sparse_very_sparse" : "sparse_active";
            out->action_required = mean > 0.9f;
            out->explanation = "Sparse state from nnz/dot features";
            break;
        case MembraneKind::kBitset:
            out->label = mean > 0.5f ? "bitset_filter_dense" : "bitset_filter_selective";
            out->action_required = mean > 0.95f;
            out->explanation = "Bitset selectivity from logical-op features";
            break;
        default:
            return Status::InvalidArgument("unsupported membrane kind");
    }

    return Status::Ok();
}

} // namespace pomai::core::ai
