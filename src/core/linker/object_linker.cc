#include "core/linker/object_linker.h"

namespace pomai::core {

Status ObjectLinker::LinkByGid(std::string gid, uint64_t vector_id, uint64_t graph_vertex_id, uint64_t mesh_id) {
    if (gid.empty()) return Status::InvalidArgument("gid must not be empty");
    if (vector_id == 0 || graph_vertex_id == 0 || mesh_id == 0) {
        return Status::InvalidArgument("vector_id/graph_vertex_id/mesh_id must be non-zero");
    }

    const auto old = ResolveByGid(gid);
    if (old.has_value()) {
        gid_by_vector_.erase(old->vector_id);
        gid_by_graph_vertex_.erase(old->graph_vertex_id);
        gid_by_mesh_.erase(old->mesh_id);
    }

    LinkedObject linked;
    linked.vector_id = vector_id;
    linked.graph_vertex_id = graph_vertex_id;
    linked.mesh_id = mesh_id;

    by_gid_[gid] = linked;
    gid_by_vector_[vector_id] = gid;
    gid_by_graph_vertex_[graph_vertex_id] = gid;
    gid_by_mesh_[mesh_id] = gid;
    return Status::Ok();
}

Status ObjectLinker::UnlinkByGid(const std::string& gid) {
    auto it = by_gid_.find(gid);
    if (it == by_gid_.end()) return Status::NotFound("gid not found");
    gid_by_vector_.erase(it->second.vector_id);
    gid_by_graph_vertex_.erase(it->second.graph_vertex_id);
    gid_by_mesh_.erase(it->second.mesh_id);
    by_gid_.erase(it);
    return Status::Ok();
}

std::optional<LinkedObject> ObjectLinker::ResolveByGid(const std::string& gid) const {
    auto it = by_gid_.find(gid);
    if (it == by_gid_.end()) return std::nullopt;
    return it->second;
}

std::optional<LinkedObject> ObjectLinker::ResolveByVectorId(uint64_t vector_id) const {
    auto it = gid_by_vector_.find(vector_id);
    if (it == gid_by_vector_.end()) return std::nullopt;
    return ResolveByGid(it->second);
}

std::optional<LinkedObject> ObjectLinker::ResolveByGraphVertexId(uint64_t graph_vertex_id) const {
    auto it = gid_by_graph_vertex_.find(graph_vertex_id);
    if (it == gid_by_graph_vertex_.end()) return std::nullopt;
    return ResolveByGid(it->second);
}

std::optional<LinkedObject> ObjectLinker::ResolveByMeshId(uint64_t mesh_id) const {
    auto it = gid_by_mesh_.find(mesh_id);
    if (it == gid_by_mesh_.end()) return std::nullopt;
    return ResolveByGid(it->second);
}

} // namespace pomai::core
