#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include "pomai/status.h"

namespace pomai::core {

struct LinkedObject {
    uint64_t vector_id = 0;
    uint64_t graph_vertex_id = 0;
    uint64_t mesh_id = 0;
};

class ObjectLinker {
public:
    Status LinkByGid(std::string gid, uint64_t vector_id, uint64_t graph_vertex_id, uint64_t mesh_id);
    Status UnlinkByGid(const std::string& gid);

    std::optional<LinkedObject> ResolveByGid(const std::string& gid) const;
    std::optional<LinkedObject> ResolveByVectorId(uint64_t vector_id) const;
    std::optional<LinkedObject> ResolveByGraphVertexId(uint64_t graph_vertex_id) const;
    std::optional<LinkedObject> ResolveByMeshId(uint64_t mesh_id) const;

private:
    std::unordered_map<std::string, LinkedObject> by_gid_;
    std::unordered_map<uint64_t, std::string> gid_by_vector_;
    std::unordered_map<uint64_t, std::string> gid_by_graph_vertex_;
    std::unordered_map<uint64_t, std::string> gid_by_mesh_;
};

} // namespace pomai::core
