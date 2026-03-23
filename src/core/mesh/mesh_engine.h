#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "pomai/status.h"

namespace pomai::core {

class MeshEngine {
public:
    MeshEngine(std::string path, std::size_t max_meshes);
    Status Open();
    Status Close();
    Status Put(std::uint64_t mesh_id, std::span<const float> xyz);
    Status Rmsd(std::uint64_t mesh_a, std::uint64_t mesh_b, double* out) const;
    Status Intersect(std::uint64_t mesh_a, std::uint64_t mesh_b, bool* out) const;
    Status Volume(std::uint64_t mesh_id, double* out) const;

private:
    std::string path_;
    std::size_t max_meshes_;
    std::unordered_map<std::uint64_t, std::vector<float>> meshes_;
};

} // namespace pomai::core

