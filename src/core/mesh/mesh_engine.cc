#include "core/mesh/mesh_engine.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "core/simd/simd_dispatch.h"

namespace pomai::core {

MeshEngine::MeshEngine(std::string path, std::size_t max_meshes)
    : path_(std::move(path)), max_meshes_(max_meshes) {}

Status MeshEngine::Open() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(path_, ec);
    if (ec) return Status::IOError("mesh create dir failed");
    std::ifstream in(path_ + "/mesh.log", std::ios::binary);
    if (!in.good()) return Status::Ok();
    while (true) {
        std::uint64_t id = 0;
        std::uint32_t n = 0;
        if (!in.read(reinterpret_cast<char*>(&id), sizeof(id))) break;
        if (!in.read(reinterpret_cast<char*>(&n), sizeof(n))) break;
        std::vector<float> xyz(n);
        if (n > 0 && !in.read(reinterpret_cast<char*>(xyz.data()), static_cast<std::streamsize>(n * sizeof(float)))) break;
        meshes_[id] = std::move(xyz);
        if (max_meshes_ > 0 && meshes_.size() > max_meshes_) meshes_.erase(meshes_.begin());
    }
    return Status::Ok();
}

Status MeshEngine::Close() { return Status::Ok(); }

Status MeshEngine::Put(std::uint64_t mesh_id, std::span<const float> xyz) {
    if (xyz.empty() || xyz.size() % 3 != 0) return Status::InvalidArgument("mesh xyz must be multiple of 3");
    if (max_meshes_ > 0 && meshes_.size() >= max_meshes_ && meshes_.find(mesh_id) == meshes_.end()) meshes_.erase(meshes_.begin());
    meshes_[mesh_id] = std::vector<float>(xyz.begin(), xyz.end());
    std::ofstream out(path_ + "/mesh.log", std::ios::binary | std::ios::app);
    if (!out.good()) return Status::IOError("mesh append failed");
    std::uint32_t n = static_cast<std::uint32_t>(xyz.size());
    out.write(reinterpret_cast<const char*>(&mesh_id), sizeof(mesh_id));
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(xyz.data()), static_cast<std::streamsize>(xyz.size_bytes()));
    return Status::Ok();
}

Status MeshEngine::Rmsd(std::uint64_t mesh_a, std::uint64_t mesh_b, double* out) const {
    if (!out) return Status::InvalidArgument("mesh out null");
    auto ia = meshes_.find(mesh_a);
    auto ib = meshes_.find(mesh_b);
    if (ia == meshes_.end() || ib == meshes_.end()) return Status::NotFound("mesh not found");
    if (ia->second.size() != ib->second.size()) return Status::InvalidArgument("mesh size mismatch");
    *out = simd::MeshRmsdF32(ia->second.data(), ib->second.data(), ia->second.size() / 3);
    return Status::Ok();
}

Status MeshEngine::Intersect(std::uint64_t mesh_a, std::uint64_t mesh_b, bool* out) const {
    if (!out) return Status::InvalidArgument("mesh out null");
    auto ia = meshes_.find(mesh_a);
    auto ib = meshes_.find(mesh_b);
    if (ia == meshes_.end() || ib == meshes_.end()) return Status::NotFound("mesh not found");
    auto bbox = [](const std::vector<float>& m, double* minx, double* miny, double* minz, double* maxx, double* maxy, double* maxz) {
        *minx = *maxx = m[0];
        *miny = *maxy = m[1];
        *minz = *maxz = m[2];
        for (std::size_t i = 3; i < m.size(); i += 3) {
            *minx = std::min<double>(*minx, m[i]);
            *miny = std::min<double>(*miny, m[i + 1]);
            *minz = std::min<double>(*minz, m[i + 2]);
            *maxx = std::max<double>(*maxx, m[i]);
            *maxy = std::max<double>(*maxy, m[i + 1]);
            *maxz = std::max<double>(*maxz, m[i + 2]);
        }
    };
    double a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5;
    bbox(ia->second, &a0,&a1,&a2,&a3,&a4,&a5);
    bbox(ib->second, &b0,&b1,&b2,&b3,&b4,&b5);
    *out = (a0 <= b3 && a3 >= b0) && (a1 <= b4 && a4 >= b1) && (a2 <= b5 && a5 >= b2);
    return Status::Ok();
}

Status MeshEngine::Volume(std::uint64_t mesh_id, double* out) const {
    if (!out) return Status::InvalidArgument("mesh out null");
    auto it = meshes_.find(mesh_id);
    if (it == meshes_.end()) return Status::NotFound("mesh not found");
    if (it->second.size() < 3) return Status::InvalidArgument("mesh empty");
    double minx = it->second[0], maxx = it->second[0];
    double miny = it->second[1], maxy = it->second[1];
    double minz = it->second[2], maxz = it->second[2];
    for (std::size_t i = 3; i < it->second.size(); i += 3) {
        minx = std::min<double>(minx, it->second[i]);
        miny = std::min<double>(miny, it->second[i + 1]);
        minz = std::min<double>(minz, it->second[i + 2]);
        maxx = std::max<double>(maxx, it->second[i]);
        maxy = std::max<double>(maxy, it->second[i + 1]);
        maxz = std::max<double>(maxz, it->second[i + 2]);
    }
    *out = (maxx - minx) * (maxy - miny) * (maxz - minz);
    return Status::Ok();
}

} // namespace pomai::core

