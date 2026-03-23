#pragma once

#include <cstddef>
#include <cstdint>

namespace pomai::core::simd {

double HaversineMeters(double lat1, double lon1, double lat2, double lon2);
double MeshRmsdF32(const float* a_xyz, const float* b_xyz, std::size_t points);
double SparseDotU32F32(const std::uint32_t* a_idx, const float* a_w, std::size_t a_n,
                       const std::uint32_t* b_idx, const float* b_w, std::size_t b_n);
std::uint32_t SparseIntersectU32(const std::uint32_t* a_idx, std::size_t a_n,
                                 const std::uint32_t* b_idx, std::size_t b_n);
double BitsetHamming(const std::uint8_t* a, const std::uint8_t* b, std::size_t n_bytes);
double BitsetJaccard(const std::uint8_t* a, const std::uint8_t* b, std::size_t n_bytes);

} // namespace pomai::core::simd

