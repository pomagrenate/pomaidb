// include/pvec/pvec.h — Master umbrella header for pvec library
// High-Performance SIMD Vector Mathematics and Quantization Library
// Copyright 2026 PomaiDB / pvec authors. MIT License.

#pragma once

#include "pvec_platform.h"
#include "pvec_distance.h"
#include "pvec_quant_sq.h"
#include "pvec_quant_fp16.h"
#include "pvec_quant_bit.h"
#include "pvec_kmeans.h"
#include "pvec_quant_pq.h"

namespace pvec {

constexpr const char* kVersion = "0.1.0";

} // namespace pvec
