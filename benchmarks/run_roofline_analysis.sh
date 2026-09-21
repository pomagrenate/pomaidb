#!/bin/bash
# run_roofline_analysis.sh — Run PomaiDB Roofline Analysis with Linux perf integration
#
# This script runs the roofline analysis benchmark with Linux perf counters
# to collect Top-Down Microarchitecture Analysis (TMAM) metrics.
#
# Copyright 2026 PomaiDB authors. MIT License.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
NUM_VECTORS=10000
DIMENSION=128
NUM_QUERIES=100
TOP_K=10
USE_HNSW=false
USE_QUANTIZATION=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vectors)
            NUM_VECTORS="$2"
            shift 2
            ;;
        --dimension)
            DIMENSION="$2"
            shift 2
            ;;
        --queries)
            NUM_QUERIES="$2"
            shift 2
            ;;
        --topk)
            TOP_K="$2"
            shift 2
            ;;
        --hnsw)
            USE_HNSW=true
            shift
            ;;
        --no-quantization)
            USE_QUANTIZATION=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --vectors N       Number of vectors (default: 10000)"
            echo "  --dimension D     Vector dimension (default: 128)"
            echo "  --queries N       Number of queries (default: 100)"
            echo "  --topk K          Top-K results (default: 10)"
            echo "  --hnsw            Enable HNSW analysis"
            echo "  --no-quantization Disable SQ8 quantization"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=== PomaiDB Roofline Analysis with Linux perf ===${NC}"
echo ""
echo "Configuration:"
echo "  Vectors: $NUM_VECTORS"
echo "  Dimension: $DIMENSION"
echo "  Queries: $NUM_QUERIES"
echo "  Top-K: $TOP_K"
echo "  HNSW: $USE_HNSW"
echo "  Quantization: $USE_QUANTIZATION"
echo ""

# Build the benchmark
echo -e "${YELLOW}Building benchmark...${NC}"
cd "$(dirname "$0")/../.."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOMAI_BUILD_BENCH=ON
make roofline_analysis_bench -j$(nproc)
echo ""

# Check if perf is available
if ! command -v perf &> /dev/null; then
    echo -e "${RED}Error: perf is not available. Please install linux-tools-generic.${NC}"
    echo "  Ubuntu/Debian: sudo apt-get install linux-tools-generic"
    echo "  Fedora: sudo dnf install perf"
    exit 1
fi

# Build command line arguments
CMD_ARGS="--vectors $NUM_VECTORS --dimension $DIMENSION --queries $NUM_QUERIES --topk $TOP_K"
if [ "$USE_HNSW" = true ]; then
    CMD_ARGS="$CMD_ARGS --hnsw"
fi
if [ "$USE_QUANTIZATION" = false ]; then
    CMD_ARGS="$CMD_ARGS --no-quantization"
fi

echo -e "${GREEN}Running Roofline Analysis Benchmark...${NC}"
echo ""

# Run basic benchmark
echo -e "${BLUE}=== Basic Benchmark Run ===${NC}"
./benchmarks/roofline_analysis_bench $CMD_ARGS
echo ""

# Run with perf for detailed CPU pipeline analysis
echo -e "${BLUE}=== CPU Pipeline Analysis (TMAM) ===${NC}"
echo "Collecting performance counters..."
echo ""

# Detect architecture
ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
    # x86_64 specific perf events
    perf stat -e \
        cycles,instructions,branches,branch-misses,\
        L1-dcache-loads,L1-dcache-load-misses,\
        LLC-loads,LLC-load-misses,\
        dTLB-loads,dtlb-load-misses,\
        cache-references,cache-misses \
        ./benchmarks/roofline_analysis_bench $CMD_ARGS
elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    # ARM64 specific perf events
    perf stat -e \
        cycles,instructions,branches,branch-misses,\
        L1-dcache-loads,L1-dcache-load-misses,\
        l2d_cache,l2d_cache_refill,\
        LLCCACHE,LLCCACHE_REFILL \
        ./benchmarks/roofline_analysis_bench $CMD_ARGS
else
    echo -e "${YELLOW}Warning: Unknown architecture $ARCH, using generic perf events${NC}"
    perf stat -e \
        cycles,instructions,branches,branch-misses,\
        cache-references,cache-misses \
        ./benchmarks/roofline_analysis_bench $CMD_ARGS
fi

echo ""
echo -e "${GREEN}=== Roofline Analysis Complete ===${NC}"
echo ""
echo "Key Metrics to Analyze:"
echo "1. Effective Bandwidth: Compare against system peak (15-35 GB/s for embedded)"
echo "2. IPC (Instructions Per Cycle): Target ≥2.0 for SIMD, <0.8 indicates memory-bound"
echo "3. LLC Miss Rate: Target <5% for cached working sets, >50% indicates cache thrashing"
echo "4. Branch Miss Rate: Target <1-2%, higher indicates need for branchless code"
echo ""
echo "For detailed analysis, check if:"
echo "- Bandwidth saturation ≥80%: Memory subsystem is optimally utilized"
echo "- IPC ≥2.5: SIMD units are well-utilized (compute-bound)"
echo "- LLC miss rate <10%: Working set fits in cache (cache-bound)"
echo "- Branch miss rate <2%: Branch predictor is effective (frontend-bound)"