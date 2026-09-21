# PomaiDB Roofline Model & TMAM Analysis Benchmark

## Overview

This benchmark applies the **Roofline Model** and **Top-Down Microarchitecture Analysis (TMAM)** to evaluate whether PomaiDB's storage and query pipelines are hardware-saturated. This analysis provides definitive evidence of whether your vector database operations are limited by software overhead or have hit the physical limits of the underlying hardware.

## What This Benchmark Measures

### 1. Memory Bandwidth Saturation
- **Effective Bandwidth**: Actual memory bandwidth achieved during vector scans
- **Saturation Percentage**: How close you are to the hardware's peak memory bandwidth
- **Operational Intensity**: FLOPs/Byte ratio that determines if you're memory-bound or compute-bound

### 2. CPU Pipeline Efficiency
- **IPC (Instructions Per Cycle)**: Measures CPU utilization efficiency
- **Cache Miss Rates**: L1, L2, LLC miss rates to evaluate cache effectiveness
- **Branch Prediction**: Branch miss rate to evaluate frontend efficiency

### 3. HNSW Graph Traversal Analysis
- **Effective Bandwidth**: Memory bandwidth during random graph access
- **Cache Hit Rate**: How well the graph structure utilizes CPU caches
- **Hop Count Analysis**: Average number of graph hops per query

### 4. Hardware Saturation Matrix
- **Subsystem Classification**: Memory-bound, compute-bound, cache-bound, latency-bound
- **Saturation Status**: Excellence rating for each hardware subsystem
- **Optimization Guidance**: Specific recommendations based on bottlenecks

## Building the Benchmark

### Prerequisites
- CMake 3.20+
- C++20 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- Linux perf (for detailed CPU pipeline analysis on Linux)
- PomaiDB dependencies (Vulkan headers, palloc submodule)

### Build Steps

```bash
# Navigate to PomaiDB root
cd /path/to/pomaidb

# Build with benchmark enabled
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOMAI_BUILD_BENCH=ON
make roofline_analysis_bench -j$(nproc)
```

### Quick Test Run

```bash
# Basic test with default parameters
./benchmarks/roofline_analysis_bench

# Custom parameters
./benchmarks/roofline_analysis_bench --vectors 50000 --dimension 256 --queries 200 --topk 20
```

## Running with Linux perf Integration

For detailed CPU pipeline analysis, use the provided shell script:

```bash
# Make script executable
chmod +x benchmarks/run_roofline_analysis.sh

# Run with default parameters
./benchmarks/run_roofline_analysis.sh

# Run with custom parameters
./benchmarks/run_roofline_analysis.sh --vectors 25000 --dimension 128 --hnsw
```

### Manual perf Execution

```bash
# x86_64 architecture
perf stat -e \
  cycles,instructions,branches,branch-misses,\
  L1-dcache-loads,L1-dcache-load-misses,\
  LLC-loads,LLC-load-misses \
  ./benchmarks/roofline_analysis_bench --vectors 10000

# ARM64 architecture (Raspberry Pi, etc.)
perf stat -e \
  cycles,instructions,branches,branch-misses,\
  L1-dcache-loads,L1-dcache-load-misses,\
  l2d_cache,l2d_cache_refill \
  ./benchmarks/roofline_analysis_bench --vectors 10000
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--vectors N` | Number of vectors in database | 10000 |
| `--dimension D` | Vector dimension | 128 |
| `--queries N` | Number of query vectors | 100 |
| `--topk K` | Top-K results per query | 10 |
| `--hnsw` | Enable HNSW graph analysis | false |
| `--no-quantization` | Disable SQ8 quantization | false (enabled) |
| `--help` | Show help message | - |

## Interpreting Results

### Memory Bandwidth Saturation

**What to Look For:**
- **Bandwidth Saturation ≥80%**: EXCELLENT - Memory subsystem optimally utilized
- **Bandwidth Saturation 50-80%**: GOOD - Good memory utilization, room for optimization
- **Bandwidth Saturation <50%**: POOR - Memory subsystem underutilized, software overhead

**Example Interpretation:**
```
Effective Bandwidth: 16.5 GB/s
Peak Bandwidth: 20.0 GB/s
Saturation: 82.5%
```
This indicates EXCELLENT memory bandwidth utilization. The database is achieving 82.5% of the hardware's peak memory bandwidth, which is optimal for memory-bound operations.

### Operational Intensity Analysis

**Roofline Knee Calculation:**
```
Knee = Peak Compute / Peak Bandwidth
Knee = 50 GFLOPs / 20 GB/s = 2.5 FLOPs/Byte
```

**FP32 Scan (0.5 FLOPs/Byte):**
- Since 0.5 < 2.5, FP32 scans are **MEMORY-BOUND**
- Performance limited by memory bandwidth, not compute
- Optimization focus: reduce memory traffic, improve cache locality

**INT8 Scan (2.0 FLOPs/Byte):**
- For knee = 2.5, INT8 is still **MEMORY-BOUND** but closer to compute-bound
- Better efficiency than FP32 due to higher operational intensity

### CPU Pipeline Efficiency

**IPC (Instructions Per Cycle):**
- **IPC ≥2.5**: EXCELLENT - SIMD units well-utilized (compute-bound)
- **IPC 1.5-2.5**: GOOD - Moderate compute utilization
- **IPC <0.8**: POOR - CPU stalled waiting on memory (memory-bound)

**Cache Miss Rates:**
- **LLC Miss Rate <5%**: EXCELLENT - Working set fits in cache
- **LLC Miss Rate 5-30%**: GOOD - Reasonable cache utilization
- **LLC Miss Rate >50%**: POOR - Cache thrashing, working set too large

**Branch Prediction:**
- **Branch Miss Rate <1%**: EXCELLENT - Branch predictor effective
- **Branch Miss Rate 1-2%**: GOOD - Acceptable branch prediction
- **Branch Miss Rate >2%**: POOR - Consider branchless code

### Hardware Saturation Matrix

The benchmark provides a matrix showing:

| Subsystem | Bottleneck Type | Saturation Status |
|-----------|----------------|-------------------|
| Flat/Quantized Scan | Memory-Bandwidth Bound | EXCELLENT/GOOD/POOR |
| SIMD Re-Ranking | Compute/L1 Bound | EXCELLENT/GOOD/POOR |
| Cluster Routing | L2 Cache Bound | EXCELLENT/GOOD/POOR |
| HNSW Traversal | Memory-Latency Bound | EXPECTED/POOR |
| Branch Prediction | Frontend Bound | EXCELLENT/GOOD/POOR |

## Embedded Device Hardware Reference

### Typical Embedded Hardware Specifications

**Raspberry Pi 4 (ARM Cortex-A72):**
- Peak Memory Bandwidth: ~4 GB/s (LPDDR4)
- Peak Compute: ~13 GFLOPs/core (NEON)
- L1 Cache: 32 KB
- L2 Cache: 256 KB
- LLC Cache: 1 MB (shared)

**Raspberry Pi 5 (ARM Cortex-A76):**
- Peak Memory Bandwidth: ~13 GB/s (LPDDR4-3200)
- Peak Compute: ~24 GFLOPs/core (NEON)
- L1 Cache: 32 KB
- L2 Cache: 512 KB
- LLC Cache: 2 MB (shared)

**Intel Celeron/N-series (Low-power x86):**
- Peak Memory Bandwidth: ~15-25 GB/s (DDR4)
- Peak Compute: ~30-50 GFLOPs/core (AVX2)
- L1 Cache: 32 KB
- L2 Cache: 256 KB
- LLC Cache: 2-4 MB (shared)

### Expected Results for Embedded Devices

**Memory Bandwidth Saturation:**
- Raspberry Pi 4: 60-80% of 4 GB/s = 2.4-3.2 GB/s
- Raspberry Pi 5: 70-85% of 13 GB/s = 9.1-11.1 GB/s
- Intel Celeron: 75-90% of 20 GB/s = 15-18 GB/s

**IPC Targets:**
- Memory-bound operations: 0.5-1.0 IPC
- Compute-bound operations: 2.0-3.0 IPC

**Cache Miss Rates:**
- Small datasets (<10K vectors): <5% LLC miss rate
- Medium datasets (10K-50K vectors): 10-30% LLC miss rate
- Large datasets (>50K vectors): >50% LLC miss rate

## Optimization Guidance Based on Results

### If Memory-Bandwidth Bound (Bandwidth Saturation <50%)

**Potential Issues:**
- Inefficient memory access patterns
- Poor cache locality
- Excessive memory traffic

**Optimization Strategies:**
1. **Improve Cache Locality**: Restructure data layout for better spatial locality
2. **Reduce Memory Traffic**: Use more aggressive quantization (INT8, binary)
3. **Memory Pooling**: Implement memory pools to reduce allocation overhead
4. **Prefetching**: Add software prefetching for memory access patterns

### If Compute-Bound (IPC <1.0 for compute-heavy code)

**Potential Issues:**
- Poor SIMD utilization
- Instruction pipeline stalls
- Branch mispredictions

**Optimization Strategies:**
1. **Improve SIMD Utilization**: Ensure data alignment and loop unrolling
2. **Branchless Code**: Replace conditional branches with SIMD masking
3. **Reduce Data Dependencies**: Restructure algorithms to reduce instruction dependencies
4. **Compiler Optimizations**: Ensure -O3 and appropriate target architecture flags

### If Cache-Thrashing (LLC Miss Rate >50%)

**Potential Issues:**
- Working set too large for cache
- Poor spatial locality
- Cache line pollution

**Optimization Strategies:**
1. **Dataset Partitioning**: Divide working set into cache-sized chunks
2. **Data Compression**: Use quantization to reduce working set size
3. **Cache-Oriented Algorithms**: Use algorithms with better cache locality
4. **Blocking/Tiling**: Implement blocking to improve cache reuse

## Example Benchmark Output Interpretation

### Good Performance Example

```
=== Memory Bandwidth Saturation Test ===
Effective Bandwidth: 17.2 GB/s
Peak Bandwidth: 20.0 GB/s
Saturation: 86.0%
Operational Intensity (FP32): 0.5 FLOPs/Byte

=== CPU Pipeline Efficiency Test ===
Estimated IPC: 2.3
L1 Miss Rate: 8.0%
L2 Miss Rate: 15.0%
LLC Miss Rate: 4.0%
Branch Miss Rate: 1.2%

=== Hardware Saturation Matrix ===
Flat / Quantized Scan      Memory-Bandwidth Bound   EXCELLENT: 86% bandwidth saturation
SIMD Re-Ranking (FP32)     Compute / L1 Bound        GOOD: IPC 2.3
Cluster Centroid Routing    L2 Cache Bound            EXCELLENT: <10% LLC miss rate
```

**Interpretation:**
- Memory bandwidth is well-saturated (86%), indicating efficient memory access
- IPC of 2.3 shows good SIMD utilization
- Low LLC miss rate (4%) indicates working set fits in cache
- Overall, the system is hardware-optimal for memory-bound operations

### Poor Performance Example

```
=== Memory Bandwidth Saturation Test ===
Effective Bandwidth: 5.2 GB/s
Peak Bandwidth: 20.0 GB/s
Saturation: 26.0%
Operational Intensity (FP32): 0.5 FLOPs/Byte

=== CPU Pipeline Efficiency Test ===
Estimated IPC: 0.6
L1 Miss Rate: 25.0%
L2 Miss Rate: 45.0%
LLC Miss Rate: 65.0%
Branch Miss Rate: 8.0%

=== Hardware Saturation Matrix ===
Flat / Quantized Scan      Memory-Bandwidth Bound   POOR: Only 26% bandwidth saturation
SIMD Re-Ranking (FP32)     Compute / L1 Bound        POOR: IPC 0.6 (memory-bound)
Cluster Centroid Routing    L2 Cache Bound            POOR: 65% LLC miss rate (working set too large)
```

**Interpretation:**
- Very low bandwidth saturation (26%) indicates inefficient memory access
- Low IPC (0.6) shows CPU is stalled waiting for memory
- High LLC miss rate (65%) indicates cache thrashing
- Overall, significant software overhead preventing hardware utilization

## Troubleshooting

### Issue: perf command not found
**Solution**: Install linux-tools:
```bash
# Ubuntu/Debian
sudo apt-get install linux-tools-generic

# Fedora
sudo dnf install perf

# RHEL/CentOS
sudo yum install perf
```

### Issue: Cannot access perf counters
**Solution**: Check perf permissions:
```bash
# Check if perf is accessible
perf stat ls

# If permission denied, try with sudo
sudo perf stat ls
```

### Issue: Benchmark crashes or hangs
**Solution**: 
1. Check available memory: `free -h`
2. Reduce dataset size: `--vectors 5000`
3. Check disk space: `df -h`

### Issue: Inconsistent results between runs
**Solution:**
1. Ensure system is idle (no other heavy processes)
2. Disable CPU frequency scaling: `sudo cpupower frequency-set -g performance`
4. Run multiple iterations and average results

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Roofline Analysis

on: [push, pull_request]

jobs:
  roofline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake g++ linux-tools-generic
      - name: Build benchmark
        run: |
          mkdir build && cd build
          cmake .. -DCMAKE_BUILD_TYPE=Release -DPOMAI_BUILD_BENCH=ON
          make roofline_analysis_bench -j$(nproc)
      - name: Run roofline analysis
        run: |
          cd build
          ./benchmarks/roofline_analysis_bench --vectors 10000 --dimension 128
      - name: Run with perf
        run: |
          cd build
          perf stat -e cycles,instructions,branches,branch-misses ./benchmarks/roofline_analysis_bench --vectors 10000
```

## Advanced Analysis

### Custom Dataset Sizes for Different Cache Levels

```bash
# L1 cache test (fits in L1)
./benchmarks/roofline_analysis_bench --vectors 1000 --dimension 32

# L2 cache test (fits in L2)
./benchmarks/roofline_analysis_bench --vectors 5000 --dimension 64

# LLC cache test (fits in LLC)
./benchmarks/roofline_analysis_bench --vectors 25000 --dimension 128

# Beyond cache (memory-bound)
./benchmarks/roofline_analysis_bench --vectors 100000 --dimension 256
```

### Quantization Impact Analysis

```bash
# Compare FP32 vs SQ8 vs INT8
./benchmarks/roofline_analysis_bench --no-quantization  # FP32 only
./benchmarks/roofline_analysis_bench                    # SQ8 (default)
# Custom INT8 benchmark would need separate implementation
```

### HNSW vs Flat Scan Comparison

```bash
# Flat scan analysis
./benchmarks/roofline_analysis_bench --vectors 25000

# HNSW analysis
./benchmarks/roofline_analysis_bench --vectors 25000 --hnsw
```

## Conclusion

This roofline analysis benchmark provides definitive evidence of hardware utilization and helps identify whether performance limitations are due to software overhead or hardware constraints. For embedded vector databases, achieving 70-90% memory bandwidth saturation is considered excellent, while IPC values above 2.0 indicate good SIMD utilization.

Use this benchmark to:
1. Validate hardware optimizations
2. Identify performance bottlenecks
3. Guide optimization efforts
4. Compare different algorithmic approaches
5. Validate performance regressions