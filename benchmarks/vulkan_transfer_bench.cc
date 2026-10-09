// ============================================================================
// PomaiDB Vulkan GPU Memory Bridge & Transfer Benchmark
//
// Evaluates GPU memory operations for vector ingestion and AI model hand-off:
// 1. Host-Visible Copy-Mapped transfer bandwidth (host memory -> GPU-mapped)
// 2. Device-Local VRAM Upload bandwidth via bounded staging ring pool (PCIe DMA)
// 3. Tests across both Discrete GPU (NVIDIA) and Integrated GPU (Intel / UMA)
// ============================================================================

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "vulkan_device_context.h"
#include "vulkan_memory_bridge.h"
#include "vulkan_staging_pool.h"

namespace {

bool SkipVulkanBench() {
    const char* s = std::getenv("POMAI_SKIP_VULKAN_TESTS");
    return s != nullptr && s[0] != '\0' && std::strcmp(s, "0") != 0;
}

std::vector<std::byte> GeneratePayload(std::size_t n) {
    std::vector<std::byte> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = static_cast<std::byte>(static_cast<unsigned>(i % 251));
    }
    return v;
}

struct TransferResult {
    std::size_t bytes;
    std::string size_label;
    double copy_mapped_ms;
    double copy_mapped_gb_s;
    double upload_device_ms;
    double upload_device_gb_s;
};

void RunDeviceBenchmark(bool prefer_unified, const std::string& profile_name) {
    pomai::compute::vulkan::BridgeOptions bopt;
    bopt.prefer_unified_memory = prefer_unified;
    bopt.staging_pool_mb = 64;
    bopt.zero_copy_min_bytes = 4096;

    pomai::compute::vulkan::VulkanComputeContext ctx;
    auto st = pomai::compute::vulkan::VulkanComputeContext::Create(bopt, &ctx);
    if (!st.ok()) {
        std::cerr << "[-] Skipping " << profile_name << ": " << st.message() << "\n";
        return;
    }

    const auto props = ctx.physical_device().getProperties();
    std::cout << "\n===============================================================================\n";
    std::cout << " GPU Benchmark Target: " << props.deviceName.data() << " (" << profile_name << ")\n";
    std::cout << " Queue Family: " << ctx.queue_family_index() << " | Staging Pool: " << bopt.staging_pool_mb << " MiB\n";
    std::cout << "===============================================================================\n";

    pomai::compute::vulkan::VulkanStagingPool pool;
    st = pomai::compute::vulkan::VulkanStagingPool::Create(&ctx, static_cast<uint64_t>(bopt.staging_pool_mb) * 1024ull * 1024ull, &pool);
    if (!st.ok()) {
        std::cerr << "[-] Staging pool creation failed: " << st.message() << "\n";
        return;
    }

    struct Workload {
        std::size_t bytes;
        std::string label;
        int iters;
    };

    std::vector<Workload> workloads = {
        {64 * 1024, "64 KiB (~128 vecs)", 50},
        {256 * 1024, "256 KiB (~512 vecs)", 50},
        {1 * 1024 * 1024, "1 MiB (~2K vecs)", 30},
        {4 * 1024 * 1024, "4 MiB (~8K vecs)", 20},
        {16 * 1024 * 1024, "16 MiB (~32K vecs)", 10}
    };

    std::vector<TransferResult> results;

    std::cout << std::left << std::setw(22) << "Payload Size"
              << std::setw(18) << "Mapped Latency"
              << std::setw(18) << "Mapped Bandwidth"
              << std::setw(18) << "VRAM Upload Lat"
              << std::setw(18) << "VRAM Upload BW"
              << "\n";
    std::cout << "---------------------------------------------------------------------------------------------\n";

    for (const auto& w : workloads) {
        const auto payload = GeneratePayload(w.bytes);
        double total_mapped_ms = 0.0;
        double total_upload_ms = 0.0;

        for (int i = 0; i < w.iters; ++i) {
            // 1. Copy Mapped (host-visible GPU memory)
            pomai::compute::vulkan::HostBuffer hb;
            auto t0 = std::chrono::steady_clock::now();
            st = pomai::compute::vulkan::VulkanMemoryBridge::CreateBufferCopyMapped(&ctx, payload, &hb);
            auto t1 = std::chrono::steady_clock::now();
            if (!st.ok()) {
                std::cerr << "Copy mapped failed: " << st.message() << "\n";
                break;
            }
            total_mapped_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

            // 2. Upload to Device-Local VRAM via Staging Pool
            pomai::compute::vulkan::HostBuffer dev;
            auto t2 = std::chrono::steady_clock::now();
            st = pomai::compute::vulkan::VulkanMemoryBridge::UploadToDeviceBuffer(&ctx, &pool, payload, &dev);
            auto t3 = std::chrono::steady_clock::now();
            if (!st.ok()) {
                std::cerr << "Upload to device failed: " << st.message() << "\n";
                break;
            }
            total_upload_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
        }

        double avg_mapped_ms = total_mapped_ms / w.iters;
        double avg_upload_ms = total_upload_ms / w.iters;
        double bytes_gb = static_cast<double>(w.bytes) / (1024.0 * 1024.0 * 1024.0);
        double mapped_bw = bytes_gb / (avg_mapped_ms / 1000.0);
        double upload_bw = bytes_gb / (avg_upload_ms / 1000.0);

        results.push_back({w.bytes, w.label, avg_mapped_ms, mapped_bw, avg_upload_ms, upload_bw});

        std::cout << std::left << std::setw(22) << w.label
                  << std::fixed << std::setprecision(3)
                  << std::setw(18) << (std::to_string(avg_mapped_ms).substr(0, 6) + " ms")
                  << std::setw(18) << (std::to_string(mapped_bw).substr(0, 5) + " GB/s")
                  << std::setw(18) << (std::to_string(avg_upload_ms).substr(0, 6) + " ms")
                  << std::setw(18) << (std::to_string(upload_bw).substr(0, 5) + " GB/s")
                  << "\n";
    }
}

}  // namespace

int main() {
    if (SkipVulkanBench()) {
        std::fprintf(stderr, "Vulkan transfer bench skipped (POMAI_SKIP_VULKAN_TESTS)\n");
        return 0;
    }

    std::cout << "===============================================================================\n";
    std::cout << "           POMAIDB ENTERPRISE VULKAN GPU TRANSFER BENCHMARK                   \n";
    std::cout << "===============================================================================\n";

    // 1. Run Discrete GPU Benchmark (prefer_unified_memory = false) -> NVIDIA GeForce GTX 1650 SUPER
    RunDeviceBenchmark(false, "Discrete GPU / Dedicated VRAM");

    // 2. Run Integrated GPU Benchmark (prefer_unified_memory = true) -> Intel UHD Graphics 630 / UMA
    RunDeviceBenchmark(true, "Integrated GPU / Unified Memory");

    std::cout << "\n===============================================================================\n";
    std::cout << "              GPU BENCHMARK COMPLETED SUCCESSFULLY                             \n";
    std::cout << "===============================================================================\n";
    return 0;
}
