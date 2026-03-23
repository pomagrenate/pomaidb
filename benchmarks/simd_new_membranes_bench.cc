#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "pomai/pomai.h"

int main() {
    using namespace pomai;
    DBOptions opt;
    opt.path = "/tmp/pomai_simd_membranes_bench";
    opt.dim = 4;
    opt.shard_count = 1;
    std::unique_ptr<DB> db;
    auto st = DB::Open(opt, &db);
    if (!st.ok()) return 1;

    MembraneSpec sp; sp.name = "sp"; sp.kind = MembraneKind::kSpatial; sp.shard_count = 1;
    MembraneSpec sps; sps.name = "sps"; sps.kind = MembraneKind::kSparse; sps.shard_count = 1;
    MembraneSpec bs; bs.name = "bs"; bs.kind = MembraneKind::kBitset; bs.shard_count = 1;
    (void)db->CreateMembrane(sp); (void)db->OpenMembrane("sp");
    (void)db->CreateMembrane(sps); (void)db->OpenMembrane("sps");
    (void)db->CreateMembrane(bs); (void)db->OpenMembrane("bs");

    const int N = 20000;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) (void)db->SpatialPut("sp", static_cast<uint64_t>(i), 10.0 + i * 1e-5, 106.0 + i * 1e-5);
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        SparseEntry e{{1u, 8u, static_cast<uint32_t>((i % 1024) + 16)}, {1.0f, 2.0f, 3.0f}};
        (void)db->SparsePut("sps", static_cast<uint64_t>(i), e);
    }
    auto t2 = std::chrono::steady_clock::now();
    std::vector<uint8_t> bits(128, 0xAA);
    for (int i = 0; i < 5000; ++i) (void)db->BitsetPut("bs", static_cast<uint64_t>(i), bits);
    auto t3 = std::chrono::steady_clock::now();

    std::cout << "spatial_put_ms " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "\n";
    std::cout << "sparse_put_ms " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\n";
    std::cout << "bitset_put_ms " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "\n";
    (void)db->Close();
    return 0;
}

