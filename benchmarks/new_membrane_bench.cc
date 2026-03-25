#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "pomai/pomai.h"

int main() {
    using namespace pomai;
    DBOptions opt;
    opt.path = "/tmp/pomai_new_membrane_bench";
    opt.dim = 4;
    opt.shard_count = 1;
    opt.max_blob_bytes_mb = 8;

    std::unique_ptr<DB> db;
    auto st = DB::Open(opt, &db);
    if (!st.ok()) {
        std::cerr << "open failed: " << st.message() << "\n";
        return 1;
    }
    MembraneSpec ts; ts.name = "ts"; ts.kind = MembraneKind::kTimeSeries; ts.shard_count = 1;
    MembraneSpec kv; kv.name = "kv"; kv.kind = MembraneKind::kKeyValue; kv.shard_count = 1;
    MembraneSpec sk; sk.name = "sk"; sk.kind = MembraneKind::kSketch; sk.shard_count = 1;
    MembraneSpec bl; bl.name = "bl"; bl.kind = MembraneKind::kBlob; bl.shard_count = 1;
    (void)db->CreateMembrane(ts); (void)db->OpenMembrane("ts");
    (void)db->CreateMembrane(kv); (void)db->OpenMembrane("kv");
    (void)db->CreateMembrane(sk); (void)db->OpenMembrane("sk");
    (void)db->CreateMembrane(bl); (void)db->OpenMembrane("bl");

    const int N = 20000;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) (void)db->TsPut("ts", 1, 1000 + i, i * 0.1);
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) (void)db->KvPut("kv", "k" + std::to_string(i), "v");
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) (void)db->SketchAdd("sk", "id" + std::to_string(i % 512), 1);
    auto t3 = std::chrono::steady_clock::now();
    std::vector<uint8_t> payload(1024, 7);
    for (int i = 0; i < 2000; ++i) (void)db->BlobPut("bl", static_cast<uint64_t>(i), payload);
    auto t4 = std::chrono::steady_clock::now();

    auto ms_ts = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto ms_kv = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto ms_sk = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    auto ms_bl = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

    std::cout << "timeseries_put_ms " << ms_ts << "\n";
    std::cout << "kv_put_ms " << ms_kv << "\n";
    std::cout << "sketch_add_ms " << ms_sk << "\n";
    std::cout << "blob_put_ms " << ms_bl << "\n";
    (void)db->Close();
    return 0;
}

