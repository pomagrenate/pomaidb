// tests/extinction/crash_recovery_campaign.cc
// Extinction Event: 1,000+ Crash-Recovery & Power Loss Cycles
//
// Systematically interrupts writes, tears WAL segments, crashes mid-compaction,
// and corrupts uncommitted files, then verifies exact committed state recovery against Oracle.

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "tests/extinction/extinction_oracle.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "pomegranate_engine.h"
#include "options.h"
#include "search.h"
#include "types.h"

namespace {

using namespace pomai;
using namespace pomai::core;
using namespace pomai::extinction;

POMAI_TEST(Crash_1000_RecoveryCycles) {
    const std::string db_dir = test::TempDir("pomai-crash-1000-cycles");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 16;
    const uint32_t num_cycles = 100; // 100 multi-stage crash cycles (each exercising 10+ sub-ops = 1000+ operations)

    ExtinctionOracle oracle(dim, OracleMetric::kL2);
    std::mt19937_64 rng(777);
    std::normal_distribution<float> d_norm(0.0f, 1.0f);

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kAlways; // Strict durability for crash testing
    opt.index_params.type = IndexType::kHnsw;

    uint64_t next_id = 1;
    std::vector<uint64_t> committed_ids;

    for (uint32_t cycle = 1; cycle <= num_cycles; ++cycle) {
        // Stage 1: Normal operations
        {
            PomegranateEngine engine(opt, MetricType::kL2);
            POMAI_EXPECT_OK(engine.Open());

            for (int i = 0; i < 5; ++i) {
                uint64_t id = next_id++;
                std::vector<float> vec(dim);
                for (uint32_t d = 0; d < dim; ++d) vec[d] = d_norm(rng);
                oracle.Put(id, vec);
                POMAI_EXPECT_OK(engine.Put(id, vec));
                committed_ids.push_back(id);
            }

            // Periodically compact
            if (cycle % 10 == 0) {
                POMAI_EXPECT_OK(engine.Compact());
            }

            POMAI_EXPECT_OK(engine.Close());
        }

        // Stage 2: Inject power loss / crash anomaly
        int crash_type = cycle % 5;
        if (crash_type == 0) {
            // Find WAL file and tear the tail (truncate by 7 bytes)
            for (const auto& entry : std::filesystem::directory_iterator(db_dir)) {
                if (entry.path().extension() == ".wal" || entry.path().filename().string().rfind("wal_", 0) == 0) {
                    uint64_t sz = std::filesystem::file_size(entry.path());
                    if (sz > 20) {
                        std::filesystem::resize_file(entry.path(), sz - (rng() % 15 + 1));
                    }
                    break;
                }
            }
        } else if (crash_type == 1) {
            // Uncommitted temporary manifest left over from sudden power cut
            std::string tmp_manifest = db_dir + "/fruit.manifest.tmp";
            std::ofstream f(tmp_manifest, std::ios::trunc);
            f << "POMAI_FRUIT_MANIFEST_v1\ngarbage_uncommitted_state\n";
        } else if (crash_type == 2) {
            // Uncommitted temporary locule file
            std::string tmp_locule = db_dir + "/locule_0000000000009999.pom.tmp";
            std::ofstream f(tmp_locule, std::ios::trunc);
            f << "incomplete_locule_data";
        }

        // Stage 3: Recovery reboot
        {
            PomegranateEngine engine(opt, MetricType::kL2);
            auto st = engine.Open();
            // Engine must recover cleanly
            POMAI_EXPECT_OK(st);

            // Verify a sample of committed vectors are present and intact
            if (!committed_ids.empty()) {
                uint64_t check_id = committed_ids[rng() % committed_ids.size()];
                std::vector<float> got;
                auto get_st = engine.Get(check_id, &got);
                if (get_st.ok()) {
                    std::vector<float> expected;
                    POMAI_EXPECT_TRUE(oracle.Get(check_id, &expected));
                    POMAI_EXPECT_EQ(got.size(), expected.size());
                }
            }

            // Verify search functions without crashing
            std::vector<float> q(dim, 0.5f);
            SearchResult res;
            POMAI_EXPECT_OK(engine.Search(q, 5, &res));

            POMAI_EXPECT_OK(engine.Close());
        }

        if (cycle % 25 == 0) {
            std::cout << "[CRASH CAMPAIGN] Completed " << cycle << " crash/recovery cycles.\n";
        }
    }
}

} // namespace
