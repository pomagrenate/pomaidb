// tests/adversarial/crash_consistency_test.cc
// Phase 3: Crash Consistency & Power Loss Harness
// Simulates crashes, power loss, torn writes, uncommitted manifests, and orphaned files.

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "tests/adversarial/golden_oracle.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include "pomegranate_engine.h"
#include "locule.h"
#include "aril.h"
#include "fruit_map.h"
#include "options.h"
#include "pomai_format.h"
#include "search.h"

namespace {

using namespace pomai;
using namespace pomai::core;
using namespace pomai::adversarial;

// 1. Crash during WAL append: Torn WAL record at EOF
POMAI_TEST(Crash_TornWalRecord_RecoversValidPrefix) {
    const std::string db_dir = test::TempDir("pomai-crash-torn-wal");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 16;
    const uint32_t N = 50;

    {
        DBOptions opt;
        opt.path = db_dir;
        opt.dim = dim;
        opt.fsync = FsyncPolicy::kAlways;
        PomegranateEngine engine(opt, MetricType::kL2);
        POMAI_EXPECT_OK(engine.Open());

        std::mt19937 rng(42);
        std::normal_distribution<float> d_norm(0.0f, 1.0f);

        for (uint32_t i = 1; i <= N; ++i) {
            std::vector<float> vec(dim);
            for (uint32_t d = 0; d < dim; ++d) vec[d] = d_norm(rng);
            POMAI_EXPECT_OK(engine.Put(i, vec));
        }
        POMAI_EXPECT_OK(engine.Close());
    }

    // Find the WAL segment file
    std::string wal_file;
    for (const auto& entry : std::filesystem::directory_iterator(db_dir)) {
        if (entry.path().extension() == ".wal" || entry.path().filename().string().rfind("wal_", 0) == 0) {
            wal_file = entry.path().string();
            break;
        }
    }
    POMAI_EXPECT_TRUE(!wal_file.empty());

    // Truncate the last record by 15 bytes to simulate a torn write / power cut mid-append
    uint64_t wal_size = std::filesystem::file_size(wal_file);
    POMAI_EXPECT_TRUE(wal_size > 50);
    uint64_t torn_size = wal_size - 15;
    std::filesystem::resize_file(wal_file, torn_size);

    // Reopen database: Must succeed and recover all records prior to the torn write
    {
        DBOptions opt;
        opt.path = db_dir;
        opt.dim = dim;
        opt.fsync = FsyncPolicy::kNever;
        PomegranateEngine engine(opt, MetricType::kL2);
        POMAI_EXPECT_OK(engine.Open());

        // First N-2 records are guaranteed committed and intact
        for (uint32_t i = 1; i <= N - 2; ++i) {
            std::vector<float> out;
            auto st = engine.Get(i, &out);
            POMAI_EXPECT_OK(st);
            POMAI_EXPECT_EQ(out.size(), dim);
        }

        // Querying must succeed without crash
        std::vector<float> q(dim, 0.0f);
        SearchResult res;
        POMAI_EXPECT_OK(engine.Search(q, 10, &res));
        POMAI_EXPECT_TRUE(!res.hits.empty());

        POMAI_EXPECT_OK(engine.Close());
    }
}

// 2. Uncommitted manifest temp file during crash
POMAI_TEST(Crash_UncommittedFruitManifest_IgnoresTmpFile) {
    const std::string db_dir = test::TempDir("pomai-crash-tmp-manifest");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 16;

    {
        DBOptions opt;
        opt.path = db_dir;
        opt.dim = dim;
        opt.fsync = FsyncPolicy::kNever;
        PomegranateEngine engine(opt, MetricType::kL2);
        POMAI_EXPECT_OK(engine.Open());

        std::vector<float> vec(dim, 1.0f);
        POMAI_EXPECT_OK(engine.Put(1, vec));
        POMAI_EXPECT_OK(engine.Compact());
        POMAI_EXPECT_OK(engine.Close());
    }

    // Plant an orphaned temporary manifest file simulating crash during write
    std::string tmp_manifest = db_dir + "/fruit.manifest.tmp";
    {
        std::ofstream f(tmp_manifest);
        f << "POMAI_FRUIT_MANIFEST_v1\ngarbage_uncommitted_state_crash_in_progress\n";
    }

    // Reopening database must ignore .tmp and use valid committed manifest
    {
        DBOptions opt;
        opt.path = db_dir;
        opt.dim = dim;
        opt.fsync = FsyncPolicy::kNever;
        PomegranateEngine engine(opt, MetricType::kL2);
        POMAI_EXPECT_OK(engine.Open());

        std::vector<float> out;
        POMAI_EXPECT_OK(engine.Get(1, &out));
        POMAI_EXPECT_EQ(out.size(), dim);

        POMAI_EXPECT_OK(engine.Close());
    }
}

// 3. Corrupted manifest CRC rejection
POMAI_TEST(Crash_ManifestChecksumMismatch_RejectsCorruptManifest) {
    const std::string db_dir = test::TempDir("pomai-crash-corrupt-manifest");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 16;

    {
        DBOptions opt;
        opt.path = db_dir;
        opt.dim = dim;
        opt.fsync = FsyncPolicy::kNever;
        PomegranateEngine engine(opt, MetricType::kL2);
        POMAI_EXPECT_OK(engine.Open());

        std::vector<float> vec(dim, 2.0f);
        POMAI_EXPECT_OK(engine.Put(1, vec));
        POMAI_EXPECT_OK(engine.Compact());
        POMAI_EXPECT_OK(engine.Close());
    }

    // Corrupt one character in fruit.manifest
    std::string manifest_path = db_dir + "/fruit.manifest";
    POMAI_EXPECT_TRUE(std::filesystem::exists(manifest_path));

    std::ifstream in(manifest_path);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    // Flip a character in "generation="
    size_t gen_pos = content.find("generation=");
    POMAI_EXPECT_TRUE(gen_pos != std::string::npos);
    content[gen_pos + 11] = (content[gen_pos + 11] == '1') ? '2' : '9';

    std::ofstream out(manifest_path, std::ios::trunc);
    out << content;
    out.close();

    // Reopen FruitMap directly — MUST return corruption due to CRC mismatch!
    manifest::FruitMap fruit_map(db_dir, dim, MetricType::kL2);
    Status st = fruit_map.Open();
    POMAI_EXPECT_TRUE(!st.ok());
    POMAI_EXPECT_TRUE(st.code() == ErrorCode::kCorruption);
}

// 4. Orphaned uncommitted Locule container
POMAI_TEST(Crash_OrphanedLocule_IgnoredCleanly) {
    const std::string db_dir = test::TempDir("pomai-crash-orphaned-locule");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 16;

    {
        DBOptions opt;
        opt.path = db_dir;
        opt.dim = dim;
        opt.fsync = FsyncPolicy::kNever;
        PomegranateEngine engine(opt, MetricType::kL2);
        POMAI_EXPECT_OK(engine.Open());

        std::vector<float> vec(dim, 3.0f);
        POMAI_EXPECT_OK(engine.Put(1, vec));
        POMAI_EXPECT_OK(engine.Compact());
        POMAI_EXPECT_OK(engine.Close());
    }

    // Drop an uncommitted/orphaned .pom file in the directory
    std::string orphaned_pom = db_dir + "/locule_0000000000009999.pom";
    {
        std::ofstream f(orphaned_pom, std::ios::binary);
        f << "random junk pretending to be interrupted locule write";
    }

    // Reopen engine: must ignore unreferenced locule and load committed state normally
    {
        DBOptions opt;
        opt.path = db_dir;
        opt.dim = dim;
        opt.fsync = FsyncPolicy::kNever;
        PomegranateEngine engine(opt, MetricType::kL2);
        POMAI_EXPECT_OK(engine.Open());

        std::vector<float> out;
        POMAI_EXPECT_OK(engine.Get(1, &out));
        POMAI_EXPECT_EQ(out.size(), dim);

        POMAI_EXPECT_OK(engine.Close());
    }
}

} // namespace
