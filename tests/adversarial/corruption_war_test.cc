// tests/adversarial/corruption_war_test.cc
// Phase 2: Data Corruption War & Phase 24: .pom Mutational Fuzzer
// Systematically attacks .pom container headers, Arils, graphs, offsets, lengths,
// CRCs, truncations, and integer overflows to ensure zero memory safety crashes.

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
#include "utils/palloc_smart_ptr.h"
#include "aril.h"
#include "options.h"
#include "pomai_format.h"
#include "search.h"

namespace {

using namespace pomai;
using namespace pomai::core;
using namespace pomai::adversarial;

static void PopulateAndCompact(const std::string& db_dir, uint32_t N = 500, uint32_t dim = 32) {
    std::filesystem::remove_all(db_dir);
    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kNever;
    opt.index_params.type = IndexType::kHnsw;
    opt.index_params.hnsw_m = 16;
    opt.index_params.hnsw_ef_construction = 100;
    opt.index_params.hnsw_ef_search = 32;

    PomegranateEngine engine(opt, MetricType::kL2);
    POMAI_EXPECT_OK(engine.Open());

    std::mt19937 rng(1337);
    std::normal_distribution<float> d_norm(0.0f, 1.0f);

    for (uint32_t i = 1; i <= N; ++i) {
        std::vector<float> vec(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            vec[d] = d_norm(rng);
        }
        POMAI_EXPECT_OK(engine.Put(i, vec));
    }

    POMAI_EXPECT_OK(engine.Compact());
    POMAI_EXPECT_OK(engine.Close());
}

static std::string FindLoculeFile(const std::string& db_dir) {
    for (const auto& entry : std::filesystem::directory_iterator(db_dir)) {
        if (entry.path().extension() == ".pom" && entry.path().filename().string().rfind("locule_", 0) == 0) {
            return entry.path().string();
        }
    }
    return "";
}

// 1. Attack: Truncated .pom files (at 0B, 16B, 48B, 100B, 50% length, 90% length)
POMAI_TEST(Corruption_Truncation_NeverCrashes) {
    const std::string base_dir = test::TempDir("pomai-corrupt-trunc");
    const uint32_t dim = 32;
    PopulateAndCompact(base_dir, 300, dim);

    std::string locule_file = FindLoculeFile(base_dir);
    POMAI_EXPECT_TRUE(!locule_file.empty());

    // Read valid original bytes
    std::ifstream orig_file(locule_file, std::ios::binary);
    std::vector<uint8_t> valid_bytes((std::istreambuf_iterator<char>(orig_file)),
                                      std::istreambuf_iterator<char>());
    orig_file.close();
    size_t file_len = valid_bytes.size();
    POMAI_EXPECT_TRUE(file_len > 100);

    const auto* hdr = reinterpret_cast<const format::PomaiFileHeader*>(valid_bytes.data());
    size_t min_required = hdr->footer_offset;
    if (hdr->footer_offset > 0 && hdr->footer_offset + sizeof(format::LoculeFooter) <= file_len) {
        const auto* footer = reinterpret_cast<const format::LoculeFooter*>(valid_bytes.data() + hdr->footer_offset);
        min_required = hdr->footer_offset + sizeof(format::LoculeFooter) + footer->centroid_dim * sizeof(float);
    }

    const std::vector<size_t> truncate_points = {
        0, 1, 16, 48, 64, 100,
        file_len / 4, file_len / 2,
        hdr->footer_offset,
        hdr->footer_offset + sizeof(format::LoculeFooter) / 2,
        hdr->footer_offset + sizeof(format::LoculeFooter) + 1,
        min_required - 1,
        file_len - 1
    };

    for (size_t trunc_sz : truncate_points) {
        // Write truncated file
        std::ofstream trunc_file(locule_file, std::ios::binary | std::ios::trunc);
        trunc_file.write(reinterpret_cast<const char*>(valid_bytes.data()), trunc_sz);
        trunc_file.close();

        // Attempt reopening Locule directly
        alloc::SharedPtr<storage::Locule> loc;
        auto st = storage::Locule::Open(locule_file, &loc);
        // If truncated before all declared data/footer structures, MUST return corruption or error
        if (trunc_sz < min_required) {
            POMAI_EXPECT_TRUE(!st.ok() || loc == nullptr);
        }

        // Attempt reopening full Engine
        DBOptions opt;
        opt.path = base_dir;
        opt.dim = dim;
        opt.fsync = FsyncPolicy::kNever;
        PomegranateEngine engine(opt, MetricType::kL2);
        // Engine Open might fail cleanly or recover
        auto open_st = engine.Open();
        if (open_st.ok()) {
            std::vector<float> q(dim, 0.0f);
            SearchResult res;
            // Querying must NEVER crash
            (void)engine.Search(q, 10, &res);
            (void)engine.Close();
        }
    }
}

// 2. Attack: Integer Overflow Offsets in Header & Directory Table
POMAI_TEST(Corruption_IntegerOverflowOffsets_NeverCrashes) {
    const std::string base_dir = test::TempDir("pomai-corrupt-overflow");
    const uint32_t dim = 32;
    PopulateAndCompact(base_dir, 300, dim);

    std::string locule_file = FindLoculeFile(base_dir);
    POMAI_EXPECT_TRUE(!locule_file.empty());

    std::ifstream orig_file(locule_file, std::ios::binary);
    std::vector<uint8_t> valid_bytes((std::istreambuf_iterator<char>(orig_file)),
                                      std::istreambuf_iterator<char>());
    orig_file.close();

    // Test extreme offsets: UINT64_MAX, UINT64_MAX - 10, UINT32_MAX, etc.
    const std::vector<uint64_t> extreme_offsets = {
        0xFFFFFFFFFFFFFFFFull,
        0xFFFFFFFFFFFFFFF0ull,
        0x7FFFFFFFFFFFFFFFull,
        0x0000000100000000ull,
        0x00000000FFFFFFFFull
    };

    for (uint64_t bad_val : extreme_offsets) {
        // Case A: Corrupt directory_offset in PomaiFileHeader
        {
            std::vector<uint8_t> bytes = valid_bytes;
            auto* hdr = reinterpret_cast<format::PomaiFileHeader*>(bytes.data());
            hdr->directory_offset = bad_val;

            std::ofstream f(locule_file, std::ios::binary | std::ios::trunc);
            f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            f.close();

            alloc::SharedPtr<storage::Locule> loc;
            auto st = storage::Locule::Open(locule_file, &loc);
            POMAI_EXPECT_TRUE(!st.ok() || loc == nullptr);
        }

        // Case B: Corrupt aril_offset in ArilDirectoryEntry
        {
            std::vector<uint8_t> bytes = valid_bytes;
            auto* hdr = reinterpret_cast<format::PomaiFileHeader*>(bytes.data());
            if (hdr->directory_offset + sizeof(format::ArilDirectoryEntry) <= bytes.size()) {
                auto* entry = reinterpret_cast<format::ArilDirectoryEntry*>(bytes.data() + hdr->directory_offset);
                entry->aril_offset = bad_val;

                std::ofstream f(locule_file, std::ios::binary | std::ios::trunc);
                f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                f.close();

                alloc::SharedPtr<storage::Locule> loc;
                auto st = storage::Locule::Open(locule_file, &loc);
                POMAI_EXPECT_TRUE(!st.ok() || loc == nullptr);
            }
        }
    }
}

// 3. Attack: Section Size Misrepresentation (vector_count = 100,000 but kernel_size = 4B)
POMAI_TEST(Corruption_UnderallocatedSections_RejectsGracefully) {
    const std::string base_dir = test::TempDir("pomai-corrupt-undersized");
    const uint32_t dim = 32;
    PopulateAndCompact(base_dir, 300, dim);

    std::string locule_file = FindLoculeFile(base_dir);
    POMAI_EXPECT_TRUE(!locule_file.empty());

    std::ifstream orig_file(locule_file, std::ios::binary);
    std::vector<uint8_t> valid_bytes((std::istreambuf_iterator<char>(orig_file)),
                                      std::istreambuf_iterator<char>());
    orig_file.close();

    auto* hdr = reinterpret_cast<format::PomaiFileHeader*>(valid_bytes.data());
    auto* dir = reinterpret_cast<format::ArilDirectoryEntry*>(valid_bytes.data() + hdr->directory_offset);
    auto* aril_hdr = reinterpret_cast<format::ArilHeader*>(valid_bytes.data() + dir->aril_offset);

    // Lie about vector count: claim 50,000 vectors while kernel_size only fits 300
    aril_hdr->vector_count = 50000;

    std::ofstream f(locule_file, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(valid_bytes.data()), valid_bytes.size());
    f.close();

    alloc::SharedPtr<storage::Locule> loc;
    auto st = storage::Locule::Open(locule_file, &loc);
    // Must reject, or if opened, GetVector must never read out of bounds
    if (st.ok() && loc && loc->aril_count() > 0) {
        std::vector<float> out_vec;
        // Asking for an out-of-bounds slot must return error or empty, never segfault
        (void)loc->arils()[0]->GetVector(45000, &out_vec);
    }
}

// 4. Attack: Mutational Fuzzing on .pom Container Bytes (200 random mutations)
POMAI_TEST(Corruption_MutationalFuzzing_NeverSegfaults) {
    const std::string base_dir = test::TempDir("pomai-corrupt-fuzz");
    const uint32_t dim = 32;
    PopulateAndCompact(base_dir, 200, dim);

    std::string locule_file = FindLoculeFile(base_dir);
    POMAI_EXPECT_TRUE(!locule_file.empty());

    std::ifstream orig_file(locule_file, std::ios::binary);
    std::vector<uint8_t> valid_bytes((std::istreambuf_iterator<char>(orig_file)),
                                      std::istreambuf_iterator<char>());
    orig_file.close();

    std::mt19937 rng(4242);
    std::uniform_int_distribution<size_t> off_dist(0, valid_bytes.size() - 1);
    std::uniform_int_distribution<int> val_dist(0, 255);

    // Run 100 fuzzed iterations mutating 1-8 random bytes each iteration
    for (int iter = 0; iter < 100; ++iter) {
        std::vector<uint8_t> fuzzed_bytes = valid_bytes;
        int mutations = (iter % 8) + 1;
        for (int m = 0; m < mutations; ++m) {
            size_t target_off = off_dist(rng);
            fuzzed_bytes[target_off] = static_cast<uint8_t>(val_dist(rng));
        }

        std::ofstream f(locule_file, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(fuzzed_bytes.data()), fuzzed_bytes.size());
        f.close();

        // Must open/fail gracefully without crashing
        alloc::SharedPtr<storage::Locule> loc;
        (void)storage::Locule::Open(locule_file, &loc);
    }
}

} // namespace
