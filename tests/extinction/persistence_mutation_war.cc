// tests/extinction/persistence_mutation_war.cc
// Extinction Event: 10,000+ Real-File Persistent Container Mutations
//
// Generates a real database with HNSW, compacts it to real .pom containers,
// then applies 10,000+ targeted and random bit/byte mutations across all structures.

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
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

static std::string FindLoculeFile(const std::string& db_dir) {
    for (const auto& entry : std::filesystem::directory_iterator(db_dir)) {
        if (entry.path().extension() == ".pom" && entry.path().filename().string().rfind("locule_", 0) == 0) {
            return entry.path().string();
        }
    }
    return "";
}

POMAI_TEST(Extinction_10000_PersistenceMutations) {
    const std::string base_dir = test::TempDir("pomai-mutation-war-real");
    std::filesystem::remove_all(base_dir);
    const uint32_t dim = 32;
    const uint32_t N = 300;

    // Step 1: Create a REAL, valid database with HNSW index and compact to .pom
    {
        DBOptions opt;
        opt.path = base_dir;
        opt.dim = dim;
        opt.fsync = FsyncPolicy::kNever;
        opt.index_params.type = IndexType::kHnsw;
        opt.index_params.hnsw_m = 16;
        opt.index_params.hnsw_ef_construction = 100;
        opt.index_params.hnsw_ef_search = 32;

        PomegranateEngine engine(opt, MetricType::kL2);
        POMAI_EXPECT_OK(engine.Open());

        std::mt19937 rng(42);
        std::normal_distribution<float> d_norm(0.0f, 1.0f);

        for (uint32_t i = 1; i <= N; ++i) {
            std::vector<float> vec(dim);
            for (uint32_t d = 0; d < dim; ++d) vec[d] = d_norm(rng);
            Metadata meta;
            meta.device_id = (i % 2 == 0) ? "dev_a" : "dev_b";
            POMAI_EXPECT_OK(engine.Put(i, vec, meta));
        }

        POMAI_EXPECT_OK(engine.Compact());
        POMAI_EXPECT_OK(engine.Close());
    }

    std::string locule_file = FindLoculeFile(base_dir);
    POMAI_EXPECT_TRUE(!locule_file.empty());

    // Read genuine original bytes
    std::ifstream orig_file(locule_file, std::ios::binary);
    std::vector<uint8_t> valid_bytes((std::istreambuf_iterator<char>(orig_file)),
                                      std::istreambuf_iterator<char>());
    orig_file.close();
    size_t file_len = valid_bytes.size();
    POMAI_EXPECT_TRUE(file_len > 1000);

    const uint32_t total_mutations = 10000;
    std::mt19937_64 mut_rng(0xC0FFEE);
    std::uniform_int_distribution<size_t> off_dist(0, file_len - 1);
    std::uniform_int_distribution<int> val_dist(0, 255);

    uint64_t rejected_corruptions = 0;
    uint64_t accepted_valid = 0;

    std::vector<uint8_t> test_buf = valid_bytes;

    for (uint32_t m = 1; m <= total_mutations; ++m) {
        test_buf = valid_bytes;

        // Choose mutation strategy:
        // 0: single bit flip
        // 1: single byte overwrite
        // 2: multi-byte contiguous stomp
        // 3: extreme integer injection in header
        int strategy = m % 4;

        if (strategy == 0) {
            size_t off = off_dist(mut_rng);
            uint8_t bit = static_cast<uint8_t>(1u << (mut_rng() % 8));
            test_buf[off] ^= bit;
        } else if (strategy == 1) {
            size_t off = off_dist(mut_rng);
            test_buf[off] = static_cast<uint8_t>(val_dist(mut_rng));
        } else if (strategy == 2) {
            size_t off = off_dist(mut_rng);
            size_t span = (mut_rng() % 16) + 1;
            for (size_t k = 0; k < span && (off + k) < file_len; ++k) {
                test_buf[off + k] = static_cast<uint8_t>(val_dist(mut_rng));
            }
        } else {
            // Target specific header offsets: magic, offsets, counts
            size_t target_off = (mut_rng() % 5) * 8;
            if (target_off + 8 <= file_len) {
                uint64_t extreme_val = (mut_rng() % 2 == 0) ? 0xFFFFFFFFFFFFFFFFull : 0x7FFFFFFFFFFFFFFFull;
                std::memcpy(test_buf.data() + target_off, &extreme_val, 8);
            }
        }

        // Test in-memory locule opening
        auto mem_copy = std::make_unique<uint8_t[]>(test_buf.size());
        std::memcpy(mem_copy.get(), test_buf.data(), test_buf.size());

        alloc::SharedPtr<storage::Locule> loc;
        Status st = storage::Locule::OpenFromMemory(mem_copy.get(), test_buf.size(), &loc);

        if (!st.ok() || loc == nullptr) {
            rejected_corruptions++;
        } else {
            accepted_valid++;
            // If it opened, querying must NEVER crash
            std::vector<float> q(dim, 0.0f);
            for (const auto& aril : loc->arils()) {
                if (aril && aril->vector_count() > 0) {
                    std::vector<float> vec;
                    (void)aril->GetVector(0, &vec);
                }
            }
        }

        if (m % 2500 == 0) {
            std::cout << "[MUTATION WAR] Completed " << m << " mutations. Rejected: "
                      << rejected_corruptions << " Accepted: " << accepted_valid << std::endl;
        }
    }

    POMAI_EXPECT_TRUE(rejected_corruptions > 0);
}

} // namespace
