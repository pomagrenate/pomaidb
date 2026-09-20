#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <filesystem>
#include <vector>
#include <fstream>
#include <span>

#include "pomai.h"
#include "options.h"
#include "search.h"
#include "types.h"
#include "segment.h"
#include "manifest.h"
#include "segment_manifest.h"

namespace {

namespace fs = std::filesystem;

POMAI_TEST(DB_SegmentLoading_ReadTest) {
    const std::string root = pomai::test::TempDir("pomai-db-segment-test");
    const std::string membrane = "default";
    const uint32_t dim = 4;
    
    pomai::DBOptions opt;
    opt.path = root;
    opt.dim = dim;

    pomai::MembraneSpec spec;
    spec.name = membrane;
    spec.dim = dim;
    spec.metric = pomai::MetricType::kInnerProduct;

    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 1.0f, 0.0f, 0.0f};

    // 1. Create DB, put data, freeze to segment on disk, and close
    {
        std::unique_ptr<pomai::DB> db;
        POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));
        POMAI_EXPECT_OK(db->CreateMembrane(spec));
        POMAI_EXPECT_OK(db->OpenMembrane(membrane));
        POMAI_EXPECT_OK(db->Put(membrane, 10, vec1));
        POMAI_EXPECT_OK(db->Put(membrane, 20, vec2));
        POMAI_EXPECT_OK(db->Freeze(membrane));
        POMAI_EXPECT_OK(db->Close());
    }
    
    // 2. Reopen DB from disk and verify segments load properly
    {
        std::unique_ptr<pomai::DB> db;
        POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));
        POMAI_EXPECT_OK(db->OpenMembrane(membrane));
        
        // Verify Get
        std::vector<float> out;
        POMAI_EXPECT_OK(db->Get(membrane, 10, &out));
        POMAI_EXPECT_EQ(out.size(), (size_t)dim);
        POMAI_EXPECT_TRUE(std::abs(out[0] - 1.0f) < 0.1f);
        
        out.clear();
        POMAI_EXPECT_OK(db->Get(membrane, 20, &out));
        POMAI_EXPECT_TRUE(std::abs(out[0] - 0.0f) < 0.1f);
        POMAI_EXPECT_TRUE(std::abs(out[1] - 1.0f) < 0.1f);
        
        // Non-existent
        pomai::Status st = db->Get(membrane, 99, &out);
        POMAI_EXPECT_EQ(st.code(), pomai::ErrorCode::kNotFound);
        
        // Verify Search
        std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f}; // Exact match for 10
        pomai::SearchResult res;
        POMAI_EXPECT_OK(db->Search(membrane, query, 5, &res));
        POMAI_EXPECT_TRUE(res.hits.size() >= 1);
        POMAI_EXPECT_EQ(res.hits[0].id, (pomai::VectorId)10);
        POMAI_EXPECT_TRUE(std::abs(res.hits[0].score - 1.0f) < 0.001f);
        
        // Verify Exists
        bool exists = false;
        POMAI_EXPECT_OK(db->Exists(membrane, 20, &exists));
        POMAI_EXPECT_TRUE(exists);
        POMAI_EXPECT_OK(db->Exists(membrane, 99, &exists));
        POMAI_EXPECT_TRUE(!exists);

        POMAI_EXPECT_OK(db->Close());
    }
    fs::remove_all(root);
}

POMAI_TEST(DB_FreezeAndCompact) {
    const std::string root = pomai::test::TempDir("pomai-db-freeze-compact");
    const std::string membrane = "default";
    const uint32_t dim = 4;

    pomai::DBOptions opt;
    opt.path = root;
    opt.dim = dim;

    std::unique_ptr<pomai::DB> db;
    POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

    pomai::MembraneSpec spec;
    spec.name = membrane;
    spec.dim = dim;
    spec.metric = pomai::MetricType::kL2;
    POMAI_EXPECT_OK(db->CreateMembrane(spec));
    POMAI_EXPECT_OK(db->OpenMembrane(membrane));
    
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 1.0f, 0.0f, 0.0f};

    // 1. Put data
    POMAI_EXPECT_OK(db->Put(membrane, 10, vec1));
    POMAI_EXPECT_OK(db->Put(membrane, 20, vec2));
    
    // 2. Freeze (MemTable -> Segment)
    POMAI_EXPECT_OK(db->Freeze(membrane));
    
    // Verify readable
    std::vector<float> out;
    POMAI_EXPECT_OK(db->Get(membrane, 10, &out));
    POMAI_EXPECT_TRUE(std::abs(out[0] - 1.0f) < 0.1f);
    
    // 3. Update (Shadowing)
    std::vector<float> vec1_v2 = {2.0f, 0.0f, 0.0f, 0.0f};
    POMAI_EXPECT_OK(db->Put(membrane, 10, vec1_v2));
    
    // 4. Freeze again (New Segment)
    POMAI_EXPECT_OK(db->Freeze(membrane));
    
    // Verify updated value
    out.clear();
    POMAI_EXPECT_OK(db->Get(membrane, 10, &out));
    POMAI_EXPECT_TRUE(std::abs(out[0] - 2.0f) < 0.1f);
    
    // 5. Delete 20
    POMAI_EXPECT_OK(db->Delete(membrane, 20));
    // Freeze (Tombstone in new segment)
    POMAI_EXPECT_OK(db->Freeze(membrane));
    
    // Verify deleted
    pomai::Status st = db->Get(membrane, 20, &out);
    POMAI_EXPECT_EQ(st.code(), pomai::ErrorCode::kNotFound);
    
    // 6. Compact
    POMAI_EXPECT_OK(db->Compact(membrane));
    
    // Verify data state
    out.clear();
    POMAI_EXPECT_OK(db->Get(membrane, 10, &out));
    POMAI_EXPECT_TRUE(std::abs(out[0] - 2.0f) < 0.1f); // Still v2
    
    st = db->Get(membrane, 20, &out);
    POMAI_EXPECT_EQ(st.code(), pomai::ErrorCode::kNotFound); // Still deleted
    
    // Scan directory to verify we have 1 segment (impl detail, strict but good)
    fs::path memb_dir = fs::path(root) / "membranes" / membrane;
    int seg_count = 0;
    if (fs::exists(memb_dir)) {
        for (const auto& entry : fs::recursive_directory_iterator(memb_dir)) {
            if (entry.path().extension() == ".pom" || entry.path().extension() == ".dat") {
                seg_count++;
            }
        }
    }
    POMAI_EXPECT_EQ(seg_count, 1);
    (void)db->Close();
    db.reset();
    fs::remove_all(root);
}

} // namespace
