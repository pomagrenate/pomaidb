#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <filesystem>
#include <vector>
#include <span>

#include "segment_manifest.h"  // SegmentManifest
#include "options.h"
#include "pomai.h"
#include "search.h"
#include "types.h"
#include "manifest.h"
#include "segment.h"

namespace {

namespace fs = std::filesystem;

POMAI_TEST(SearchNewestWins_DeterministicAndTombstone) {
    const std::string root = pomai::test::TempDir("pomai-search-newest-wins");
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
    spec.metric = pomai::MetricType::kInnerProduct;  // exact match gives score 1.0; L2 would give 0
    POMAI_EXPECT_OK(db->CreateMembrane(spec));
    POMAI_EXPECT_OK(db->OpenMembrane(membrane));

    const pomai::VectorId target_id = 50000;
    const pomai::VectorId tomb_id = 60000;

    std::vector<float> vec_old = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec_new = {0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> vec_tomb = {0.0f, 0.0f, 1.0f, 0.0f};

    // 1. Put old version of target_id and tomb_id, then freeze
    POMAI_EXPECT_OK(db->Put(membrane, target_id, vec_old));
    POMAI_EXPECT_OK(db->Put(membrane, tomb_id, vec_tomb));
    POMAI_EXPECT_OK(db->Freeze(membrane));

    // 2. Put newer version of target_id, and delete tomb_id, then freeze
    POMAI_EXPECT_OK(db->Put(membrane, target_id, vec_new));
    POMAI_EXPECT_OK(db->Delete(membrane, tomb_id));
    POMAI_EXPECT_OK(db->Freeze(membrane));

    for (int i = 0; i < 50; ++i) {
        pomai::SearchResult res;
        POMAI_EXPECT_OK(db->Search(membrane, vec_new, 5, &res));
        POMAI_EXPECT_TRUE(!res.hits.empty());
        POMAI_EXPECT_EQ(res.hits[0].id, target_id);
        POMAI_EXPECT_TRUE(res.hits[0].score > 0.9f);  // IP: vec_new·vec_new = 1.0
    }

    {
        pomai::SearchResult res;
        POMAI_EXPECT_OK(db->Search(membrane, vec_tomb, 10, &res));
        bool found = false;
        for (const auto& hit : res.hits) {
            if (hit.id == tomb_id) {
                found = true;
                break;
            }
        }
        POMAI_EXPECT_TRUE(!found);
    }

    (void)db->Close();
    std::filesystem::remove_all(root);
}

} // namespace
