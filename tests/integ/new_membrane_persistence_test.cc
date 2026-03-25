#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include "pomai/pomai.h"

namespace {
using namespace pomai;

POMAI_TEST(NewMembrane_Persistence_Restart) {
    auto path = pomai::test::TempDir("new_membrane_persist");
    {
        DBOptions opt;
        opt.path = path;
        opt.dim = 4;
        opt.shard_count = 1;
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));

        MembraneSpec ts; ts.name = "ts"; ts.kind = MembraneKind::kTimeSeries; ts.shard_count = 1;
        MembraneSpec kv; kv.name = "kv"; kv.kind = MembraneKind::kKeyValue; kv.shard_count = 1;
        MembraneSpec bl; bl.name = "bl"; bl.kind = MembraneKind::kBlob; bl.shard_count = 1;
        POMAI_EXPECT_OK(db->CreateMembrane(ts)); POMAI_EXPECT_OK(db->OpenMembrane("ts"));
        POMAI_EXPECT_OK(db->CreateMembrane(kv)); POMAI_EXPECT_OK(db->OpenMembrane("kv"));
        POMAI_EXPECT_OK(db->CreateMembrane(bl)); POMAI_EXPECT_OK(db->OpenMembrane("bl"));

        POMAI_EXPECT_OK(db->TsPut("ts", 9, 1000, 1.25));
        POMAI_EXPECT_OK(db->KvPut("kv", "a", "b"));
        std::vector<uint8_t> payload = {9,8,7,6};
        POMAI_EXPECT_OK(db->BlobPut("bl", 99, payload));
        POMAI_EXPECT_OK(db->Close());
    }
    {
        DBOptions opt;
        opt.path = path;
        opt.dim = 4;
        opt.shard_count = 1;
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));

        std::vector<TimeSeriesPoint> points;
        POMAI_EXPECT_OK(db->TsRange("ts", 9, 900, 1100, &points));
        POMAI_EXPECT_EQ(points.size(), 1u);
        std::string out;
        POMAI_EXPECT_OK(db->KvGet("kv", "a", &out));
        POMAI_EXPECT_EQ(out, std::string("b"));
        std::vector<uint8_t> got;
        POMAI_EXPECT_OK(db->BlobGet("bl", 99, &got));
        POMAI_EXPECT_EQ(got.size(), 4u);
        POMAI_EXPECT_OK(db->Close());
    }
}

} // namespace

