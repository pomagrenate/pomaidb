#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <fstream>

#include "pomai/pomai.h"

namespace {
using namespace pomai;

POMAI_TEST(SimdNewMembranes_Fault_ReopenAfterCorruptTail) {
    auto path = pomai::test::TempDir("simd_membranes_fault");
    DBOptions opt;
    opt.path = path;
    opt.dim = 4;
    opt.shard_count = 1;
    {
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));
        MembraneSpec sp; sp.name = "sp"; sp.kind = MembraneKind::kSpatial; sp.shard_count = 1;
        POMAI_EXPECT_OK(db->CreateMembrane(sp)); POMAI_EXPECT_OK(db->OpenMembrane("sp"));
        POMAI_EXPECT_OK(db->SpatialPut("sp", 1, 10.0, 10.0));
        POMAI_EXPECT_OK(db->Close());
    }
    std::fstream f(path + "/membranes/sp/spatial.log", std::ios::in | std::ios::out | std::ios::binary);
    if (f.good()) {
        f.seekp(0, std::ios::end);
        const auto sz = static_cast<std::streamoff>(f.tellp());
        if (sz > 1) {
            f.seekp(sz - static_cast<std::streamoff>(1));
            char bad = '\xFF';
            f.write(&bad, 1);
        }
    }
    std::unique_ptr<DB> db2;
    POMAI_EXPECT_OK(DB::Open(opt, &db2));
    std::vector<SpatialPoint> out;
    auto st = db2->SpatialRadiusSearch("sp", 10.0, 10.0, 100.0, &out);
    POMAI_EXPECT_TRUE(st.ok() || st.code() == ErrorCode::kNotFound);
    POMAI_EXPECT_OK(db2->Close());
}

} // namespace

