#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <fstream>

#include "pomai/pomai.h"

namespace {
using namespace pomai;

POMAI_TEST(NewMembrane_Fault_CorruptLog_Reopen) {
    auto path = pomai::test::TempDir("new_membrane_fault");
    DBOptions opt;
    opt.path = path;
    opt.dim = 4;
    opt.shard_count = 1;
    {
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));
        MembraneSpec kv; kv.name = "kv"; kv.kind = MembraneKind::kKeyValue; kv.shard_count = 1;
        POMAI_EXPECT_OK(db->CreateMembrane(kv));
        POMAI_EXPECT_OK(db->OpenMembrane("kv"));
        POMAI_EXPECT_OK(db->KvPut("kv", "k1", "v1"));
        POMAI_EXPECT_OK(db->Close());
    }

    std::fstream f(path + "/membranes/kv/kv.log", std::ios::in | std::ios::out | std::ios::binary);
    if (f.good()) {
        f.seekp(0, std::ios::end);
        auto sz = static_cast<std::streamoff>(f.tellp());
        if (sz > 2) {
            f.seekp(sz - static_cast<std::streamoff>(2));
            char z = '\xFF';
            f.write(&z, 1);
        }
    }

    std::unique_ptr<DB> reopened;
    POMAI_EXPECT_OK(DB::Open(opt, &reopened));
    std::string out;
    // tolerate corruption tail; must not crash
    auto st = reopened->KvGet("kv", "k1", &out);
    POMAI_EXPECT_TRUE(st.ok() || st.code() == ErrorCode::kNotFound);
    POMAI_EXPECT_OK(reopened->Close());
}

} // namespace

