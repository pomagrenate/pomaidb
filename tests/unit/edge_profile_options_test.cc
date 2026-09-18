#include "tests/common/test_main.h"

#include "options.h"

namespace pomai {

POMAI_TEST(EdgeProfile_LowRamDoesNotOverrideIndexParams) {
    DBOptions opt;
    opt.edge_profile = EdgeProfile::kLowRam;
    opt.index_params.type = IndexType::kHnsw;
    opt.index_params.hnsw_m = 48;
    opt.index_params.hnsw_ef_construction = 256;
    opt.index_params.hnsw_ef_search = 96;
    opt.index_params.adaptive_threshold = 777;

    opt.ApplyEdgeProfile();

    POMAI_EXPECT_EQ(static_cast<int>(opt.index_params.type), static_cast<int>(IndexType::kHnsw));
    POMAI_EXPECT_EQ(opt.index_params.hnsw_m, 48u);
    POMAI_EXPECT_EQ(opt.index_params.hnsw_ef_construction, 256u);
    POMAI_EXPECT_EQ(opt.index_params.hnsw_ef_search, 96u);
    POMAI_EXPECT_EQ(opt.index_params.adaptive_threshold, 777u);
    // Dynamic sizing is now enabled by default (memtable_flush_threshold_mb = 0)
    POMAI_EXPECT_EQ(opt.memtable_flush_threshold_mb, 0u);
    POMAI_EXPECT_EQ(opt.memtable_budget_pct, 0.10f);  // 10% for memory-constrained
    POMAI_EXPECT_EQ(opt.memtable_min_threshold_mb, 8u);
    POMAI_EXPECT_EQ(opt.memtable_max_threshold_mb, 32u);
    POMAI_EXPECT_EQ(opt.max_memtable_mb, 64u);
    POMAI_EXPECT_EQ(static_cast<int>(opt.fsync), static_cast<int>(FsyncPolicy::kAlways));
}

POMAI_TEST(EdgeProfile_UserDefinedNoMutation) {
    DBOptions opt;
    opt.edge_profile = EdgeProfile::kUserDefined;
    opt.memtable_flush_threshold_mb = 77u;
    opt.max_memtable_mb = 1234u;

    opt.ApplyEdgeProfile();

    POMAI_EXPECT_EQ(opt.memtable_flush_threshold_mb, 77u);
    POMAI_EXPECT_EQ(opt.max_memtable_mb, 1234u);
}

}  // namespace pomai
