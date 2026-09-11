#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "storage_engine.hpp"

#include <cmath>
#include <filesystem>
#include <vector>

POMAI_TEST(StorageEngine_BasicAppendAndGet) {
    std::string dir = pomai::test::TempDir("storage_engine_basic");
    std::string path = dir + "/vectors.log";

    pomaidb::StorageEngine se;
    const std::uint32_t dim = 4;
    POMAI_EXPECT_OK(se.Open(path, dim, nullptr, 1024 * 1024, false));
    POMAI_EXPECT_TRUE(se.is_open());
    POMAI_EXPECT_EQ(se.dim(), dim);

    std::vector<float> v1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> v2 = {5.0f, 6.0f, 7.0f, 8.0f};

    POMAI_EXPECT_OK(se.Append(101, v1));
    POMAI_EXPECT_OK(se.Append(102, v2));

    // Get before flush (from buffer)
    pomaidb::StorageEngine::GetResult res1;
    POMAI_EXPECT_OK(se.Get(101, &res1));
    POMAI_EXPECT_TRUE(!res1.is_tombstone);
    POMAI_EXPECT_EQ(res1.dim, dim);
    for (std::uint32_t i = 0; i < dim; ++i) {
        POMAI_EXPECT_TRUE(std::fabs(res1.data[i] - v1[i]) < 1e-5f);
    }

    // Flush to disk and mmap
    POMAI_EXPECT_OK(se.Flush());

    // Get after flush (from mmap)
    pomaidb::StorageEngine::GetResult res2;
    POMAI_EXPECT_OK(se.Get(102, &res2));
    POMAI_EXPECT_TRUE(!res2.is_tombstone);
    POMAI_EXPECT_EQ(res2.dim, dim);
    for (std::uint32_t i = 0; i < dim; ++i) {
        POMAI_EXPECT_TRUE(std::fabs(res2.data[i] - v2[i]) < 1e-5f);
    }

    // Delete vector 101
    POMAI_EXPECT_OK(se.Delete(101));
    pomaidb::StorageEngine::GetResult res_del;
    POMAI_EXPECT_OK(se.Get(101, &res_del));
    POMAI_EXPECT_TRUE(res_del.is_tombstone);

    // Flush delete
    POMAI_EXPECT_OK(se.Flush());
    POMAI_EXPECT_OK(se.Get(101, &res_del));
    POMAI_EXPECT_TRUE(res_del.is_tombstone);

    POMAI_EXPECT_OK(se.Close());
    POMAI_EXPECT_TRUE(!se.is_open());

    // Reopen and reload from disk
    pomaidb::StorageEngine se2;
    POMAI_EXPECT_OK(se2.Open(path, dim, nullptr, 1024 * 1024, false));
    POMAI_EXPECT_TRUE(se2.is_open());

    pomaidb::StorageEngine::GetResult res_reopen;
    POMAI_EXPECT_OK(se2.Get(102, &res_reopen));
    POMAI_EXPECT_TRUE(!res_reopen.is_tombstone);
    for (std::uint32_t i = 0; i < dim; ++i) {
        POMAI_EXPECT_TRUE(std::fabs(res_reopen.data[i] - v2[i]) < 1e-5f);
    }

    POMAI_EXPECT_OK(se2.Get(101, &res_reopen));
    POMAI_EXPECT_TRUE(res_reopen.is_tombstone);

    POMAI_EXPECT_OK(se2.Close());
    std::filesystem::remove_all(dir);
}

POMAI_TEST(StorageEngine_AppendBatchQuantized) {
    std::string dir = pomai::test::TempDir("storage_engine_quant");
    std::string path = dir + "/quant.log";

    pomaidb::StorageEngine se;
    const std::uint32_t dim = 4;
    POMAI_EXPECT_OK(se.Open(path, dim, nullptr, 1024 * 1024, true));

    std::vector<pomai::VectorId> ids = {1, 2};
    std::vector<float> vecs = {
        0.0f, 0.5f, 1.0f, 2.0f,
        -1.0f, 0.0f, 1.0f, 3.0f
    };

    POMAI_EXPECT_OK(se.AppendBatch(ids, vecs, dim));
    POMAI_EXPECT_OK(se.Flush());

    pomaidb::StorageEngine::GetResult res;
    POMAI_EXPECT_OK(se.Get(1, &res));
    POMAI_EXPECT_TRUE(!res.is_tombstone);
    POMAI_EXPECT_EQ(res.dim, dim);
    // SQ8 quantization has bounded error
    for (std::uint32_t i = 0; i < dim; ++i) {
        POMAI_EXPECT_TRUE(std::fabs(res.data[i] - vecs[i]) < 0.05f);
    }

    POMAI_EXPECT_OK(se.Close());
    std::filesystem::remove_all(dir);
}
