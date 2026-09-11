#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include "palloc_page_pool.h"

#include <cstdint>
#include <cstring>
#include <filesystem>

POMAI_TEST(PallocPagePool_CreateAndStats)
{
    palloc_page_pool_options opts{};
    opts.page_size = 4096;
    opts.capacity_bytes = 4096 * 4;  // 4 pages
    opts.swap_file_path = nullptr;   // swap disabled for this basic test
    opts.device_profile = PALLOC_DEVICE_PROFILE_SMALL;

    palloc_page_pool* pool = palloc_page_pool_create(&opts);
    POMAI_EXPECT_TRUE(pool != nullptr);

    palloc_page_pool_stats stats{};
    palloc_page_pool_get_stats(pool, &stats);
    POMAI_EXPECT_EQ(stats.page_size, opts.page_size);
    POMAI_EXPECT_EQ(stats.capacity_bytes, opts.capacity_bytes);

    palloc_page_pool_destroy(pool);
}

POMAI_TEST(PallocPagePool_SwapEvictionAndReload)
{
    std::string dir = pomai::test::TempDir("palloc_swap_test");
    std::string swap_path = dir + "/swap.bin";

    palloc_page_pool_options opts{};
    opts.page_size = 4096;
    opts.capacity_bytes = 4096 * 2;  // Exactly 2 resident pages
    opts.swap_file_path = swap_path.c_str();
    opts.device_profile = PALLOC_DEVICE_PROFILE_SMALL;

    palloc_page_pool* pool = palloc_page_pool_create(&opts);
    POMAI_EXPECT_TRUE(pool != nullptr);

    // Fetch page 1, write pattern
    int is_new = 0;
    char* p1 = static_cast<char*>(palloc_fetch_page(pool, 1, 1, &is_new));
    POMAI_EXPECT_TRUE(p1 != nullptr);
    POMAI_EXPECT_TRUE(is_new != 0);
    std::memset(p1, 0xAB, 4096);
    palloc_unpin_page(pool, 1, 1);

    // Fetch page 2, write pattern
    char* p2 = static_cast<char*>(palloc_fetch_page(pool, 2, 1, &is_new));
    POMAI_EXPECT_TRUE(p2 != nullptr);
    POMAI_EXPECT_TRUE(is_new != 0);
    std::memset(p2, 0xCD, 4096);
    palloc_unpin_page(pool, 2, 1);

    // Fetch page 3 - capacity is 2 pages, so one page must be evicted to swap
    char* p3 = static_cast<char*>(palloc_fetch_page(pool, 3, 1, &is_new));
    POMAI_EXPECT_TRUE(p3 != nullptr);
    std::memset(p3, 0xEF, 4096);
    palloc_unpin_page(pool, 3, 1);

    palloc_page_pool_stats stats{};
    palloc_page_pool_get_stats(pool, &stats);
    POMAI_EXPECT_TRUE(stats.evictions > 0);
    POMAI_EXPECT_TRUE(stats.bytes_in_swap > 0);

    // Now reload page 1 from swap!
    char* p1_reload = static_cast<char*>(palloc_fetch_page(pool, 1, 0, &is_new));
    POMAI_EXPECT_TRUE(p1_reload != nullptr);
    POMAI_EXPECT_EQ(is_new, 0);
    // Verify contents survived swap roundtrip
    for (size_t i = 0; i < 4096; ++i) {
        POMAI_EXPECT_EQ(static_cast<unsigned char>(p1_reload[i]), 0xAB);
    }
    palloc_unpin_page(pool, 1, 0);

    palloc_page_pool_destroy(pool);
    std::filesystem::remove_all(dir);
}


