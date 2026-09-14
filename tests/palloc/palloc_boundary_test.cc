#include "tests/common/test_main.h"
#include "palloc_oracle.h"

#include <cstdint>
#include <cstring>
#include <vector>
#include <limits>
#include <iostream>

namespace pomai::palloc_qa {

POMAI_TEST(PallocBoundary_ZeroAndSmallSizes) {
    // 1. size = 0
    void* p0 = pa_malloc(0);
    // palloc returns non-null or null, but if non-null it must be safely freeable
    if (p0) {
        POMAI_EXPECT_TRUE(palloc_is_owned(p0));
        pa_free(p0);
    }

    // 2. size = 0 aligned
    void* p0_al = pa_malloc_aligned(0, 64);
    if (p0_al) {
        POMAI_EXPECT_EQ(reinterpret_cast<uintptr_t>(p0_al) % 64, 0u);
        pa_free(p0_al);
    }

    // 3. Free nullptr must be safe
    pa_free(nullptr);
    pa_free_aligned(nullptr, 64);
    pa_free_size(nullptr, 100);
    pa_free_size_aligned(nullptr, 100, 64);

    // 4. size = 1, 2, 3, 4, 7, 8
    for (size_t sz = 1; sz <= 8; ++sz) {
        void* p = pa_malloc(sz);
        POMAI_EXPECT_TRUE(p != nullptr);
        POMAI_EXPECT_TRUE(pa_usable_size(p) >= sz);
        std::memset(p, 0x55, sz);
        pa_free(p);
    }
}

POMAI_TEST(PallocBoundary_Alignments) {
    // Powers of two: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
    const size_t kAligns[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 65536};
    for (size_t align : kAligns) {
        for (size_t sz : {1, 15, 16, 17, 63, 64, 65, 255, 256, 1000, 4096}) {
            void* p = pa_malloc_aligned(sz, align);
            POMAI_EXPECT_TRUE(p != nullptr);
            POMAI_EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % (align < sizeof(void*) ? sizeof(void*) : align), 0u);
            POMAI_EXPECT_TRUE(pa_usable_size(p) >= sz);
            std::memset(p, 0xEE, sz);
            pa_free(p);
        }
    }
}

POMAI_TEST(PallocBoundary_SizeClassHammering) {
    // Attack classes: class-2, class-1, class, class+1, class+2
    const size_t kClasses[] = {
        8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384,
        512, 640, 768, 1024, 1280, 1536, 2048, 4096, 8192, 16384, 32768, 65536
    };

    AllocatorOracle oracle(true, 32);

    for (size_t c : kClasses) {
        size_t deltas[] = {c > 2 ? c - 2 : 1, c > 1 ? c - 1 : 1, c, c + 1, c + 2};
        std::vector<void*> ptrs;
        for (size_t sz : deltas) {
            void* p = oracle.TrackAlloc(sz, 16, static_cast<uint32_t>(sz));
            POMAI_EXPECT_TRUE(p != nullptr);
            ptrs.push_back(p);
        }
        for (void* p : ptrs) {
            POMAI_EXPECT_TRUE(oracle.VerifyAndTouch(p));
            oracle.TrackFree(p);
        }
    }
}

POMAI_TEST(PallocBoundary_IntegerOverflows) {
    // 1. SIZE_MAX
    void* p1 = pa_malloc(std::numeric_limits<size_t>::max());
    POMAI_EXPECT_TRUE(p1 == nullptr);

    // 2. SIZE_MAX - 1
    void* p2 = pa_malloc(std::numeric_limits<size_t>::max() - 1);
    POMAI_EXPECT_TRUE(p2 == nullptr);

    // 3. SIZE_MAX / 2 + 1000
    void* p3 = pa_malloc(std::numeric_limits<size_t>::max() / 2 + 1000);
    POMAI_EXPECT_TRUE(p3 == nullptr);

    // 4. calloc overflow
    void* p4 = pa_calloc(std::numeric_limits<size_t>::max() / 2 + 1, 2);
    POMAI_EXPECT_TRUE(p4 == nullptr);

    void* p5 = pa_calloc(std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max());
    POMAI_EXPECT_TRUE(p5 == nullptr);

    // 5. aligned malloc with overflow
    void* p6 = pa_malloc_aligned(std::numeric_limits<size_t>::max() - 63, 64);
    POMAI_EXPECT_TRUE(p6 == nullptr);
}

POMAI_TEST(PallocBoundary_ReallocExtinction) {
    // 1. realloc(nullptr, 64)
    void* p = pa_realloc(nullptr, 64);
    POMAI_EXPECT_TRUE(p != nullptr);
    std::memset(p, 0x42, 64);

    // 2. realloc(p, 64) same size
    p = pa_realloc(p, 64);
    POMAI_EXPECT_TRUE(p != nullptr);
    for (size_t i = 0; i < 64; ++i) {
        POMAI_EXPECT_EQ(static_cast<uint8_t*>(p)[i], 0x42);
    }

    // 3. grow to 256
    p = pa_realloc(p, 256);
    POMAI_EXPECT_TRUE(p != nullptr);
    for (size_t i = 0; i < 64; ++i) {
        POMAI_EXPECT_EQ(static_cast<uint8_t*>(p)[i], 0x42);
    }
    std::memset(static_cast<uint8_t*>(p) + 64, 0x43, 192);

    // 4. shrink to 32
    p = pa_realloc(p, 32);
    POMAI_EXPECT_TRUE(p != nullptr);
    for (size_t i = 0; i < 32; ++i) {
        POMAI_EXPECT_EQ(static_cast<uint8_t*>(p)[i], 0x42);
    }

    // 5. realloc with overflow must return NULL and KEEP original pointer valid!
    void* p_overflow = pa_realloc(p, std::numeric_limits<size_t>::max());
    POMAI_EXPECT_TRUE(p_overflow == nullptr);
    // Original p must still be intact!
    for (size_t i = 0; i < 32; ++i) {
        POMAI_EXPECT_EQ(static_cast<uint8_t*>(p)[i], 0x42);
    }

    // 6. realloc(p, 0)
    void* p_zero = pa_realloc(p, 0);
    if (p_zero) {
        pa_free(p_zero);
    }
}

} // namespace pomai::palloc_qa
