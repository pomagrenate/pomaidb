#include "tests/common/test_main.h"
#include "palloc_compat.h"
#include "utils/palloc_allocator.h"
#include "utils/palloc_smart_ptr.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace pomai {

POMAI_TEST(Palloc_SmartPointers_UsePalloc) {
    // 1. alloc::UniquePtr::Make
    auto uptr = alloc::UniquePtr<int>::Make(nullptr, 42);
    POMAI_EXPECT_TRUE(uptr != nullptr);
    POMAI_EXPECT_EQ(*uptr, 42);
    POMAI_EXPECT_TRUE(palloc_is_owned(uptr.get()));
    POMAI_EXPECT_TRUE(pa_usable_size(uptr.get()) >= sizeof(int));

    // 2. alloc::UniquePtr::MakeAligned (64-byte aligned)
    auto uptr_aligned = alloc::UniquePtr<float>::MakeAligned(nullptr, 64, 3.14f);
    POMAI_EXPECT_TRUE(uptr_aligned != nullptr);
    POMAI_EXPECT_EQ(reinterpret_cast<std::uintptr_t>(uptr_aligned.get()) % 64, static_cast<std::uintptr_t>(0));
    POMAI_EXPECT_TRUE(palloc_is_owned(uptr_aligned.get()));

    // 3. alloc::SharedPtr::Make
    auto sptr = alloc::SharedPtr<int>::Make(nullptr, 100);
    POMAI_EXPECT_TRUE(sptr != nullptr);
    POMAI_EXPECT_EQ(*sptr, 100);
    POMAI_EXPECT_TRUE(palloc_is_owned(sptr.get()));

    // 4. alloc::SharedPtr::MakeAligned (64-byte aligned)
    auto sptr_aligned = alloc::SharedPtr<double>::MakeAligned(nullptr, 64, 2.718);
    POMAI_EXPECT_TRUE(sptr_aligned != nullptr);
    POMAI_EXPECT_EQ(reinterpret_cast<std::uintptr_t>(sptr_aligned.get()) % 64, static_cast<std::uintptr_t>(0));
    POMAI_EXPECT_TRUE(palloc_is_owned(sptr_aligned.get()));
}

POMAI_TEST(Palloc_StlContainers_UsePalloc) {
    // 1. PallocVector
    alloc::PallocVector<float> vec(512, 3.14f);
    POMAI_EXPECT_EQ(vec.size(), static_cast<size_t>(512));
    POMAI_EXPECT_TRUE(palloc_is_owned(vec.data()));
    POMAI_EXPECT_TRUE(pa_usable_size(vec.data()) >= 512 * sizeof(float));

    // Grow vector to force reallocations
    for (int i = 0; i < 1000; ++i) {
        vec.push_back(static_cast<float>(i));
    }
    POMAI_EXPECT_TRUE(palloc_is_owned(vec.data()));

    // 2. PallocUnorderedMap
    alloc::PallocUnorderedMap<uint64_t, uint64_t> map;
    for (uint64_t i = 0; i < 100; ++i) {
        map[i] = i * 10;
    }
    POMAI_EXPECT_EQ(map.size(), 100u);
    POMAI_EXPECT_EQ(map[42], 420u);
}

POMAI_TEST(Palloc_PallocAllocator_Alignment) {
    // Explicit 64-byte aligned STL allocator
    std::vector<float, pomai::PallocAllocator<float, 64>> custom_vec(128, 1.0f);
    POMAI_EXPECT_EQ(reinterpret_cast<std::uintptr_t>(custom_vec.data()) % 64, static_cast<std::uintptr_t>(0));
    POMAI_EXPECT_TRUE(palloc_is_owned(custom_vec.data()));
}

POMAI_TEST(Palloc_VectorModule_BatchAlloc) {
    // Native palloc vector module batch allocation
    constexpr size_t kBatchCount = 32;
    constexpr size_t kDim = 128;
    float** batch = pa_vector_batch_alloc_floats(kBatchCount, kDim);
    POMAI_EXPECT_TRUE(batch != nullptr);
    POMAI_EXPECT_TRUE(palloc_is_owned(batch));

    for (size_t i = 0; i < kBatchCount; ++i) {
        POMAI_EXPECT_TRUE(batch[i] != nullptr);
        POMAI_EXPECT_TRUE(palloc_is_owned(batch[i]));
        POMAI_EXPECT_EQ(reinterpret_cast<std::uintptr_t>(batch[i]) % 64, static_cast<std::uintptr_t>(0));
        batch[i][0] = static_cast<float>(i);
    }
    pa_vector_batch_free(reinterpret_cast<void**>(batch), kBatchCount);
}

POMAI_TEST(Palloc_VectorModule_Arena) {
    // Native palloc vector module bump arena
    pa_vec_arena_t* arena = pa_vec_arena_create(64 * 1024, 64);
    POMAI_EXPECT_TRUE(arena != nullptr);

    void* p1 = pa_vec_arena_alloc(arena, 1024);
    POMAI_EXPECT_TRUE(p1 != nullptr);
    POMAI_EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p1) % 64, static_cast<std::uintptr_t>(0));

    void* p2 = pa_vec_arena_alloc(arena, 2048);
    POMAI_EXPECT_TRUE(p2 != nullptr);
    POMAI_EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p2) % 64, static_cast<std::uintptr_t>(0));

    pa_vec_arena_reset(arena);
    pa_vec_arena_destroy(arena);
}

} // namespace pomai
