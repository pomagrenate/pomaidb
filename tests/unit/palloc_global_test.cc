#include "tests/common/test_main.h"
#include "palloc_compat.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace pomai {

POMAI_TEST(Palloc_GlobalNewDelete_Intercepted) {
    // 1. Primitive new/delete
    int* val = new int(42);
    POMAI_EXPECT_TRUE(val != nullptr);
    POMAI_EXPECT_EQ(*val, 42);
    POMAI_EXPECT_TRUE(palloc_is_owned(val));
    POMAI_EXPECT_TRUE(pa_usable_size(val) >= sizeof(int));
    delete val;

    // 2. Array new/delete
    float* arr = new float[1024];
    POMAI_EXPECT_TRUE(arr != nullptr);
    POMAI_EXPECT_TRUE(palloc_is_owned(arr));
    POMAI_EXPECT_TRUE(pa_usable_size(arr) >= 1024 * sizeof(float));
    delete[] arr;

    // 3. C++17 Aligned new/delete (64-byte alignment)
    void* aligned_p = operator new(256, std::align_val_t(64));
    POMAI_EXPECT_TRUE(aligned_p != nullptr);
    POMAI_EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned_p) % 64, static_cast<std::uintptr_t>(0));
    POMAI_EXPECT_TRUE(palloc_is_owned(aligned_p));
    operator delete(aligned_p, std::align_val_t(64));
}

POMAI_TEST(Palloc_StlContainers_UsePalloc) {
    // 1. Standard std::vector (uses global operator new via std::allocator)
    std::vector<float> vec(512, 3.14f);
    POMAI_EXPECT_EQ(vec.size(), static_cast<size_t>(512));
    POMAI_EXPECT_TRUE(palloc_is_owned(vec.data()));
    POMAI_EXPECT_TRUE(pa_usable_size(vec.data()) >= 512 * sizeof(float));

    // Grow vector to force reallocations
    for (int i = 0; i < 1000; ++i) {
        vec.push_back(static_cast<float>(i));
    }
    POMAI_EXPECT_TRUE(palloc_is_owned(vec.data()));

    // 2. std::string exceeding SSO
    std::string str(256, 'X');
    POMAI_EXPECT_TRUE(palloc_is_owned(str.data()));

    // 3. std::make_unique
    auto uptr = std::make_unique<std::vector<int>>(128, 7);
    POMAI_EXPECT_TRUE(palloc_is_owned(uptr.get()));
    POMAI_EXPECT_TRUE(palloc_is_owned(uptr->data()));

    // 4. std::make_shared
    auto sptr = std::make_shared<std::string>(128, 'Z');
    POMAI_EXPECT_TRUE(palloc_is_owned(sptr.get()));
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
    POMAI_EXPECT_TRUE(palloc_is_owned(p1));
    POMAI_EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p1) % 64, static_cast<std::uintptr_t>(0));

    void* p2 = pa_vec_arena_alloc(arena, 2048);
    POMAI_EXPECT_TRUE(p2 != nullptr);
    POMAI_EXPECT_TRUE(palloc_is_owned(p2));
    POMAI_EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p2) % 64, static_cast<std::uintptr_t>(0));

    pa_vec_arena_reset(arena);
    pa_vec_arena_destroy(arena);
}

} // namespace pomai
