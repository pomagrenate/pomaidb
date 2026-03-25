#include "tests/common/test_main.h"

#include <array>
#include <vector>

#include "core/ai/no_train_dispatch.h"

namespace pomai::core::ai {

POMAI_TEST(NoTrainDispatch_AllMembraneKindsSupported) {
    const std::vector<float> features = {0.15f, 0.32f, 0.58f, 0.91f};
    const std::array<MembraneKind, 12> all_kinds = {
        MembraneKind::kVector, MembraneKind::kRag,   MembraneKind::kGraph, MembraneKind::kText,
        MembraneKind::kTimeSeries, MembraneKind::kKeyValue, MembraneKind::kSketch, MembraneKind::kBlob,
        MembraneKind::kSpatial, MembraneKind::kMesh, MembraneKind::kSparse, MembraneKind::kBitset,
    };

    for (const auto kind : all_kinds) {
        InferenceSummary out;
        POMAI_EXPECT_OK(InferNoTrainForKind(kind, features, &out));
        POMAI_EXPECT_TRUE(out.label != nullptr);
        POMAI_EXPECT_TRUE(out.explanation != nullptr);
        POMAI_EXPECT_TRUE(out.score >= 0.0f);
        POMAI_EXPECT_TRUE(out.score <= 1.0f);
    }
}

POMAI_TEST(NoTrainDispatch_ValidatesInputs) {
    InferenceSummary out;
    const std::vector<float> empty;
    const std::vector<float> non_empty = {0.2f, 0.3f};

    auto st = InferNoTrainForKind(MembraneKind::kVector, empty, &out);
    POMAI_EXPECT_TRUE(!st.ok());

    st = InferNoTrainForKind(MembraneKind::kVector, non_empty, nullptr);
    POMAI_EXPECT_TRUE(!st.ok());
}

} // namespace pomai::core::ai
