#include "tests/common/test_main.h"
#include "ai/analytical_engine.h"
#include <vector>
#include <iostream>
#include <cmath>

namespace pomai {
namespace core {

POMAI_TEST(AnalyticalEngine_ELMModelXOR) {
    AnalyticalEngine engine;
    POMAI_EXPECT_OK(engine.CreateELMModel("xor_model", 2, 20, 1));
    
    AnalyticalModel* model = engine.GetModel("xor_model");
    POMAI_EXPECT_TRUE(model != nullptr);

    // XOR dataset
    std::vector<float> inputs = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    std::vector<float> targets = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    // Train ELM in a single pass
    POMAI_EXPECT_OK(model->Train(inputs, targets, 4));

    // Verify predictions
    for (size_t i = 0; i < 4; ++i) {
        float out = 0.0f;
        model->Predict(std::span<const float>(inputs.data() + i * 2, 2), std::span<float>(&out, 1));
        
        // Since ELM uses L2 regularization, it won't be exact 0 or 1, but should be close
        bool predicted = out > 0.5f;
        bool expected = targets[i] > 0.5f;
        POMAI_EXPECT_TRUE(predicted == expected);
        std::cout << "Input: " << inputs[i*2] << ", " << inputs[i*2+1] 
                  << " | Target: " << targets[i] << " | Predicted: " << out << std::endl;
    }
}

} // namespace core
} // namespace pomai
