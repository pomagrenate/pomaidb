#pragma once

#include <vector>
#include <memory>
#include <string>
#include <span>
#include <random>

#include "pomai/status.h"

namespace pomai::core {

/**
 * AnalyticalModel represents a "no-train" AI model that uses deterministic
 * projections and analytical solutions (e.g., Extreme Learning Machine).
 */
class AnalyticalModel {
public:
    virtual ~AnalyticalModel() = default;

    /**
     * Train the model on a batch of data in a single pass.
     * inputs: matrix of size [num_samples, input_dim]
     * targets: matrix of size [num_samples, output_dim]
     */
    virtual Status Train(std::span<const float> inputs, 
                        std::span<const float> targets,
                        size_t num_samples) = 0;

    /** Perform inference on a single sample. */
    virtual void Predict(std::span<const float> input, 
                        std::span<float> output) const = 0;

    virtual size_t InputDim() const = 0;
    virtual size_t OutputDim() const = 0;
};

/**
 * Extreme Learning Machine (ELM) implementation.
 * Uses a fixed random hidden layer and solves for output weights analytically.
 */
class ELMModel : public AnalyticalModel {
public:
    ELMModel(size_t input_dim, size_t hidden_dim, size_t output_dim, float l2_reg = 1e-4f);
    ~ELMModel() override;

    Status Train(std::span<const float> inputs, 
                std::span<const float> targets,
                size_t num_samples) override;

    void Predict(std::span<const float> input, 
                std::span<float> output) const override;

    size_t InputDim() const override { return input_dim_; }
    size_t OutputDim() const override { return output_dim_; }

private:
    size_t input_dim_;
    size_t hidden_dim_;
    size_t output_dim_;
    float l2_reg_;

    std::vector<float> input_weights_; // [hidden_dim, input_dim] - fixed random
    std::vector<float> input_bias_;    // [hidden_dim] - fixed random
    std::vector<float> output_weights_; // [output_dim, hidden_dim] - solved analytically

    void InitializeWeights();
    void MapToHidden(std::span<const float> input, std::span<float> hidden) const;
};

/**
 * AnalyticalEngine manages analytical models within PomaiDB.
 */
class AnalyticalEngine {
public:
    AnalyticalEngine();
    ~AnalyticalEngine();

    static AnalyticalEngine& Global();

    Status CreateELMModel(const std::string& name, size_t in_dim, size_t hidden_dim, size_t out_dim);
    AnalyticalModel* GetModel(const std::string& name);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pomai::core
