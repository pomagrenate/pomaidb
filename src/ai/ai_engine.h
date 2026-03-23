#pragma once

#include <memory>
#include <string>
#include <vector>
#include <span>
#include <variant>

#include "pomai/status.h"

namespace pomai::core {
enum class ModelType {
    kGGUF,       // Large Language Models / SLMs
    kTensor,     // ML models (TFLite)
    kAnalytical, // "No-Train" deterministic models (ELM, RKS)
};

/**
 * AIEngine provides a unified, edge-optimized interface for local AI inference.
 * It abstracts over GGUF (cheesebrain) and TFLite (tensor_core) backends.
 */
class AIEngine {
public:
    AIEngine();
    ~AIEngine();

    // Disable copy for safety on memory-constrained edge devices.
    AIEngine(const AIEngine&) = delete;
    AIEngine& operator=(const AIEngine&) = delete;

    /** Load a model into the specific backend. */
    Status LoadModel(const std::string& path, ModelType type);

    /** 
     * Perform a single inference slice. 
     * Returns true if inference is complete, false if more slices are needed (re-queue).
     * This ensures the AI doesn't block the MicroKernel/TaskScheduler.
     */
    bool StepInference(float* progress = nullptr);

    /** Set input tensor data. Zero-copy where possible. */
    void SetInput(std::span<const float> data);

    /** Get output. Result index depends on the model. */
    std::span<const float> GetOutput(int index = 0) const;

    /** For GGUF: Get generated text chunk. */
    std::string GetTextResult() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pomai::core
