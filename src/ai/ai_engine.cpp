#include "ai/ai_engine.h"

#include <iostream>
#include <fstream>

// GGUF (cheesebrain) headers
#include "ai/cheesebrain_core/cheese-context.h"
#include "ai/cheesebrain_core/cheese-model-loader.h"

// TFLite headers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "ai/analytical_engine.h"

namespace tflite { namespace ops { namespace builtin {
TfLiteRegistration* Register_ADD();
TfLiteRegistration* Register_MUL();
TfLiteRegistration* Register_SUB();
TfLiteRegistration* Register_RESHAPE();
}}}

namespace pomai::core {

struct AIEngine::Impl {
    ModelType type;
    
    // GGUF state
    std::unique_ptr<cheese_context> gguf_ctx;
    
    // TFLite state
    std::unique_ptr<tflite::Interpreter> tflite_interpreter;
    std::unique_ptr<tflite::FlatBufferModel> tflite_model;

    // Analytical Backend
    std::unique_ptr<AnalyticalEngine> analytical_engine;

    Impl() : type(ModelType::kGGUF) {}
};

AIEngine::AIEngine() : impl_(std::make_unique<Impl>()) {}
AIEngine::~AIEngine() = default;

Status AIEngine::LoadModel(const std::string& path, ModelType type) {
    impl_->type = type;

    if (type == ModelType::kGGUF) {
        try {
            // Simplified GGUF loading using cheesebrain's loader.
            // In a real implementation, we'd pass palloc hints here.
            cheese_model_params mparams = cheese_model_default_params();
            (void)mparams;
            // impl_->gguf_ctx = ...
            return Status::Ok();
        } catch (const std::exception& e) {
            return Status::IoError(e.what());
        }
    } else {
        // TFLite loading
        impl_->tflite_model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
        if (!impl_->tflite_model) {
            return Status::IoError("Failed to load TFLite model from " + path);
        }

        tflite::MutableOpResolver resolver;
        // Register essential operations for custom models using the builtin op registration functions.
        // These are defined in the surgically restored kernel sources.
        resolver.AddBuiltin(tflite::BuiltinOperator_ADD, tflite::ops::builtin::Register_ADD());
        resolver.AddBuiltin(tflite::BuiltinOperator_MUL, tflite::ops::builtin::Register_MUL());
        resolver.AddBuiltin(tflite::BuiltinOperator_SUB, tflite::ops::builtin::Register_SUB());
        resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE, tflite::ops::builtin::Register_RESHAPE());
        
        tflite::InterpreterBuilder builder(*impl_->tflite_model, resolver);
        if (builder(&impl_->tflite_interpreter) != kTfLiteOk) {
            return Status::Internal("Failed to build TFLite interpreter");
        }

        if (impl_->tflite_interpreter->AllocateTensors() != kTfLiteOk) {
            return Status::Internal("Failed to allocate TFLite tensors");
        }
        return Status::Ok();
    }
}

bool AIEngine::StepInference(float* progress) {
    if (impl_->type == ModelType::kTensor && impl_->tflite_interpreter) {
        // TFLite standard Invoke is not natively sliceable without custom kernels,
        // but for PomaiDB we'll wrap it or use a sub-graph approach if needed.
        if (impl_->tflite_interpreter->Invoke() == kTfLiteOk) {
            if (progress) *progress = 1.0f;
            return true;
        }
        return false;
    }
    
    // TODO: Implement GGUF sliceable stepping (token by token)
    return true; 
}

void AIEngine::SetInput(std::span<const float> data) {
    if (impl_->type == ModelType::kTensor && impl_->tflite_interpreter) {
        float* input = impl_->tflite_interpreter->typed_input_tensor<float>(0);
        if (input) {
            std::copy(data.begin(), data.end(), input);
        }
    }
}

std::span<const float> AIEngine::GetOutput(int index) const {
    if (impl_->type == ModelType::kTensor && impl_->tflite_interpreter) {
        const float* output = impl_->tflite_interpreter->typed_output_tensor<float>(index);
        size_t size = impl_->tflite_interpreter->output_tensor(index)->bytes / sizeof(float);
        return {output, size};
    }
    return {};
}

std::string AIEngine::GetTextResult() const {
    // Return placeholder for now.
    return "AI generation result placeholder";
}

} // namespace pomai::core
