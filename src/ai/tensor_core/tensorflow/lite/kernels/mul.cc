#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {

TfLiteStatus MulEval(TfLiteContext* context, TfLiteNode* node) {
     const TfLiteTensor* input1 = tflite::GetInput(context, node, 0);
     const TfLiteTensor* input2 = tflite::GetInput(context, node, 1);
     TfLiteTensor* output = tflite::GetOutput(context, node, 0);
     
     if (output->type == kTfLiteFloat32) {
         float* out_data = output->data.f;
         const float* in1 = input1->data.f;
         const float* in2 = input2->data.f;
         int count = tflite::NumElements(output);
         for (int i = 0; i < count; ++i) out_data[i] = in1[i] * in2[i];
     }
     return kTfLiteOk;
}

TfLiteRegistration* Register_MUL() {
  static TfLiteRegistration r = {nullptr, nullptr, nullptr, MulEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
