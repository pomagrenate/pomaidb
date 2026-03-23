#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include <string.h>

namespace tflite {
namespace ops {
namespace builtin {

TfLiteStatus ReshapeEval(TfLiteContext* context, TfLiteNode* node) {
     const TfLiteTensor* input = tflite::GetInput(context, node, 0);
     TfLiteTensor* output = tflite::GetOutput(context, node, 0);
     
     if (input->data.raw != output->data.raw) {
         memcpy(output->data.raw, input->data.raw, input->bytes);
     }
     return kTfLiteOk;
}

TfLiteRegistration* Register_RESHAPE() {
  static TfLiteRegistration r = {nullptr, nullptr, nullptr, ReshapeEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
