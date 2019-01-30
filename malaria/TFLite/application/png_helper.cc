#include "png_helper.h"

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/version.h"

#include "tensorflow/core/lib/png/png_io.h"

#include <fstream>
#include <memory>

namespace{

std::string read_all_png(const std::string& path){

  int begin, end;

  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    return "";
  }

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t len = end - begin;

  std::string img_bytes(len, '\0');
  file.seekg(0, std::ios::beg);
  file.read(&img_bytes[0], len);
  return img_bytes;
}


std::vector<float> resize(const std::vector<std::uint8_t>& in, const int image_height, const int image_width,
            const int image_channels, const int wanted_height, const int wanted_width,
            const int wanted_channels) {

  int number_of_pixels = image_height * image_width * image_channels;
  std::unique_ptr<tflite::Interpreter> interpreter{new tflite::Interpreter{}};

  int base_index = 0;

  // two inputs: input and new_sizes
  interpreter->AddTensors(2, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);
  // set input and output tensors
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});

  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input",
      {1, image_height, image_width, image_channels}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                            quant);
  interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "output",
      {1, wanted_height, wanted_width, wanted_channels}, quant);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  const TfLiteRegistration* resize_op =
      resolver.FindOp(tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR, 1);
  TfLiteResizeNearestNeighborParams op_params{false};
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, &op_params, resize_op,
                                     nullptr);

  interpreter->AllocateTensors();

  // fill input image
  // in[] are integers, cannot do memcpy() directly
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < number_of_pixels; i++) {
    input[i] = in[i];
  }

  // fill new_sizes
  interpreter->typed_tensor<int>(1)[0] = wanted_height;
  interpreter->typed_tensor<int>(1)[1] = wanted_width;

  if(interpreter->Invoke() != kTfLiteOk)
	return {};

  auto output = interpreter->typed_tensor<float>(2);
  auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;
  std::vector<float> out;
  out.reserve(output_number_of_pixels);
  for (int i = 0; i < output_number_of_pixels; i++) {
  	out.push_back(output[i]);
  }
  return out;
}

}//end anonymous namespace



std::vector<float> malaria::read_png_and_resize(const std::string& path, const int wanted_height, const int wanted_width, const int wanted_channels){

 auto input = read_all_png(path);

 if(input.empty()) return {};

 tensorflow::png::DecodeContext decode;
 const int wanted_num_bits = 8;

 if(!tensorflow::png::CommonInitDecode(input, wanted_channels, wanted_num_bits, &decode)){
	tensorflow::png::CommonFreeDecode(&decode);
	return {};
 }

 const int width = static_cast<int>(decode.width);
 const int height = static_cast<int>(decode.height);
 const std::int64_t total_size =
        static_cast<std::int64_t>(width) * static_cast<std::int64_t>(height);
    
 if (width != static_cast<std::int64_t>(decode.width) || width <= 0 ||
        width >= (1LL << 27) || height != static_cast<std::int64_t>(decode.height) ||
        height <= 0 || height >= (1LL << 27) || total_size >= (1LL << 29)) {
      return {}; //image too big
 }

 const int num_pixels = decode.channels * width * height;
 std::vector<std::uint8_t> out8(static_cast<std::size_t>(num_pixels));
 const int row_bytes = decode.channels * width * sizeof(std::uint8_t);
 
 if(!tensorflow::png::CommonFinishDecode(
              reinterpret_cast<png_bytep>(out8.data()),
              row_bytes, &decode))
	return {};

 return resize(out8, height, width, wanted_channels, wanted_height, wanted_width, wanted_channels);

}
