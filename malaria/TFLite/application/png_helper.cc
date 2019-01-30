#include "png_helper.h"

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/version.h"

#include "png.h"

#include <memory>
#include <cstdio>

namespace{

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
  
  if(!resize_op)
	return{};

  auto op_params = static_cast<TfLiteResizeNearestNeighborParams*>(malloc(sizeof(TfLiteResizeNearestNeighborParams)));
  op_params->align_corners = false;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, op_params, resize_op,
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
  	out.push_back(output[i] / 255.0f);
  }
  return out;
}

class fCloseGuard{
	private:
		FILE * fp;
 	public:
		fCloseGuard(FILE * fp) : fp{fp}{}
		~fCloseGuard(){fclose(fp);}
};

class pngPtrsGuard
{
	private:
		png_structp& png;
		png_infop& info;

	public: 
		pngPtrsGuard(png_structp& png, png_infop& info) : png{png}, info{info} {}
		~pngPtrsGuard(){
			if(!png) return;
			png_destroy_read_struct(&png, info ? &info : nullptr, nullptr);
		}

};

}//end anonymous namespace



std::vector<float> malaria::read_png_and_resize(const std::string& path, const int wanted_height, const int wanted_width, const int wanted_channels){
  
  FILE *fp = fopen(path.c_str(), "rb");

  fCloseGuard guard(fp);

  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = nullptr;

  if(!png) return{};

  pngPtrsGuard pngPtrsGrd{png, info};

  info = png_create_info_struct(png);
  if(!info) return{};

  if(setjmp(png_jmpbuf(png))) return{};

  png_init_io(png, fp);
  png_read_info(png, info);

  const int width           = png_get_image_width(png, info);
  const int height          = png_get_image_height(png, info);
  const png_byte color_type = png_get_color_type(png, info);
  const png_byte bit_depth  = png_get_bit_depth(png, info);
  const int channels        = png_get_channels(png, info);

  if(bit_depth != 8 || channels != 3)
    return {};

  if(color_type != PNG_COLOR_TYPE_RGB)
    return{};

  auto row_bytes = png_get_rowbytes(png, info);
  std::vector<std::uint8_t> out8(height * row_bytes);
  
  auto row = out8.data();
  for (int h = height; h-- != 0; row += row_bytes) {
	png_read_row(png, row, nullptr);
  }

  return resize(out8, height, width, wanted_channels, wanted_height, wanted_width, wanted_channels);
}
