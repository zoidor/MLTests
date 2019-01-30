#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"

#include "png_helper.h"

#include <iostream>
#include <iterator>
		
const char* model_path = "application/data/test_model_tensorflow.pb";
const char* png_file_path = "segmentation/data/image.tif";
const int cropped_input_width = 64;
const int cropped_input_height = 64;
const bool useNNAPI = true;
const bool verbose = true;
const int number_of_threads = 4;

#define LOG(x) std::cerr

void RunInference() {

  if (!model_path) {
    LOG(ERROR) << "no model file name\n";
    exit(-1);
  }

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path);
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << model_path << "\n";
    exit(-1);
  }

  LOG(INFO) << "Loaded model " << model_path << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

  tflite::ops::builtin::BuiltinOpResolver resolver;

  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }

  interpreter->UseNNAPI(useNNAPI);

  if (verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
    LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }
  }

  if (number_of_threads != -1) {
    interpreter->SetNumThreads(number_of_threads);
  }

  auto in = malaria::read_png_and_resize(png_file_path, cropped_input_height, cropped_input_width, 3);

  if(in.empty()){
    LOG(INFO) << "unable to read: " << png_file_path << "\n";
    exit(-1);
  }

  int input = interpreter->inputs()[0];
  if (verbose) LOG(INFO) << "input: " << input << "\n";

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  if (verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
    exit(-1);
  }

  if (verbose) PrintInterpreterState(interpreter.get());

  // get input dimension from the input tensor metadata
  // assuming one input only
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  const int wanted_height = dims->data[1];
  const int wanted_width = dims->data[2];
  const int wanted_channels = dims->data[3];

  if(wanted_height != cropped_input_height){
	LOG(ERR) << "Height does not match";
        exit(-1);
  }

  if(wanted_width != cropped_input_width){
	LOG(ERR) << "Width does not match";
        exit(-1);
  }

  if(wanted_channels != 3){
	LOG(ERR) << "Only RGB images are supported";
        exit(-1);
  }

  std::copy(in.cbegin(), in.cend(), interpreter->typed_tensor<float>(input));

  
  if (interpreter->Invoke() != kTfLiteOk) {
	LOG(FATAL) << "Failed to invoke tflite!\n";
        exit(-1);
  }
  
  
  int output = interpreter->outputs()[0];
  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
  
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];
  
}


int main(int argc, char** argv) {

  RunInference();

  return 0;
}
