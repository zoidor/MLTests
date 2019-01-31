#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"

#include "png_helper.h"

#include <iterator>
#include <iostream>
#include <cmath>
	
#define LOG(x) std::cerr

struct InferOutput{
	float c1{0};
	float c2{0};
	bool isValid() const {return !std::isnan(c1) && !std::isnan(c2);}
	InferOutput(const float c1, const float c2) : c1(c1), c2(c2) {}
};

class Inference{
	private:
		std::unique_ptr<tflite::Interpreter> interpreter;
		tflite::ops::builtin::BuiltinOpResolver resolver;
		std::unique_ptr<tflite::FlatBufferModel> model;	
		int input{0};
		int output{0};
	public:
		Inference(const char * model_path, const bool useNNAPI, const int number_of_threads = -1){

			if(!model_path)
				return;
					
			model = tflite::FlatBufferModel::BuildFromFile(model_path);
			if (!model) {
				LOG(FATAL) << "Failed to mmap model " << model_path << "\n";
				return;
			}

			tflite::InterpreterBuilder(*model, resolver)(&interpreter);
			if (!interpreter) {
				LOG(FATAL) << "Failed to construct interpreter\n";
				return;
			}
			
			interpreter->UseNNAPI(useNNAPI);

			if (number_of_threads != -1) {
				interpreter->SetNumThreads(number_of_threads);
			}

			if (interpreter->AllocateTensors() != kTfLiteOk) {
			    interpreter = nullptr;
			}

			const auto& inputs = interpreter->inputs();
			const auto& outputs = interpreter->outputs();

			input = inputs[0];
			output = outputs[0];
		}
		
		bool isInterpreterInstantiated() const {return interpreter != nullptr;}	
		
		int wanted_height() const {
			if(!isInterpreterInstantiated()) return -1;
			return interpreter->tensor(input)->dims->data[1];
		}

		int wanted_width() const {
			if(!isInterpreterInstantiated()) return -1;
			return interpreter->tensor(input)->dims->data[2];
		}

		int wanted_channels() const {
			if(!isInterpreterInstantiated()) return -1;
			return interpreter->tensor(input)->dims->data[3];
		}

		InferOutput calculateInference(const char * png_file_path){
			
			if(!isInterpreterInstantiated()) return {NAN, NAN};
	
			auto in = malaria::read_png_and_resize(png_file_path, wanted_height(), wanted_width(), wanted_channels());

			if(in.size() != (wanted_height() * wanted_width() * wanted_channels())){
				LOG(FATAL) << "Resize  failed, output of wrong size: " << in.size() << '\n';
				return {NAN, NAN};
			}

			std::copy(in.cbegin(), in.cend(), interpreter->typed_tensor<float>(input));

			if (interpreter->Invoke() != kTfLiteOk) {
				LOG(FATAL) << "Interpreter invocation failed\n";
				return {NAN, NAN};
			}

			TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;

			auto output_size = output_dims->data[output_dims->size - 1];

			if (output_size != 2) {
				LOG(FATAL) << "Output tensor size different\n";
				return {NAN, NAN};
			}

			auto data = interpreter->typed_tensor<float>(output);

			return InferOutput{data[0], data[1]};		  	
		}
};

void RunInference() {

  const char* model_path = "application/data/malaria_model.tflite";
  const bool useNNAPI = true;
  const int number_of_threads = 4;

  Inference infer{model_path, useNNAPI, number_of_threads};
  
  if(!infer.isInterpreterInstantiated()){
    LOG(FATAL) << "Failed to instantiated interpreter\n";
    exit(-1);
  }
  
  for(const auto& path : malaria::get_png_paths()){
  	auto p = infer.calculateInference(path.c_str()); 
  	std::cout << path << ' ' << p.c1 << ' ' << p.c2 << '\n';
 }
}


int main(int argc, char** argv) {

  RunInference();

  return 0;
}
