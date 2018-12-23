#include <fstream>
#include <utility>
#include <string>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"

#include "tiff_utils.h"

namespace { //anonymous namespace

tensorflow::Status ReadTensorFromTiff(const std::string& file_name, const int expected_cropped_height,
        const int expected_cropped_width, tensorflow::Tensor& out_tensor)
{
	
	out_tensor = segmentation::readTiffImage(file_name.c_str(), 0, 0, expected_cropped_height, expected_cropped_width);

	if(out_tensor.NumElements() == 0)
		return tensorflow::errors::NotFound("Unable to decode TIFF file");

        return tensorflow::Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
std::unique_ptr<tensorflow::Session> LoadGraph(const std::string& graph_file_name)
{

  tensorflow::GraphDef graph_def;
    tensorflow::Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
        std::cout << "Load failed, error: " << load_graph_status << '\n';
	return nullptr;
  }

  std::unique_ptr<tensorflow::Session> session{tensorflow::NewSession(tensorflow::SessionOptions())};

  tensorflow::Status session_create_status = session->Create(graph_def);

  if (!session_create_status.ok()) {
    return nullptr;
  }

  return session;
}

}//END anonymous namespace

int main(int argc, char* argv[]) {

  const char* model_path = "segmentation/data/test_model_tensorflow.pb";
  const char * tiff_file = "segmentation/data/image.tif";
  const int cropped_input_height = 512;
  const int cropped_input_width = 688;

  if (argc == 0) {
    LOG(ERROR) << "Unable to detect executable name\n";
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  // First we load and initialize the model.
  auto session = LoadGraph(model_path);
  if (!session) {
    LOG(ERROR) << "Unable to load the model";
    return -1;
  }

  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  tensorflow::Tensor resized_tensor;
  tensorflow::Status read_tensor_status =
          ReadTensorFromTiff(tiff_file, cropped_input_height , cropped_input_width, resized_tensor);

  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }

  std::string input_layer{"input_1:0"};
  std::string output_layer{"conv2d_12/Sigmoid:0"};

  // Actually run the image through the model.
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status run_status = session->Run({{input_layer, resized_tensor}},
                                   {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  auto& output = outputs[0];

  auto& shape = output.shape();
  std::size_t area = 1;
  for(std::size_t i = 0; i < 4; ++i)
	area *= shape.dim_size(i);

  if(output.dtype() != tensorflow::DT_FLOAT){
	std::cout<<"Unexpected type for output layer, aborting\n";
	return -1; 
  }

  const float* data = output.flat<float>().data();
  for(std::size_t i = 0; i < 10; ++i)
	std::cout << data[i] << '\n';
  
  return 0;
}
