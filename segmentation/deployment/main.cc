#include <fstream>
#include <utility>
#include <string>
#include <vector>

#include <cstdlib>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/io/path.h"

namespace { //anonymous namespace

tensorflow::Status ReadTensorFromTiff(const std::string& file_name, const int expected_cropped_height,
        const int expected_cropped_width, tensorflow::Tensor& out_tensor)
{

    const int batch_index = 1;
    tensorflow::TensorShape shape{batch_index, expected_cropped_height, expected_cropped_width, 1};
    out_tensor = tensorflow::Tensor{tensorflow::DT_FLOAT, shape};

    std::vector<std::uint16_t> ori_img(expected_cropped_height * expected_cropped_width, 6);
    std::copy(ori_img.cbegin(), ori_img.cend(), out_tensor.flat<float>().data());

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
          ReadTensorFromTiff("", cropped_input_height , cropped_input_width, resized_tensor);


  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }

  std::string input_layer = "input";
  std::string output_layer = "output";

  // Actually run the image through the model.
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status run_status = session->Run({{input_layer, resized_tensor}},
                                   {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  return 0;
}
