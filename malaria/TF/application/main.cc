#include <fstream>
#include <utility>
#include <string>
#include <fstream>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/ops/image_ops.h"

namespace { //anonymous namespace

tensorflow::Status ReadEntireFile(tensorflow::Env* env, const std::string& filename,
                             tensorflow::Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  std::string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<std::string>()() = std::string(data);
  return tensorflow::Status::OK();
}


// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
tensorflow::Status ReadTensorFromPng(const std::string& file_name, const int input_height,
                               const int input_width, 
                               tensorflow::Tensor& out_tensor) {
  
  auto root = tensorflow::Scope::NewRootScope();

  std::string input_name = "file_reader";
  std::string output_name = "read_image";

  // read file_name into a tensor named input
  tensorflow::Tensor input{tensorflow::DT_STRING, tensorflow::TensorShape()};
  TF_RETURN_IF_ERROR(ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader =
      tensorflow::ops::Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };

  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  image_reader = tensorflow::ops::DecodePng(root.WithOpName("png_reader"), file_reader,
                     tensorflow::ops::DecodePng::Channels(wanted_channels));

  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = tensorflow::ops::ExpandDims(root, image_reader, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = tensorflow::ops::ResizeBilinear(
      root.WithOpName(output_name), dims_expander,
      tensorflow::ops::Const(root.WithOpName("size"), {input_height, input_width}));
  
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  std::vector<tensorflow::Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, &out_tensors));

  if(out_tensors.size() != 1)
	return tensorflow::Status{tensorflow::error::Code::FAILED_PRECONDITION, 
				  "output of reading png must contain only one tensor"};

  out_tensor = std::move(out_tensors[0]);

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

  const char* model_path = "application/data/malaria_model.pb";
  const char* png_file_path = "application/data/images/Parasitized/C182P143NThinF_IMG_20151201_172607_cell_45.png";
  const int resized_input_height = 64;
  const int resized_input_width = 64;

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
          ReadTensorFromPng(png_file_path, resized_input_height , resized_input_width, resized_tensor);

  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }

  std::string input_layer{"input_1"};
  std::string output_layer{"activation_42/Softmax"};

  // Actually run the image through the model.
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status run_status = session->Run({{input_layer, resized_tensor}},
                                   {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  auto& output = outputs[0];

  if(output.dtype() != tensorflow::DT_FLOAT){
	std::cout<<"Unexpected type for output layer, aborting\n";
	return -1; 
  }

  return 0;
}
