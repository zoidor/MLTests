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

  auto image_reader_float = tensorflow::ops::Cast(root.WithOpName("float_caster"), 
						  image_reader, tensorflow::DT_FLOAT);

  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = tensorflow::ops::ExpandDims(root, image_reader_float, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = tensorflow::ops::ResizeNearestNeighbor(
      root, dims_expander,
      tensorflow::ops::Const(root.WithOpName("size"), {input_height, input_width}));
  
  auto rescaled = tensorflow::ops::Div(root.WithOpName(output_name), resized, {255.0f});

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

class Predictor
{
	public:
		Predictor(std::unique_ptr<tensorflow::Session> session) 
		: m_session{std::move(session)}
		{}	


		tensorflow::Status inference(const std::string& png_file_path, tensorflow::Tensor& output){

			if(!m_session)
				return tensorflow::Status{tensorflow::error::Code::FAILED_PRECONDITION, 
				  "Session was not created correctly"};

			  // Get the image from disk as a float array of numbers, resized and normalized
	  		  // to the specifications the main graph expects.
			  tensorflow::Tensor resized_tensor;
			  tensorflow::Status read_tensor_status =
			  	ReadTensorFromPng(png_file_path, m_resized_input_height, m_resized_input_width, resized_tensor);
				
			 if(!read_tensor_status.ok())
				return read_tensor_status;
			 
		
			  // Actually run the image through the model.
			  std::vector<tensorflow::Tensor> outputs;
			  tensorflow::Status run_status = m_session->Run({{m_input_layer_name, resized_tensor}},
						           {m_output_layer_name}, {}, &outputs);
			  if (!run_status.ok()) {
			  	return run_status;
			  }

			  output = outputs[0];

			  if(output.dtype() != tensorflow::DT_FLOAT){
				return tensorflow::Status{tensorflow::error::Code::INTERNAL, 
				  "Output tensor of unexpected type"};
			}

			return tensorflow::Status::OK();
		}
	private:
		std::unique_ptr<tensorflow::Session> m_session;	
		std::string m_input_layer_name{"input_1"};
		std::string m_output_layer_name{"activation_42/Softmax"};
		int m_resized_input_height = 64;
  		int m_resized_input_width = 64;

};

std::vector<std::string> get_images(const std::string& dir_path){

	auto env = tensorflow::Env::Default();
	std::vector<std::string> to_ret;

	auto res = env->GetMatchingPaths(dir_path + "/*/*.png", &to_ret);
	if(!res.ok()) return {};

	return to_ret;
}

}//END anonymous namespace

int main(int argc, char* argv[]) {

  const char* model_path = "application/data/malaria_model.pb";
  const char* png_main_dir_path = "application/data/images";
  

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

  Predictor pred{std::move(session)};

  auto png_paths = get_images(png_main_dir_path);

  if(png_paths.empty()){
	std::cout<<"Unable to find images for inference in directory "<<png_main_dir_path;
	return -1;
  }

  for(const auto& png_file_path : png_paths){

	  std::cout << png_file_path << ": ";

  	  tensorflow::Tensor output;
	  auto inference_status = pred.inference(png_file_path, output);
	  
	  if(!inference_status.ok()){
		LOG(ERROR)<<"Image "<<  png_file_path << ", error: "<< inference_status;
		return -1;
	  }
	  
	  const float* data = output.flat<float>().data();
	  
	  for(int i = 0; i < output.NumElements(); ++i){
		std::cout <<  data[i] <<' ';
	  }  
	  std::cout<<'\n';
  }

  return 0;
}
