#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/utils/proto_utils.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>

std::string init_net_fname{"../data/init_net.pb"};
std::string predict_net_fname{"../data/predict_net.pb"};

std::string file{"../data/images/Parasitized/C99P60ThinF_IMG_20150918_142334_cell_8.png"};

const int resized_input_height = 64;
const int resized_input_width = 64;

namespace {

cv::Mat resize(const cv::Mat& img, const int n_cols, const int n_rows){

	cv::Mat out_img;
	cv::resize(img, out_img, cv::Size(n_cols, n_rows), 0, 0, cv::INTER_NEAREST);
	return out_img;
}

int run() {
  
  cv::Mat image = cv::imread(file); 

  if(image.cols == 0 || image.rows == 0) {
	std::cout << "unable to read image: " << file << '\n';
	return -1;
  }

  if(image.channels() != 3){
	std::cout << "the number of channels is not correct for: " << file << '\n';
	return -1;
  }

  image.convertTo(image, CV_32FC3,  1.0 / 255.0, 0.0);  	
  image = resize(image, resized_input_height, resized_input_width);

  const std::size_t img_num_pixels = image.cols * image.rows * image.channels();

  std::vector<float> data;
  data.reserve(img_num_pixels);
  
  std::vector<cv::Mat> imgChannels(image.channels());
  cv::split(image, imgChannels);

  for(int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x){
	for(int channel = 0; channel < image.channels(); ++channel){
		const auto px = imgChannels[channel].at<float>(y, x);
		data.push_back(px);
	}
    }
  }

  std::vector<std::int64_t> dims({1, image.rows, image.cols, image.channels()});

  auto tensor = caffe2::TensorCPUFromValues(dims, at::ArrayRef<decltype(data)::value_type>{data});
 
  caffe2::NetDef init_net, predict_net;

  CAFFE_ENFORCE(caffe2::ReadProtoFromFile(init_net_fname, &init_net));
  CAFFE_ENFORCE(caffe2::ReadProtoFromFile(predict_net_fname, &predict_net));

  caffe2::Workspace workspace("default");
  CAFFE_ENFORCE(workspace.RunNetOnce(init_net));
   
  const char * inputLayerName = "input_1_0";

  if(!workspace.HasBlob(inputLayerName)){
	std::cout << "Unable to load input layer " << inputLayerName <<'\n';
	return -1;
  }

  auto input = workspace.CreateBlob(inputLayerName)->GetMutable<caffe2::TensorCPU>();

  input->ResizeLike(tensor);
  input->ShareData(tensor);
 
  CAFFE_ENFORCE(workspace.RunNetOnce(predict_net));

  auto &output_name = predict_net.external_output(0);
  auto output = workspace.GetBlob(output_name)->Get<caffe2::TensorCPU>();

  if(static_cast<std::size_t>(output.numel()) != 2){
	std::cout<<"Output tensor should have 2 elements\n";
	return -1;
   }

  return 0;
}

}  // anonymous namespace

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  auto success = run();
  return success;
}
