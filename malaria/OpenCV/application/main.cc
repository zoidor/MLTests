#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <vector>
#include <iostream>

namespace{

const char* model_path = "../data/malaria_model.pb";
const char* png_main_dir_path = "../data/images";

//const char* input_layer_name = "input_1";
const char* output_layer_name = "activation_42/Softmax";
const int resized_input_height = 64;
const int resized_input_width = 64;  

cv::Mat resize(const cv::Mat& img, const int n_cols, const int n_rows){

	cv::Mat out_img;
	cv::resize(img, out_img, cv::Size(n_cols, n_rows), 0, 0, cv::INTER_NEAREST);
	return out_img;
}

cv::Mat readImage(const std::string& path){

  cv::Mat image = cv::imread(path); 

  if(image.cols == 0 || image.rows == 0) {
	std::cout << "unable to read image: " << path << '\n';
	return {};
  }

  if(image.channels() != 3){
	std::cout << "the number of channels is not correct for: " << path << '\n';
	return {};
  }

  image.convertTo(image, CV_32FC3,  1.0 / 255.0, 0.0);  	
  image = resize(image, resized_input_height, resized_input_width);

  if(image.channels() != 3){
	std::cout << "the number of channels of the resized image is not correct for: " << path << '\n';
	return {};
  }

  return image;
}

} //end anonymous namespace

int main(){

  auto net = cv::dnn::readNet(model_path);
  

  auto img = readImage(png_main_dir_path);
  if(img.cols == 0 || img.rows == 0)
	return -1;

  auto blob = cv::dnn::blobFromImage(img);

  net.setInput(blob);
  
  std::vector<cv::Mat> output;
  std::vector<std::string> outputLayers{output_layer_name};
  net.forward(output, outputLayers);
  
  if(output.size() != 1)
	return -1;

  if(output[0].total() != 2)
	return -1;

  std::cout << output[0].at<float>(0) << " " << output[0].at<float>(1) <<'\n';
}

