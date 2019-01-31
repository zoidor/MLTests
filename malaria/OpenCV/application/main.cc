#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <vector>
#include <iostream>

namespace{

cv::Mat resize(const cv::Mat& img, const int n_cols, const int n_rows){

	cv::Mat out_img;
	cv::resize(img, out_img, cv::Size(n_cols, n_rows), 0, 0, cv::INTER_NEAREST);
	return out_img;
}

cv::Mat readImageAndResize(const std::string& path, const int resized_input_height = 64, const int resized_input_width = 64){

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

std::vector<std::string> get_paths(const std::string& base_path){
	std::vector<std::string> out;
	const bool recursive = true;
	cv::utils::fs::glob(base_path, "*.png", out, recursive);
	return out;
}

} //end anonymous namespace

int main(){

  const char* model_path = "../data/malaria_model.pb";
  const char* model_txt_path = "../data/malaria_model.pbtxt";
  const char* output_layer_name = "activation_42/Softmax";
  const char* png_main_dir_path = "../data/images";

  auto net = cv::dnn::readNetFromTensorflow(model_path, model_txt_path);
  std::vector<std::string> outputLayers{output_layer_name};

  for(const auto& path : get_paths(png_main_dir_path)){
	  
	  auto img = readImageAndResize(path);
	  if(img.cols == 0 || img.rows == 0)
		return -1;

	  auto blob = cv::dnn::blobFromImage(img);

	  net.setInput(blob);
	  
	  std::vector<cv::Mat> output;
	  net.forward(output, outputLayers);
	  
	  if(output.size() != 1)
		return -1;

	  if(output[0].total() != 2)
		return -1;

	  std::cout << path << ' ' << output[0].at<float>(0) << " " << output[0].at<float>(1) <<'\n';
  }
}
