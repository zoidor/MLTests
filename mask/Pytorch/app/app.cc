#include <torch/script.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>

std::string model_path{"../data/test_model.pt"};
std::string image_path{"../data/image.tif"};
std::string out_image_path{"../data/image_out.tif"};

const int cropped_input_height = 512;
const int cropped_input_width = 688;

int save_tensor(const at::Tensor& tensor, const std::string& out_path){

  const auto &probs = tensor.data<float>();
  
  cv::Mat image{cropped_input_height, cropped_input_width, CV_32FC1};
  std::size_t i = 0;
  for(int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x){
	image.at<float>(y, x) = probs[i];
	++i;
    }
  }

  cv::imwrite(out_path, image);
  return 0;
}

std::vector<float> read_image(const std::string& path, const int crop_h, const int crop_w){

  cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE); 
  std::cout << "image size: " << image.size() << std::endl;

  if(image.cols == 0 || image.rows == 0) {
	std::cout << "unable to read image: " << path << '\n';
	return {};
  }

  // crop image to fit
  cv::Rect crop(0, 0, std::min(crop_w, image.cols), std::min(crop_h, image.rows));
  image = image(crop);
  std::cout << "cropped size: " << image.size() << std::endl;

  //cast image to float
  image.convertTo(image, CV_32FC3);
  std::cout << "value range: ("
            << *std::min_element((float *)image.datastart,
                                 (float *)image.dataend)
            << ", "
            << *std::max_element((float *)image.datastart,
                                 (float *)image.dataend)
            << ")" << std::endl;
  
  std::cout << "Channels: " << image.channels() << '\n';

  // convert image to the expected input format I_R(0,0), I_G(0,0), I_B(0,0), I_R(1,0) .....

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

  return data;
}

int main(int argc, const char* argv[]) {
  
  auto module = torch::jit::load(model_path);

  if(module == nullptr){
  	std::cout << "Unable to load model\n";
	return -1;
  }

  auto data = read_image(image_path, cropped_input_height, cropped_input_width);

  if(data.empty()) {
  	std::cout << "Unable to load image\n";
	return -1;
  }

  auto dataTensor = torch::from_blob(data.data(), data.size(), at::TensorOptions{at::ScalarType::Float});
  std::vector<torch::jit::IValue> inputs{std::move(dataTensor)};

  // Execute the model and turn its output into a tensor.
  auto output = module->forward(inputs).toTensor();
  save_tensor(output, out_image_path);
}
