#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/utils/proto_utils.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>

std::string init_net_fname{"data/init_net.pb"};
std::string predict_net_fname{"data/test_model.pb"};

std::string file{"data/image.tif"};

const int cropped_input_height = 512;
const int cropped_input_width = 688;

namespace {

int run() {

  using namespace caffe2;
  std::cout << std::endl;
  std::cout << "## Caffe2 Loading Pre-Trained Models Tutorial ##" << std::endl;
  std::cout << "https://caffe2.ai/docs/zoo.html" << std::endl;
  std::cout << "https://caffe2.ai/docs/tutorial-loading-pre-trained-models.html"
            << std::endl;
  std::cout << "https://caffe2.ai/docs/tutorial-image-pre-processing.html"
            << std::endl;
  std::cout << std::endl;

  
  cv::Mat image = cv::imread(file, CV_LOAD_IMAGE_GRAYSCALE); 
  std::cout << "image size: " << image.size() << std::endl;

  if(image.cols == 0 || image.rows == 0) {
	std::cout << "unable to read image: " << file << '\n';
	return -1;
  }

  // crop image to fit
  cv::Rect crop(0, 0, std::min(cropped_input_width, image.cols), std::min(cropped_input_height, image.rows));
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

  std::vector<std::int64_t> dims({1, image.rows, image.cols, image.channels()});

  auto tensor = caffe2::TensorCPUFromValues(dims, at::ArrayRef<decltype(data)::value_type>{data});
 
  // Load Squeezenet model
  NetDef init_net, predict_net;

  CAFFE_ENFORCE(ReadProtoFromFile(init_net_fname, &init_net));
  CAFFE_ENFORCE(ReadProtoFromFile(predict_net_fname, &predict_net));

  Workspace workspace("default");
  CAFFE_ENFORCE(workspace.RunNetOnce(init_net));
   
  const char *  inputLayerName = "input_1_0";

  if(!workspace.HasBlob(inputLayerName)){
	std::cout << "Unable to load input layer " << inputLayerName <<'\n';
	return -1;
  }

  auto input = workspace.CreateBlob(inputLayerName)->GetMutable<TensorCPU>();

  input->ResizeLike(tensor);
  input->ShareData(tensor);
  

  CAFFE_ENFORCE(workspace.RunNetOnce(predict_net));

  auto &output_name = predict_net.external_output(0);
  auto output = workspace.GetBlob(output_name)->Get<TensorCPU>();

  if(static_cast<std::size_t>(output.numel()) != img_num_pixels){
	std::cout<<"Input and Output tensors must have the same dimensions\n";
	return -1;
   }

  const auto &probs = output.data<float>();
  
  std::size_t i = 0;
  for(int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x){
	for(int channel = 0; channel < image.channels(); ++channel){
		image.at<float>(y, x, channel) = probs[i];
		++i;
	}
    }
  }

  return 0;
}

}  // anonymous namespace

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  auto success = run();
  return success;
}
