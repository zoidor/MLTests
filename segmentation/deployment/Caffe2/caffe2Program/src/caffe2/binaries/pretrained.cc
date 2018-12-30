#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/utils/proto_utils.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>

std::string init_net_fname{"res/squeezenet_init_net.pb"};
std::string predict_net_fname{"res/squeezenet_predict_net.pb"};
std::string file{"res/image_file.jpg"};
std::string classes{ "res/imagenet_classes.txt"};
const int size = 227;

namespace caffe2 {

void run() {
  std::cout << std::endl;
  std::cout << "## Caffe2 Loading Pre-Trained Models Tutorial ##" << std::endl;
  std::cout << "https://caffe2.ai/docs/zoo.html" << std::endl;
  std::cout << "https://caffe2.ai/docs/tutorial-loading-pre-trained-models.html"
            << std::endl;
  std::cout << "https://caffe2.ai/docs/tutorial-image-pre-processing.html"
            << std::endl;
  std::cout << std::endl;

  
  auto image = cv::imread(file);  // CV_8UC3
  std::cout << "image size: " << image.size() << std::endl;

  // scale image to fit
  cv::Size scale(std::max(size * image.cols / image.rows, size),
                 std::max(size, size * image.rows / image.cols));
  cv::resize(image, image, scale);
  std::cout << "scaled size: " << image.size() << std::endl;

  // crop image to fit
  cv::Rect crop((image.cols - size) / 2, (image.rows - size) / 2,
                size, size);
  image = image(crop);
  std::cout << "cropped size: " << image.size() << std::endl;

  // convert to float, normalize to mean 128
  image.convertTo(image, CV_32FC3, 1.0, -128);
  std::cout << "value range: ("
            << *std::min_element((float *)image.datastart,
                                 (float *)image.dataend)
            << ", "
            << *std::max_element((float *)image.datastart,
                                 (float *)image.dataend)
            << ")" << std::endl;

  // convert NHWC to NCHW
  vector<cv::Mat> channels(3);
  cv::split(image, channels);
  std::vector<float> data;
  for (auto &c : channels) {
    data.insert(data.end(), (float *)c.datastart, (float *)c.dataend);
  }
  std::vector<std::int64_t> dims({1, image.channels(), image.rows, image.cols});

  auto tensor = caffe2::TensorCPUFromValues(dims, at::ArrayRef<decltype(data)::value_type>{data});
 
  // Load Squeezenet model
  NetDef init_net, predict_net;

  // >>> with open(path_to_INIT_NET) as f:
  CAFFE_ENFORCE(ReadProtoFromFile(init_net_fname, &init_net));

  // >>> with open(path_to_PREDICT_NET) as f:
  CAFFE_ENFORCE(ReadProtoFromFile(predict_net_fname, &predict_net));

  // >>> p = workspace.Predictor(init_net, predict_net)
  Workspace workspace("tmp");
  CAFFE_ENFORCE(workspace.RunNetOnce(init_net));
  auto input = workspace.CreateBlob("data")->GetMutable<TensorCPU>();
  input->ResizeLike(tensor);
  input->ShareData(tensor);
  CAFFE_ENFORCE(workspace.RunNetOnce(predict_net));

  // >>> results = p.run([img])
  auto &output_name = predict_net.external_output(0);
  auto output = workspace.GetBlob(output_name)->Get<TensorCPU>();

  // sort top results
  const auto &probs = output.data<float>();
  std::vector<std::pair<int, int>> pairs;
  for (auto i = 0; i < output.size(); i++) {
    if (probs[i] > 0.01) {
      pairs.push_back(std::make_pair(probs[i] * 100, i));
    }
  }

  std::sort(pairs.begin(), pairs.end());

  std::cout << std::endl;

  // read classes
  std::ifstream file(classes);
  std::string temp;
  std::vector<std::string> classes;
  while (std::getline(file, temp)) {
    classes.push_back(temp);
  }

  // show results
  std::cout << "output: " << std::endl;
  for (auto pair : pairs) {
    std::cout << "  " << pair.first << "% '" << classes[pair.second] << "' ("
              << pair.second << ")" << std::endl;
  }
}

}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
