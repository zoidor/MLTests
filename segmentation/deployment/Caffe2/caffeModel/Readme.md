Conversion tentative: KERAS -> Caffe -> Caffe2

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#inspecting-graphs

bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=xxxx.pb


Found 1 possible inputs: (name=input_1, type=float(1), shape=[?,512,688,1]) 
No variables spotted.
Found 1 possible outputs: (name=conv2d_12/Sigmoid, op=Sigmoid) 
Found 438536 (438.54k) const parameters, 0 (0) variable parameters, and 0 control_edges
Op types used: 43 Const, 24 Identity, 12 BiasAdd, 12 Conv2D, 11 Relu, 4 MaxPool, 4 Mul, 4 ResizeNearestNeighbor, 4 Shape, 4 StridedSlice, 3 ConcatV2, 1 Placeholder, 1 Sigmoid
To use with tensorflow/tools/benchmark:benchmark_model try these arguments:
bazel run tensorflow/tools/benchmark:benchmark_model -- --graph=$HOME/MLTests/segmentation/deployment/Caffe2/test_model_tensorflow.pb --show_flops --input_layer=input_1 --input_layer_type=float --input_layer_shape=-1,512,688,1 --output_layer=conv2d_12/Sigmoid

---------------------------------------------------------------------------------------------------------

mmconvert  --srcFramework keras --dstFramework caffe  -om test_model -iw ../test_model.h5

after installing mmconvert from master using pip3.




pip3 install -U  git+https://github.com/Microsoft/MMdnn.git@master


pip3 install caffe
pip3 install tensorflow
pip3 install keras

Note sudo-apt get install caffe-cpu installs only the python3 version and not the python2.7 version
 


Converting from Keras -> Caffe OK!

Converting from Caffe -> Caffe2 FAILURE! 
When trying to use https://caffe2.ai/docs/caffe-migration.html#caffe-to-caffe2 I get:

.......
INFO:caffe_translator:Translate layer up_sampling2d_1
.......
NotImplementedError: Translator currently does not support group deconvolution.

