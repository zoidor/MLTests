mmconvert -sf keras -iw ../../../learning/test_model.h5 -df onnx -om test_model.onnx

convert-onnx-to-caffe2 test_model.onnx --output test_model.pb --init-net-output init_net.pb

Warning:

OnnxEmitter has not supported operator [UpSampling2D].
up_sampling2d_1
OnnxEmitter has not supported operator [UpSampling2D].
up_sampling2d_2
OnnxEmitter has not supported operator [UpSampling2D].
up_sampling2d_3
OnnxEmitter has not supported operator [UpSampling2D].
up_sampling2d_4
OnnxEmitter has not supported operator [Sigmoid].
conv2d_12_activation


When trying to feed the model to Caffe2 I get:

 what():  [enforce fail at net.cc:69] . op Concat: Source for input up_sampling2d_1 is unknown for net mmdnn, operator input: "conv2d_6_activation" input: "up_sampling2d_1" output: "concatenate_1" output: "OC2_DUMMY_0" name: "" type: "Concat" arg { name: "axis" i: 1 }

