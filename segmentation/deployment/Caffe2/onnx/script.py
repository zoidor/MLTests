import onnxmltools
import keras.models

import onnx
from onnx_caffe2.backend import Caffe2Backend

keras_model = keras.models.load_model("../../../learning/test_model.h5")
onnx_model = onnxmltools.convert_keras(keras_model) 
onnxmltools.utils.save_model(onnx_model, 'test_model.onnx')

init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model.graph)
with open("init_net.pb", "wb") as f:
    f.write(init_net.SerializeToString())

with open("predict_net.pb", "wb") as f:
    f.write(predict_net.SerializeToString())

