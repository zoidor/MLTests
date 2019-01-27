import onnxmltools
import keras.models
import os

import caffe2.python.onnx
from caffe2.python.onnx.backend import Caffe2Backend

def safe_mkdir(dirpath):
	if not os.path.exists(dirpath):
	    os.makedirs(dirpath)

output_dir = "convertedModel/"

safe_mkdir(output_dir)

keras_model = keras.models.load_model("../../kerasModel/malaria_model.h5")
onnx_model = onnxmltools.convert_keras(keras_model) 
onnxmltools.utils.save_model(onnx_model, output_dir + "test_model.onnx")

init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)

with open(output_dir + "init_net.pb", "wb") as f:
    f.write(init_net.SerializeToString())

with open(output_dir + "predict_net.pb", "wb") as f:
    f.write(predict_net.SerializeToString())

write_for_debug = False
if(write_for_debug):
	with open(output_dir + "onnx-predict.pbtxt", "w") as f:
	    f.write(str(predict_net))
