#Adapted from https://github.com/bitbionic/keras-to-tensorflow/blob/master/k2tf_convert.py

from keras.models import load_model
from keras import backend as K

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

modelPath = "../../kerasModel/malaria_model.h5"

net_model = load_model(modelPath)

print("Keras input layer ", net_model.input.name)
print("Keras output layer ", net_model.output.name)

sess = K.get_session()

file_writer = tf.summary.FileWriter('logs', sess.graph)

#for some weird reason the final part of the name (i.e. :0 in this case) has to be removed when passed to convert_variables_to_constants
out_name = net_model.output.name.split(":")[0]
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [out_name])
graph_io.write_graph(constant_graph, "tf_model_dir", "malaria_model.pb", as_text = False)



