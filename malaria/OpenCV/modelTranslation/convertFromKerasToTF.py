#Adapted from https://github.com/bitbionic/keras-to-tensorflow/blob/master/k2tf_convert.py

from keras.models import load_model
from keras import backend as K

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

K.set_learning_phase(0)  

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

# Remove Const nodes.
for i in reversed(range(len(constant_graph.node))):
    if constant_graph.node[i].op == 'Const':
        del constant_graph.node[i]
    for attr in ['T', 'data_format', 'Tshape', 'N', 'Tidx', 'Tdim',
                 'use_cudnn_on_gpu', 'Index', 'Tperm', 'is_training',
                 'Tpaddings']:
        if attr in constant_graph.node[i].attr:
            del constant_graph.node[i].attr[attr]

# Save as text.
graph_io.write_graph(constant_graph, "tf_model_dir", "malaria_model.pbtxt", as_text=True)

