To compile, copy this directory inside the tensorflow main directory. This directory should be on the same level the file WORKSPACE is.

We used the Tensorflow revision: 676ff175114a0642003e9b55be0173473e1e5140

To compile and run try, inside the segmentation directory

bazel run :malaria 


>>> from keras.models import load_model
>>> model = load_model("malaria_model.h5")

>>> model.input
<tf.Tensor 'input_1:0' shape=(?, 64, 64, 3) dtype=float32>

>>> model.output
<tf.Tensor 'activation_42/Softmax:0' shape=(?, 2) dtype=float32>

