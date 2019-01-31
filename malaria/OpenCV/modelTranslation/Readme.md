Use the same instructions used for TensorFlow but we need to set 

K.set_learning_phase(0)

see http://answers.opencv.org/question/204121/keras-densenet121-breaks-on-opencv-dnn/

Once done this we have another problem due to the "Shape" layers, hence follow the instructions in 

http://answers.opencv.org/question/183507/opencv-dnn-import-error-for-keras-pretrained-vgg16-model/

Summarized here for your convenience (note I was unable to get the optimization/transformation part to work)


Open tf_model_dir/malaria_model.pbtxt and remove nodes with names flatten/Shape, flatten/strided_slice, flatten/Prod, flatten/stack. Replace the nodes similar to

node {
  name: "flatten/Reshape"
  op: "Reshape"
  input: "block5_pool/MaxPool"
  input: "flatten/stack"
}

with

node {
  name: "flatten/Reshape"
  op: "Flatten"
  input: "block5_pool/MaxPool"
}


