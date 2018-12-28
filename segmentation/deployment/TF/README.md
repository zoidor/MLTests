To compile, copy this directory inside the tensorflow main directory. This directory should be on the same level the file WORKSPACE is.

We used the Tensorflow revision: b37a329a6fc4aaf94767ebf5158689a3ee233b89

To compile and run try, inside the segmentation directory

bazel run :cell_segmentation 

The output is stored inside bazel-out, try to use 

find -name dumpSegmentation.txt

for retrieving the output 

find -name dumpInput.txt

to retrieve the input tensor for debugging purposes 


