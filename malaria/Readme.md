The .h5 file was generated using the instructions in:
https://www.pyimagesearch.com/2018/12/03/deep-learning-and-medical-image-analysis-with-keras/

NOTE: some of the files, despite various download attempts, would geneated an EOF error when being read and resized by the generator. Hence, the build_dataset.py was modified to filter out the 12 images that were causing the error. The error comes probably from the version of PIL used. Most likely the author of the cited article was using a different version of PIL. For more info, refer to kerasModel/build_dataset.py .
 
At the moment of writing the conversion from Keras to frameworks supporting inference in C++ has been successfull only for Tensorflow, OpenCV and TensorflowLite. As far as I can tell, conversion from Keras to Caffe2 using ONNX is not easy and it is a lossy process.
