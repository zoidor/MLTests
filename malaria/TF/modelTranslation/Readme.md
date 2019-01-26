Conversion from keras to TF using mmconvert (https://github.com/Microsoft/MMdnn):


mmconvert --srcFramework keras --inputWeight ../../malaria_model.h5 --dstFramework tensorflow -om tf_model


The model is then saved inside the tf_model directory
