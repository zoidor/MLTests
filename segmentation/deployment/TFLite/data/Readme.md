tflite_convert --keras_model_file=test_model.h5 --output_file=test_model.tflite


b'2019-01-06 17:49:49.945812: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1080] Converting unsupported operation: ResizeNearestNeighbor\n2019-01-06 17:49:49.945978: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1080] Converting unsupported operation: ResizeNearestNeighbor\n2019-01-06 17:49:49.946061: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1080] Converting unsupported operation: ResizeNearestNeighbor\n2019-01-06 17:49:49.946119: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1080] Converting unsupported operation: ResizeNearestNeighbor\n2019-01-06 17:49:49.946618: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before Removing unused ops: 95 operators, 139 arrays (0 quantized)\n2019-01-06 17:49:49.947156: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before general graph transformations: 95 operators, 139 arrays (0 quantized)\n2019-01-06 17:49:49.948617: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] After general graph transformations pass 1: 33 operators, 71 arrays (0 quantized)\n2019-01-06 17:49:49.948866: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before dequantization graph transformations: 33 operators, 71 arrays (0 quantized)\n2019-01-06 17:49:49.949170: I tensorflow/contrib/lite/toco/allocate_transient_arrays.cc:345] Total transient array allocated size: 28180480 bytes, theoretical optimal value: 28180480 bytes.\n2019-01-06 17:49:49.949361: F tensorflow/contrib/lite/toco/tflite/export.cc:386] Some of the operators in the model are not supported by the standard TensorFlow Lite runtime. If you have a custom implementation for them you can disable this error with --allow_custom_ops, or by setting allow_custom_ops=True when calling tf.contrib.lite.TFLiteConverter(). Here is a list of operators for which  you will need custom implementations: ResizeNearestNeighbor.\nAborted (core dumped)\n'
None
