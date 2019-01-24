package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")

tf_cc_binary(
    name = "cell_segmentation",
    srcs = [
        "main.cc",
	"tiff_utils.cc",
        "tiff_utils.h",
    ],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
    ],

    linkopts = ["-lm -ltiff"],
  
     data = ["data/test_model_tensorflow.pb",
	   "data/image.tif"],
)


