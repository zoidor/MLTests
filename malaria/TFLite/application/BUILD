package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")

tf_cc_binary(
    name = "malaria",
    srcs = [
        "main.cc",
	"png_helper.cc",
        "png_helper.h",
    ],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite:builtin_op_data",
	"@png_archive//:png",
    ],

    linkopts = ["-lm"],
  
     data = ["data"],
)


