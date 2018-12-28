load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "cell_segmentation",
    srcs = [
        "main.cc",
        "tiff_utils.cc",
        "tiff_utils.h",
    ],
    linkopts = select({
        "//tensorflow:android": [
            "-pie",
            "-landroid",
            "-ljnigraphics",
            "-llog",
            "-lm",
            "-z defs",
            "-s",
            "-Wl,--exclude-libs,ALL",
        ],
        "//conditions:default": ["-lm -ltiff"],
    }),
    deps = select({
        "//tensorflow:android": [
            # cc:cc_ops is used to include image ops (for label_image)
            # Jpg, gif, and png related code won't be included
            "//tensorflow/cc:cc_ops",
            "//tensorflow/core:android_tensorflow_lib",
            # cc:android_tensorflow_image_op is for including jpeg/gif/png
            # decoder to enable real-image evaluation on Android
            "//tensorflow/core/kernels:android_tensorflow_image_op",
        ],
        "//conditions:default": [
            "//tensorflow/cc:cc_ops",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:tensorflow",
        ],
    }),

   data = ["data/test_model_tensorflow.pb",
	   "data/image.tif"],
)

