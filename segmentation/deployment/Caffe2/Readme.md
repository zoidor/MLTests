Status:
- Compilation OK, using the weird Makefile that runs the CMake, taken from https://github.com/jhjin/caffe2-cpp .
Tried to use libtorch instead of compiling from scratch, but if I do it, at runtime, the generation of some operators fails. 

With this the generation is successfull as far as I can tell.

- Translation of the model: FAILURE

Tried 
- Keras-Caffe-Caffe2 (modelTranslations/caffeModel)
- Keras-Caffe2 (modelTranslations/caffe2Model)
- Keras-ONNX-Caffe2 (modelTranslations/onnx)

All three cases are failing because of different issues regarding Upsampling2D . The "Onnx" version seems to currently  the most promising one, but it looks like that:
keras->onnx uses a version of onnx
onnx->caffe2 uses another version

