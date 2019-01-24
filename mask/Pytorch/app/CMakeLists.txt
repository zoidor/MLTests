cmake_minimum_required(VERSION 2.8.3)

project(cpp_pytorch_segmentation)
include(ExternalProject)

find_package(Protobuf REQUIRED)
find_package(OpenCV REQUIRED)

ExternalProject_Add(libTorch
  PREFIX ext
  GIT_REPOSITORY http://github.com/pytorch/pytorch.git
  GIT_TAG b6a8c45f57b65d11894c4a6e5a3267708ecec1c5
  INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}

  CMAKE_ARGS    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
		-DBUILD_TORCH=ON
                -DUSE_CUDA=OFF
                -DUSE_OPENMP=ON
                -DUSE_LEVELDB=OFF
                -DUSE_LMDB=OFF
                -DUSE_NATIVE_ARCH=ON
                -DUSE_METAL=OFF
                -DBUILD_CUSTOM_PROTOBUF=ON
)

include_directories("${CMAKE_CURRENT_BINARY_DIR}/lib/include" "${CMAKE_CURRENT_BINARY_DIR}/include")
link_directories("${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(DYNAMIC_LIBS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(main "app.cc")
target_link_libraries(main torch c10 caffe2 glog  ${OpenCV_LIBS} ${PROTOBUF_LIBRARY})
#skip of errors needed to compile caffe2 files, like the header for intrusive_ptr
target_compile_options(main PRIVATE -Werror -Wall -Wextra -Wno-unused-parameter)

