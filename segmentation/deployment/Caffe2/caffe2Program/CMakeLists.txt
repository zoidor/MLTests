cmake_minimum_required(VERSION 2.6)

project (caffe2_cpp_tutorial)

find_package(Protobuf REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Caffe2 REQUIRED PATHS libtorch)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

include_directories(include)
include_directories(libtorch/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(DYNAMIC_LIBS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(main "src/pretrained.cc")
target_link_libraries(main caffe2_library c10 ${OpenCV_LIBS} ${PROTOBUF_LIBRARY})

