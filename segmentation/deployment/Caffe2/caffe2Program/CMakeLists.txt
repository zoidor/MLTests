cmake_minimum_required(VERSION 2.6)

project (caffe2_cpp_tutorial)

find_package(Protobuf REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Caffe2 REQUIRED PATHS libtorch)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

include_directories(include)
include_directories(libtorch/include)

set(PB_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/caffe2/util")
execute_process(
  COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --proto_path=../caffe2 --proto_path=${PB_SOURCE_DIR} --cpp_out=${PB_SOURCE_DIR} "${PB_SOURCE_DIR}/model.proto"
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(DYNAMIC_LIBS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")


file(GLOB LIB_SOURCES "${PROJECT_SOURCE_DIR}/src/caffe2/util/*.cc" "${PROJECT_SOURCE_DIR}/src/caffe2/operator/*.cc")
add_library(caffe2_cpp ${LIB_SOURCES})

#list(APPEND ALL_LIBRARIES ${PROTOBUF_LIBRARY})
#list(APPEND ALL_LIBRARIES ${GLOG_LIB} ${GFLAGS_LIB})
#list(APPEND ALL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

add_executable(main "src/caffe2/binaries/pretrained.cc")
target_link_libraries(main ${caffe2_cpp} caffe2_library c10 ${OpenCV_LIBS})

