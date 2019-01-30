cmake_minimum_required(VERSION 2.8.3)

project(cpp_opencv_malaria)

#find_package(Protobuf REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OPENCV_EXTRA_MODULES_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(DYNAMIC_LIBS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(main "main.cc")
target_link_libraries(main ${OpenCV_LIBS} ${PROTOBUF_LIBRARY})
target_compile_options(main PRIVATE -Werror -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare)

