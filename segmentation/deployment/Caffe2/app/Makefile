OPENCV_LIB=$(shell dirname $$(find /usr/lib -name "*opencv_imgproc*" | head -n1))
OPENCV_INCLUDE=/usr/include

BUILD_DIR=build
INCLUDE = -I./$(BUILD_DIR)/lib/include -I./$(BUILD_DIR)/include -I$(OPENCV_INCLUDE) -I/opt/intel/compilers_and_libraries_2018.0.128/linux/mkl/include
LIBOPTS = -L./$(BUILD_DIR)/lib -L$(OPENCV_LIB) 
LDFLAGS :=  -lglog -lprotobuf -lcaffe2 -lgflags -lopencv_core -lopencv_highgui -lopencv_imgproc -lc10 -lopencv_imgcodecs
CFLAGS = -O3 -fpic -Wall -std=c++11
CC = gcc
CXX = g++
NB_THREADS = 8

CUDNN_URL = "http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"

.PHONY : all
all : run

dependency :
	sudo apt-get update
	sudo apt-get install -y --no-install-recommends \
	build-essential \
	cmake \
	git \
	libgoogle-glog-dev \
	libprotobuf-dev \
	protobuf-compiler \
	python-dev \
	python-pip
	sudo pip install numpy protobuf

build :
	rm -rf $(BUILD_DIR) && mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make -j$(NB_THREADS)


app : $(BUILD_DIR) 
	$(CXX) app.cc $(CFLAGS) $(INCLUDE) $(LIBOPTS) -o $@ $(LDFLAGS)

run : app
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:./$(BUILD_DIR)/lib ./app 

clean :
	rm -f *.o app

