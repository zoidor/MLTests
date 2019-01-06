Step to install:

1. Download LibTorch https://pytorch.org/get-started/locally/ and unzip in the directory caffe2Program, so that caffe2Program/libtorch is created. 
2. Install MKL, I used the first Option in https://gist.github.com/pachamaltese/afc4faef2f191b533556f261a46b3aa8. I am copying the instruction for completeness

cd ~
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB

sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt-get update && sudo apt-get install intel-mkl-64bit

NOTE: I had to issue:

sudo apt-get install intel-mkl-64bit-2018.0-033

because intel-mkl-64bit is an alias for various versions

4. Find where MKL was installed searching for files like mkl_*.h, in my case it was in /opt/intel/compilers_and_libraries_2018.0.128/linux/mkl/include

5. Create a directory, e.g. make inside caffe2Program

cd make
cmake -DCMAKE_CXX_FLAGS="-I /opt/intel/compilers_and_libraries_2018.0.128/linux/mkl/include"  ..
make

6. The executable is available in  caffe2Program/bin

If there is an error with protobuf:

1. Install version 3.5.0 of protobuf, download from https://github.com/protocolbuffers/protobuf/tags the "all" version 
2. Unzip
3. Then

cd inside the directory
./configure
make 
make check
sudo make install

Then issue:
protoc --version 

4. In case of error of wrong version, issue

sudo ldconf
