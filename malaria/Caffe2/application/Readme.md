Step to install:

1. Install MKL, I used the first Option in https://gist.github.com/pachamaltese/afc4faef2f191b533556f261a46b3aa8. I am copying the instruction for completeness

cd ~
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB

sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt-get update && sudo apt-get install intel-mkl-64bit

NOTE: I had to issue:

sudo apt-get install intel-mkl-64bit-2018.0-033

because intel-mkl-64bit is an alias for various versions

2. 

mkdir make
cd make
cmake ..
make

3. The executable is available in bin directory inside the main directory
