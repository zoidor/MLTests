To install Opencv follow the instructions from https://www.learnopencv.com/opencv-installation-on-ubuntu-macos-windows-and-raspberry-pi/

If you experience linking issues with LibTiff or ZLib add -D BUILD_TIFF=ON -D BUILD_ZLIB=ON to cmake as follows:
 
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D BUILD_TESTS=OFF -D OPENCV_ENABLE_NONFREE=ON -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF -D BUILD_C_EXAMPLES=OFF -D BUILD_TIFF=ON -D BUILD_ZLIB=ON ..

It is needed because ubuntu is now shipping with new versions of libtiff and zlib that are not compatbile with the versions required by opencv. -D BUILD_xxx makes Opencv build its own versions
