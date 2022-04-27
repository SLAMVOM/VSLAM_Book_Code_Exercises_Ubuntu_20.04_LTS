# Notes about installation of common robotics packages on Ubuntu 20.04 LTS

#### Author: MT

#### Creation Date: 2022-April-18

#### Previous Edit: 2022-April-18



## Cmake

The Cmake package can be installed via calling:

`$ sudo apt-get install cmake`



## Eigen

The Eigen package can be easily installed by using the apt command in a command prompt as:

` $ sudo apt-get install libeigen3-dev`



## Pangolin

References:
- (GitHub) https://github.com/stevenlovegrove/Pangolin
- (Installation) https://cdmana.com/2021/02/20210204202321078t.html
- (Installation) https://blog.actorsfit.com/a?ID=00450-b67fc16a-5be8-4ea0-b72f-e5f9a36cc72a

#### Install some necessary dependencies:
`
 $ sudo apt-get install libglew-dev
 $ sudo apt-get install libboost-dev libboost-thread-dev libboost-filesystem-dev
 $ sudo apt-get install ffmpeg libavcodec-dev libavutil-dev libavformat-dev libswscale-dev
 $ sudo apt-get install libpng-dev
`

#### Or directly using the script provided by the Pangolin package
`
 $ cd ~/<your_fav_code_directory>
 $ git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
 $ cd Pangolin
 $ ./scripts/install_prerequisites.sh recommendeds
`

#### Configure and Build
`
 $ mkdir build && cd build
 $ cmake ..
 $ cmake --build .
 $ sudo make install
`



## fmt

The fmt package can be installed via running the follows:

`
 $ sudo add-apt-repository universe
 $ sudo apt update
 $ sudo apt install libfmt-dev
`



## Sophus
References:
- (Installation) https://chowdera.com/2021/06/20210602213411499x.html
- (GitHub) https://github.com/strasdat/Sophus

#### To install Sophus, one should compile from source
`
 $ cd ~/<your_fav_code_directory>
 $ git clone https://github.com/strasdat/Sophus.git
 $ cd Sophus
 $ mkdir build && cd build
 $ cmake ..
 $ make
 $ sudo make install
`



## OpenCV 3 or 4.5.5

#### Recommended:  **Install OpenCV through the source**. But before building from source, one will need to install the necessary dependencies. Note that different modules in OpenCV will require different dependencies, so the follows are only the ones needed for the VSLAM book.

Reference: https://tecadmin.net/how-to-install-opencv-on-ubuntu-20-04/ and https://linuxize.com/post/how-to-install-opencv-on-ubuntu-20-04/ and https://vitux.com/opencv_ubuntu/

- #### Install dependencies for VSLAM book on Ubuntu 20.04:

`
 $ sudo apt-get install build-essential libgtk2.0-dev libvtk7-dev libjpeg-dev libtiff-dev libopenexr-dev libtbb-dev
`

- #### (Alternatively) One may install the full set of dependencies on Ubuntu 20.04:
`
$ sudo apt install build-essential cmake git pkg-config libpng-dev libtiff-dev gfortran openexr libgtk-3-dev libavcodec-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev libopenexr-dev 
`

- #### Download the corresponding Zip file from the official OpenCV downloads site (recommend using the latest version): https://opencv.org/releases/

- #### Move and Unzip the compressed file into the <opencv_package_folder>

- If one needs to install the **contributed** functionality, the extra modules can be downloaded from: https://github.com/opencv/opencv_contrib . Then extract the zip files into the same <opencv_package_folder>

- Once the download is complete, perform the follows:

`
 $ cd ~/<opencv_package_folder>/opencv
 $ mkdir -p build && cd build
 $ cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/<change_here_to_opencv_package_folder>/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..
 $ make -j4
 $ sudo make install
`



## Ceres

Reference: http://ceres-solver.org/installation.html#linux

To install Ceres on Ubuntu 20.04 LTS, one can follow the below steps:

#### Install the dependencies
`
 $ sudo apt-get install libgoogle-glog-dev libgflags-dev
 $ sudo apt-get install libatlas-base-dev
`
#### (Optional) SuiteSparse and CXSparse
`
 $ sudo apt-get install libsuitesparse-dev # SuiteSparse and CXSparse (optional)
 $ sudo apt-get install libgtest-dev
`
#### Download and install from source
`
 $ cd ~/<Ceres_directory>
 $ git clone https://github.com/ceres-solver/ceres-solver.git
 $ mkdir build && cd build
 $ cmake ..
 $ make -j3
 $ make test
 $ sudo make install
`



## G2O

References:

- (GitHub) https://github.com/RainerKuemmerle/g2o#requirements
- (Installation) http://luohanjie.com/2018-08-09/installing-g2o-on-ubuntu.html

To install G2O on Ubuntu 20.04 LTS:

`
 $ sudo apt−get install qt5−qmake qt5−default libqglviewer−dev−qt5 libsuitesparse−dev
libcxsparse3 libcholmod3 qtdeclarative5-dev
 $ git clone https://github.com/RainerKuemmerle/g2o.git
 $ cd g2o
 $ mkdir build && cd build
 $ cmake ..
 $ make
 $ sudo make install
`



## Meshlab

The Meshlab can be installed via running:

`$ sudo apt-get install -y meshlab`



## DBoW3

Reference: (GitHub) https://github.com/rmsalinas/DBow3

To install DBoW3 on Ubuntu 20.04 LTS:

`
 $ git clone https://github.com/rmsalinas/DBow3.git
 $ cd DBoW3
 $ mkdir build && cd build
 $ cmake ..
 $ make
 $ sudo make install
`



## PCL

References: (Official Site) https://pointclouds.org/downloads/ and (GitHub) https://pcl.readthedocs.io/projects/tutorials/en/latest/compiling_pcl_posix.html

The PCL library can be installed via apt or build from source. To install via apt:

`
 $ sudo apt install libpcl-dev pcl-tools
`



## Octomap and Octomap-tools

To install Octomap and Octomap-tools:

`
 $ sudo apt-get install liboctomap-dev
 $ sudo apt-get install octomap-tools
`



## Octovis

To install Octovis, which is a tool to visualize octomaps:

`
 $ sudo apt-get install octovis
`



## Glut

To install Glut library:

`
 $ sudo apt-get install freeglut3-dev
`