cmake_minimum_required(VERSION 2.8)

project(imageBasics)

# using C++11 standard support
set( CMAKE_CXX_FLAGS "-std=c++11" )

#find OpenCV
find_package( OpenCV REQUIRED )
# include OpenCV's headers
include_directories( ${OpenCV_INCLUDE_DIRS} )

# find eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable( imageBasics imageBasics.cpp )

# link the exe to OpenCV's libs
target_link_libraries( imageBasics ${OpenCV_LIBS} )

add_executable( undistortImage undistortImage.cpp )
# link the eve to OpenCV's libs
target_link_libraries( undistortImage ${OpenCV_LIBS} )
