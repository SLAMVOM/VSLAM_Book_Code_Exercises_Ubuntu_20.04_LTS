option(USE_UBUNTU_20 "Set to ON if you are using Ubuntu 20.04" ON)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++14")

find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

if(USE_UBUNTU_20)
	message("You are using Ubuntu 20.04, fmt::fmt will be linked")
	find_package(fmt REQUIRED)
	set(FMT_LIBRARIES fmt::fmt)
endif()


include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(stereoVision stereoVision.cpp)
target_link_libraries( stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${FMT_LIBRARIES} fmt)

