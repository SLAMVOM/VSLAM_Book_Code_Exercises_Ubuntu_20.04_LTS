cmake_minimum_required( VERSION 3.0 )

project( useSophus )

# in order to use Sophus, should use find_package command to find it
find_package(Sophus REQUIRED)
include_directories( ${Sophus_INCLUDE_DIRS}$ )

# Eigen
include_directories("/usr/include/eigen3")
add_executable(useSophus useSophus.cpp)
target_link_libraries(useSophus Sophus::Sophus)

add_subdirectory(example)
