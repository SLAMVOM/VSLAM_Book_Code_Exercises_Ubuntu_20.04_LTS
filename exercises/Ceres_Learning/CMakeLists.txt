cmake_minimum_required(VERSION 2.8)

project(ceresLearn)

set(CMAKE_CXX_FLAGS "-std=c++14 -O2")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Eigen
include_directories("/usr/include/eigen3")

add_executable(linearFunc linearFunc.cpp)
target_link_libraries(linearFunc ${CERES_LIBRARIES})

add_executable(linearFuncNumeric linearFunc_numeric.cpp)
target_link_libraries(linearFuncNumeric ${CERES_LIBRARIES})

add_executable(linearFuncAnalytic linearFunc_analytic.cpp)
target_link_libraries(linearFuncAnalytic ${CERES_LIBRARIES})

add_executable(powellAuto powellAuto.cpp)
target_link_libraries(powellAuto ${CERES_LIBRARIES})

add_executable(curveFitAuto curveFittingAuto.cpp)
target_link_libraries(curveFitAuto ${CERES_LIBRARIES})

add_executable(robustCurveFitAuto robustCurveFitAuto.cpp)
target_link_libraries(robustCurveFitAuto ${CERES_LIBRARIES})