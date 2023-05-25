cmake_minimum_required(VERSION 3.25)
project(YOLOv8_TensorRT_CPP)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

find_package(OpenCV REQUIRED)

# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR /home/cyrus/work/libs/TensorRT-8.2.1.8/)

add_subdirectory(libs/tensorrt-cpp-api)

add_executable(YOLOv8_TensorRT_CPP main.cpp)

target_link_libraries(YOLOv8_TensorRT_CPP tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(YOLOv8_TensorRT_CPP PUBLIC libs/tensorrt-cpp-api/src)

