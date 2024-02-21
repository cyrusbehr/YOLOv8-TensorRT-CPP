cmake_minimum_required(VERSION 3.18)
# Since CMake3.9, CUDA support can be enabled directly
project(YOLOv8_TensorRT_CPP LANGUAGES CXX CUDA)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 11)
if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")
endif()

# When using MSVC, set the encoding to UTF-8.
ADD_COMPILE_OPTIONS("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
ADD_COMPILE_OPTIONS("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# TODO: Specify the path to TensorRT root dir
if (NOT TensorRT_DIR)
    set(TensorRT_DIR E:/3rdparty/TensorRT-8.6.1.6)
endif()

# TODO: Specify the path to OpenCV root dir
if (NOT OpenCV_DIR)
    set(OpenCV_DIR E:/3rdparty/OpenCV/opencv470WithCuda/build)
endif()

# We require OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

# CMake Options
# option(ENABLE_BENCHMARKS "Benchmark the preprocessing, inference, and post processing" OFF)
# if (ENABLE_BENCHMARKS)
#     add_compile_definitions(ENABLE_BENCHMARKS)
# endif()

add_subdirectory(libs/tensorrt-cpp-api)
add_subdirectory(YOLOv8_TensorRT_CPP)
add_subdirectory(Demo)

# cmake_minimum_required(VERSION 3.25)
# project(YOLOv8_TensorRT_CPP)

# # Use ccache to speed up rebuilds
# include(cmake/ccache.cmake)

# # Set C++ version and optimization level
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# # CMake Options
# option(ENABLE_BENCHMARKS "Benchmark the preprocessing, inference, and post processing" OFF)
# if (ENABLE_BENCHMARKS)
#     add_compile_definitions(ENABLE_BENCHMARKS)
# endif()

# find_package(OpenCV REQUIRED)

# # TODO: Specify the path to TensorRT root dir
# set(TensorRT_DIR /home/cyrus/work/libs/TensorRT-8.6.1.6/)

# # Build the TensorRT inference engine library
# # TensorRT is found and linked by the tensorrt-cpp-api
# add_subdirectory(libs/tensorrt-cpp-api)

# # Build the YoloV8 library
# add_library(YoloV8_TRT SHARED src/yolov8.cpp)
# target_link_libraries(YoloV8_TRT PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
# target_include_directories(YoloV8_TRT PUBLIC libs/tensorrt-cpp-api/src)

# # Build and link the executables
# add_executable(detect_object_image src/object_detection_image.cpp)
# target_link_libraries(detect_object_image YoloV8_TRT)

# add_executable(benchmark src/benchmark.cpp)
# target_link_libraries(benchmark YoloV8_TRT)

# add_executable(detect_object_video src/object_detection_video_stream.cpp)
# target_link_libraries(detect_object_video YoloV8_TRT)