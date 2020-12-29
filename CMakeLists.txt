cmake_minimum_required(VERSION 3.19)

project(3D-reconstruction CUDA CXX)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -gencode arch=compute_75,code=[sm_75,compute_75] -gencode arch=compute_86,code=[sm_86,compute_86]")
set(CUDA_VERBOSE_BUILD ON)

#set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Ofast  -fopenmp -fopenmp-simd -march=native -mtune=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast  -fopenmp -fopenmp-simd -march=native -mtune=native -std=gnu++17")

find_package(CUDA REQUIRED)

add_subdirectory(src)
