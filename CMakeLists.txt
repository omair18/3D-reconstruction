cmake_minimum_required(VERSION 3.19)

project(3D-reconstruction CUDA CXX)

set(LIBRARIES_PATH /var/3DReconstructionEnv)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS}
        -std=c++17
        -gencode arch=compute_61,code=[sm_61,compute_61]
        -gencode arch=compute_75,code=[sm_75,compute_75]
        -gencode arch=compute_80,code=[sm_80,compute_80]
        -gencode arch=compute_86,code=[sm_86,compute_86]")
set(CUDA_VERBOSE_BUILD ON)

#set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Ofast  -fopenmp -fopenmp-simd -march=native -mtune=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast  -fopenmp -fopenmp-simd -march=native -mtune=native -std=c++17")

find_package(CUDA REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${LIBRARIES_PATH}/librdkafka/lib/cmake/RdKafka")
find_package(RdKafka REQUIRED PATHS ${LIBRARIES_PATH}/librdkafka NO_DEFAULT_PATH)

find_package(Boost COMPONENTS
        log
        stacktrace_addr2line
        program_options
        system
        json
        REQUIRED
        PATHS ${LIBRARIES_PATH}/boost
        NO_DEFAULT_PATH)

find_package(OpenCV COMPONENTS
        opencv_core
        opencv_cudaarithm
        opencv_imgproc
        opencv_videoio
        opencv_cudaimgproc
        opencv_cudawarping
        opencv_sfm
        opencv_rgbd
        opencv_cudafeatures2d
        opencv_features2d
        opencv_xfeatures2d
        opencv_highgui
        opencv_imgcodecs
        REQUIRED
        PATHS ${LIBRARIES_PATH}/opencv
        NO_DEFAULT_PATH)

find_package(AWSSDK REQUIRED PATHS ${LIBRARIES_PATH}/aws NO_DEFAULT_PATH)

find_library(NVJPEG_LIBRARY nvjpeg ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

find_library(NVJPEG2K_LIBRARY nvjpeg2k ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

message("CXX compiler -- " ${CMAKE_CXX_COMPILER})
message("C compiler -- " ${CMAKE_C_COMPILER})
message("CUDA compiler -- " ${CMAKE_CUDA_COMPILER})

message("-- Found NvJPEG: " ${NVJPEG_LIBRARY})
message("-- Found NvJPEG2000: " ${NVJPEG2K_LIBRARY})

add_subdirectory(src)
add_subdirectory(test)

file(COPY web DESTINATION ${PROJECT_BINARY_DIR}/bin)
file(COPY python/kafka_dataset_sender.py DESTINATION ${PROJECT_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/bin/models)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/bin/Logs)