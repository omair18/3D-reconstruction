include_directories(${LIBRARIES_PATH}/aws/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(http_server_test http_server_test.cpp test_cuda_image_descriptor.cpp)
target_link_libraries(http_server_test Boost::system pthread ssl crypto)

add_executable(test test.cpp)
target_link_libraries(test Boost::json opencv_cudaarithm opencv_cudawarping opencv_core opencv_imgcodecs decoding datastructures log)

add_executable(test_json test_json.cpp)
target_link_libraries(test_json Boost::json)

add_executable(test_npp_resize test_npp_resize.cpp cuda_test_kernels.h cuda_test_kernels.cu)
target_link_libraries(test_npp_resize
        opencv_core
        opencv_cudaarithm
        opencv_cudawarping
        opencv_highgui
        ${CUDA_nppc_LIBRARY}
        ${CUDA_nppig_LIBRARY}
        stdc++fs
        opencv_imgcodecs
        ${CUDA_LIBRARIES})

add_executable(test_npp_convolution test_npp_convolution.cpp)
target_link_libraries(test_npp_convolution
        opencv_core
        opencv_cudaarithm
        opencv_cudaimgproc
        opencv_highgui
        stdc++fs
        opencv_imgcodecs
        ${CUDA_nppc_LIBRARY}
        ${CUDA_nppig_LIBRARY}
        ${CUDA_LIBRARIES})

add_executable(test_aws_bucket test_aws_bucket.cpp)
target_link_libraries(test_aws_bucket ${AWSSDK_LINK_LIBRARIES})

add_executable(test_scharr_filter test_scharr_filter.cpp)
target_link_libraries(test_scharr_filter
        ${CUDA_nppc_LIBRARY}
        ${CUDA_nppig_LIBRARY}
        opencv_core
        opencv_imgcodecs
        opencv_highgui
        stdc++fs
        opencv_imgproc)

add_executable(test_cuda_image_descriptor test_cuda_image_descriptor.cpp)
target_link_libraries(test_cuda_image_descriptor datastructures)