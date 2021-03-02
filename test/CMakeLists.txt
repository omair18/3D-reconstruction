add_executable(http_server_test http_server_test.cpp)
target_link_libraries(http_server_test Boost::system pthread ssl crypto)

add_executable(test test.cpp)
target_link_libraries(test Boost::json)

add_executable(test_json test_json.cpp)
target_link_libraries(test_json Boost::json)

add_executable(test_npp_resize test_npp_resize.cpp cuda_test_kernels.h cuda_test_kernels.cu)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(test_npp_resize
        opencv_core
        opencv_cudaarithm
        opencv_cudawarping
        opencv_highgui
        ${NPPC_LIBRARY}
        ${NPPIG_LIBRARY}
        stdc++fs
        opencv_imgcodecs
        ${CUDA_LIBRARIES})

add_executable(test_npp_convolution test_npp_convolution.cpp)
target_link_libraries(test_npp_convolution
        opencv_core
        opencv_cudaarithm
        opencv_cudaimgproc
        opencv_highgui
        ${NPPC_LIBRARY}
        ${NPPIF_LIBRARY}
        stdc++fs
        opencv_imgcodecs
        ${CUDA_LIBRARIES})