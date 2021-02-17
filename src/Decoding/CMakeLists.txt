set(DECODING_LIB_HEADERS
        IImageDecoder.h
        ImageDecoderFactory.h
        OpenCVImageDecoder.h
        NvJPEGDecoder.h
        NvJPEG2kDecoder.h)

set(DECODING_LIB_CPPS
        ImageDecoderFactory.cpp
        OpenCVImageDecoder.cpp
        NvJPEGDecoder.cpp
        NvJPEG2kDecoder.cpp)

source_group("Header Files" FILES ${DECODING_LIB_HEADERS})
source_group("Source Files" FILES ${DECODING_LIB_CPPS})

add_library(decoding SHARED ${DECODING_LIB_HEADERS} ${DECODING_LIB_CPPS})

include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(decoding
        PUBLIC

        PRIVATE
        log
        ${NVJPEG_LIBRARY}
        ${CUDA_LIBRARIES}
        opencv_cudaarithm
        )

target_include_directories(decoding PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
