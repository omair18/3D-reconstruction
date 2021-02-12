set(DECODING_LIB_HEADERS
        IImageDecoder.h
        ImageDecoderFactory.h
        OpenCVImageDecoder.h
        NvidiaImageDecoder.h)

set(DECODING_LIB_CPPS
        ImageDecoderFactory.cpp
        OpenCVImageDecoder.cpp
        NvidiaImageDecoder.cpp
        )

source_group("Header Files" FILES ${DECODING_LIB_HEADERS})
source_group("Source Files" FILES ${DECODING_LIB_CPPS})

add_library(decoding SHARED ${DECODING_LIB_HEADERS} ${DECODING_LIB_CPPS})

target_link_libraries(decoding

        )

target_include_directories(decoding PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
