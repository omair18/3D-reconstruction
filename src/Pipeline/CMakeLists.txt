set(PIPELINE_LIB_HEADERS
    ModelDataset.h)

set(PIPELINE_LIB_CPPS
    ModelDataset.cpp)

source_group("Header Files" FILES ${PIPELINE_LIB_HEADERS})
source_group("Source Files" FILES ${PIPELINE_LIB_CPPS})

add_library(pipeline SHARED ${PIPELINE_LIB_HEADERS} ${PIPELINE_LIB_CPPS})

target_link_libraries(pipeline

    )
