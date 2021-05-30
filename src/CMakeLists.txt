target_sources(${EXECUTABLE}
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/program.cu
    ${CMAKE_CURRENT_LIST_DIR}/softmax.cu
    ${CMAKE_CURRENT_LIST_DIR}/helper_cuda.cu
    ${CMAKE_CURRENT_LIST_DIR}/stdin_reader.cpp
)