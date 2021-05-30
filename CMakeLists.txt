cmake_minimum_required(VERSION 3.16)
# https://codeyarns.com/2013/09/13/how-to-build-cuda-programs-using-cmake

set(EXECUTABLE softmax)
include(CheckLanguage)
if(${CMAKE_CUDA_COMPILER})
    message("Found CUDA at: ${CMAKE_CUDA_COMPILER}")
    project(${EXECUTABLE} LANGUAGES CXX CUDA)
    set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3 -gencode arch=compute_52,code=sm_52 -Xcompiler "-march=native -std=c++0x"
    )
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    message("CUDA compiler not found. Using c++ compiler with CPU libraries")
    project(${EXECUTABLE} LANGUAGES CXX)
endif()

SET(EXTRA_LIBS "")
SET(EXTRA_INCLUDE_DIRS "")

if(${CMAKE_CUDA_COMPILER})
    cuda_add_executable(${EXECUTABLE} ${SOURCES} ${HEADERS})
    set_target_properties(${EXECUTABLE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    add_executable(${EXECUTABLE} ${SOURCES} ${HEADERS})
endif()

add_subdirectory(src) # source files
include_directories(src) # header files
include(src/CMakeLists.headers.txt)

if(MSVC)
    add_compile_options(/W4)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++14")
else()
    set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11")
    add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
endif()

