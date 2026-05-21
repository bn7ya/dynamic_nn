cmake_minimum_required(VERSION 3.18)
project(elasticneuralnetwork CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

option(ENN_BUILD_TESTS "Build C++ unit tests" ON)
option(ENN_BUILD_BENCH "Build microbenchmarks" OFF)

file(GLOB_RECURSE ENN_CORE_SRC
    "${CMAKE_SOURCE_DIR}/csrc/controllers/*.cpp"
    "${CMAKE_SOURCE_DIR}/csrc/modules/*.cpp"
    "${CMAKE_SOURCE_DIR}/csrc/training/*.cpp"
    "${CMAKE_SOURCE_DIR}/csrc/ops/*.cpp"
)

add_library(enn_core STATIC ${ENN_CORE_SRC})
target_include_directories(enn_core PUBLIC ${CMAKE_SOURCE_DIR}/csrc/include)
target_link_libraries(enn_core PUBLIC ${TORCH_LIBRARIES})
set_property(TARGET enn_core PROPERTY CXX_STANDARD 17)

if(ENN_BUILD_TESTS)
    enable_testing()
    find_package(GTest)
    if(GTest_FOUND)
        add_subdirectory(tests/cpp)
    endif()
endif()

if(ENN_BUILD_BENCH)
    find_package(benchmark)
    if(benchmark_FOUND)
        add_subdirectory(microbench)
    endif()
endif()
