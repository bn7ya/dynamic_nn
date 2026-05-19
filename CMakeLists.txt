cmake_minimum_required(VERSION 3.16)
project(DynamicNN VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(DNN_BUILD_TESTS "Build tests" ON)
option(DNN_BUILD_PYTHON "Build Python bindings" ON)
option(DNN_BUILD_EXAMPLES "Build examples" ON)
option(DNN_ENABLE_AVX "Enable AVX optimizations" ON)
option(DNN_ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(DNN_ENABLE_AVX512 "Enable AVX512 optimizations" OFF)
option(DNN_ENABLE_CUDA "Enable CUDA GPU acceleration" ON)

# CUDA Configuration
if(DNN_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)

        # Set CUDA standard
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        # Compute capabilities: Turing(75), Ampere(80,86), Ada(89), Blackwell(100)
        set(DNN_CUDA_ARCHITECTURES "75;80;86;89;100" CACHE STRING "CUDA architectures to compile for")
        set(CMAKE_CUDA_ARCHITECTURES ${DNN_CUDA_ARCHITECTURES})

        # CUDA compiler flags
        if(MSVC)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/W3")
            set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler=/O2")
        else()
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall")
            set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
        endif()

        # Optional fast math
        option(DNN_CUDA_FAST_MATH "Enable CUDA fast math" OFF)
        if(DNN_CUDA_FAST_MATH)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
        endif()

        set(DNN_HAS_CUDA TRUE)
        message(STATUS "CUDA enabled with architectures: ${DNN_CUDA_ARCHITECTURES}")
        message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
    else()
        message(STATUS "CUDA compiler not found, CUDA support disabled")
        set(DNN_HAS_CUDA FALSE)
    endif()
else()
    set(DNN_HAS_CUDA FALSE)
    message(STATUS "CUDA support disabled by user")
endif()

# Detect architecture
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(DNN_ARCH "x64")
else()
    set(DNN_ARCH "x86")
endif()

message(STATUS "Building for architecture: ${DNN_ARCH}")

# Compiler flags (use generator expressions to avoid passing C++ flags to CUDA compiler)
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    )
    if(DNN_ENABLE_AVX2)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    elseif(DNN_ENABLE_AVX)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
    endif()
    # Optimization flags for Release
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot /GL")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    )
    if(DNN_ENABLE_AVX512)
        add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512f>
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512dq>
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512vl>
        )
    elseif(DNN_ENABLE_AVX2)
        add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
            $<$<COMPILE_LANGUAGE:CXX>:-mfma>
        )
    elseif(DNN_ENABLE_AVX)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mavx>)
    endif()
    # Optimization flags for Release
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -flto")
endif()

# Source files
set(DNN_CORE_SOURCES
    src/core/tensor.cpp
    src/core/random.cpp
    src/core/initializers.cpp
    src/core/activations.cpp
    src/core/node.cpp
    src/core/layer.cpp
    src/core/network.cpp
)

set(DNN_MEMORY_SOURCES
    src/memory/pool_allocator.cpp
    src/memory/resource_monitor.cpp
)

set(DNN_SIMD_SOURCES
    src/simd/simd_detect.cpp
    src/simd/simd_ops.cpp
)

set(DNN_TRAINING_SOURCES
    src/training/cost_functions.cpp
    src/training/batch_manager.cpp
    src/training/early_stopping.cpp
    src/training/optimizer.cpp
    src/training/trainer.cpp
    src/training/runtime/runtime_compile_check.cpp
)

set(DNN_DYNAMICS_SOURCES
    src/dynamics/node_efficiency.cpp
    src/dynamics/layer_efficiency.cpp
    src/dynamics/layer_manager.cpp
    src/dynamics/health_monitor.cpp
    src/dynamics/trainable_scheduler.cpp
)

set(DNN_CQRS_SOURCES
    src/cqrs/command_bus.cpp
    src/cqrs/query_bus.cpp
)

set(DNN_IO_SOURCES
    src/io/input_adapter.cpp
    src/io/model_serializer.cpp
)

# Transformer ops (CPU). Powers pydnn.transformer's optional native
# backend. Header at include/dnn/transformer/ops.hpp. The CUDA twin
# lives in src/cuda/transformer_ops.cu (only built when CUDA is on);
# src/transformer/cuda_stubs.cpp supplies the *_cuda symbols when CUDA
# is off so transformer code can link unchanged in either config.
set(DNN_TRANSFORMER_SOURCES
    src/transformer/cpu_ops.cpp
)
if(NOT DNN_HAS_CUDA)
    list(APPEND DNN_TRANSFORMER_SOURCES src/transformer/cuda_stubs.cpp)
endif()

# CUDA sources (only if CUDA is available)
if(DNN_HAS_CUDA)
    set(DNN_CUDA_SOURCES
        src/cuda/cuda_runtime.cu
        src/cuda/cuda_blas.cu
        src/cuda/cuda_elementwise.cu
        src/cuda/cuda_activations.cu
        src/cuda/cuda_reductions.cu
        src/cuda/cuda_memory_pool.cu
        src/cuda/cuda_optimizers.cu
        src/cuda/transformer_ops.cu
    )
    set_source_files_properties(${DNN_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

# Core library
if(DNN_HAS_CUDA)
    add_library(dnn_core STATIC
        ${DNN_CORE_SOURCES}
        ${DNN_MEMORY_SOURCES}
        ${DNN_SIMD_SOURCES}
        ${DNN_TRAINING_SOURCES}
        ${DNN_DYNAMICS_SOURCES}
        ${DNN_CQRS_SOURCES}
        ${DNN_IO_SOURCES}
        ${DNN_TRANSFORMER_SOURCES}
        ${DNN_CUDA_SOURCES}
    )
else()
    add_library(dnn_core STATIC
        ${DNN_CORE_SOURCES}
        ${DNN_MEMORY_SOURCES}
        ${DNN_SIMD_SOURCES}
        ${DNN_TRAINING_SOURCES}
        ${DNN_DYNAMICS_SOURCES}
        ${DNN_CQRS_SOURCES}
        ${DNN_IO_SOURCES}
        ${DNN_TRANSFORMER_SOURCES}
    )
endif()

target_include_directories(dnn_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# dnn_core is a static archive that gets linked into the _dnn_core
# Python shared module, so its translation units must be position-
# independent. Without this, throw-statements in cuda_stubs.cpp emit
# absolute relocations against std::runtime_error's typeinfo and the
# shared-module link fails.
set_target_properties(dnn_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Add compile definitions for SIMD detection
target_compile_definitions(dnn_core PRIVATE
    $<$<BOOL:${DNN_ENABLE_AVX}>:DNN_ENABLE_AVX>
    $<$<BOOL:${DNN_ENABLE_AVX2}>:DNN_ENABLE_AVX2>
    $<$<BOOL:${DNN_ENABLE_AVX512}>:DNN_ENABLE_AVX512>
)

# Windows: disable min/max macros that conflict with std::min/std::max
if(MSVC)
    target_compile_definitions(dnn_core PUBLIC NOMINMAX)
endif()

# CUDA compile definitions and linking
if(DNN_HAS_CUDA)
    target_compile_definitions(dnn_core PUBLIC DNN_ENABLE_CUDA)
    target_include_directories(dnn_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(dnn_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Thread support
find_package(Threads REQUIRED)
target_link_libraries(dnn_core PUBLIC Threads::Threads)

# OpenMP for the CPU SIMD / batch loops. Optional: if not found, the
# #pragma omp directives below degrade silently to serial loops.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dnn_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(dnn_core PUBLIC DNN_HAS_OPENMP=1)
    message(STATUS "OpenMP enabled for CPU parallelism")
else()
    message(STATUS "OpenMP not found; CPU loops will run serially")
endif()

# Python bindings
if(DNN_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development)
    if(Python3_FOUND)
        # Try to find pybind11
        find_package(pybind11 CONFIG QUIET)
        if(pybind11_FOUND)
            pybind11_add_module(_dnn_core
                python/pydnn/_bindings.cpp
                python/pydnn/_bindings_transformer.cpp
            )
            target_link_libraries(_dnn_core PRIVATE dnn_core)

            # Install Python module
            install(TARGETS _dnn_core
                LIBRARY DESTINATION python/pydnn
            )
        else()
            message(STATUS "pybind11 not found, Python bindings disabled")
        endif()
    else()
        message(STATUS "Python not found, Python bindings disabled")
    endif()
endif()

# Tests
if(DNN_BUILD_TESTS)
    enable_testing()

    # Prefer a system GoogleTest if present; otherwise fetch a pinned release.
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.12.1
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(dnn_tests
        tests/cpp/test_tensor.cpp
        tests/cpp/test_network.cpp
        tests/cpp/test_simd.cpp
    )
    target_link_libraries(dnn_tests PRIVATE dnn_core GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(dnn_tests)
endif()

# Examples
if(DNN_BUILD_EXAMPLES)
    add_executable(basic_example examples/cpp/basic_network.cpp)
    target_link_libraries(basic_example PRIVATE dnn_core)
endif()

# Installation
install(TARGETS dnn_core
    EXPORT DynamicNNTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT DynamicNNTargets
    FILE DynamicNNTargets.cmake
    NAMESPACE DNN::
    DESTINATION lib/cmake/DynamicNN
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DynamicNNConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DynamicNNConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/DynamicNNConfig.cmake"
    INSTALL_DESTINATION lib/cmake/DynamicNN
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/DynamicNNConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/DynamicNNConfigVersion.cmake"
    DESTINATION lib/cmake/DynamicNN
)
