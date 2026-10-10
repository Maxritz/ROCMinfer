# ROCMinfer build definition.
# ROCMinfer is an independent, from-scratch AMD-first inference engine. See AGENTS.md
# for project rules and docs/PERF_TRACE.md for the instrumentation contract.

cmake_minimum_required(VERSION 3.21)

project(rocminfer
    VERSION 0.1.0
    DESCRIPTION "AMD-first LLM inference engine"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# HIP / ROCm
# ---------------------------------------------------------------------------
# Set ROCMINFER_ENABLE_HIP=ON to compile the device backend and the HIP event
# pool used for per-component device timing. When OFF the engine builds as a
# host-only binary whose reports carry no device column content.
option(ROCMINFER_ENABLE_HIP "Compile the HIP device backend" OFF)

if(ROCMINFER_ENABLE_HIP)
    enable_language(HIP)
    find_package(hip REQUIRED)
    set(CMAKE_HIP_ARCHITECTURES "${ROCMINFER_HIP_ARCHITECTURES}" CACHE STRING
        "Comma-separated gfx targets, e.g. gfx1031;gfx1201")
    add_compile_definitions(ROCMINFER_HAVE_HIP=1)
endif()

# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------
add_compile_options(
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wextra>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror>)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(rocminfer_core STATIC
    src/profiler.cpp
    src/profiler_device_null.cpp
    src/format_registry.cpp
    src/rocmfp4_reference.cpp
    src/gguf_reader.cpp)

target_include_directories(rocminfer_core
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(ROCMINFER_ENABLE_HIP)
    target_sources(rocminfer_core PRIVATE src/profiler_hip.cpp)
    target_link_libraries(rocminfer_core PUBLIC hip::host)
endif()

# ---------------------------------------------------------------------------
# Front end
# ---------------------------------------------------------------------------
add_executable(rocminfer src/main.cpp)
target_link_libraries(rocminfer PRIVATE rocminfer_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_executable(rocminfer_tests tests/test_profiler.cpp)
target_link_libraries(rocminfer_tests PRIVATE rocminfer_core)
add_test(NAME profiler COMMAND rocminfer_tests)

add_executable(rocminfer_format_tests tests/test_rocmfp4.cpp)
target_link_libraries(rocminfer_format_tests PRIVATE rocminfer_core)
add_test(NAME rocmfp4_reference COMMAND rocminfer_format_tests)

add_executable(rocminfer_gguf_tests tests/test_gguf.cpp)
target_link_libraries(rocminfer_gguf_tests PRIVATE rocminfer_core)
add_test(NAME gguf_reader COMMAND rocminfer_gguf_tests)