# ROCMinfer build definition.
# ROCMinfer is an independent, from-scratch AMD-first inference engine. See AGENTS.md
# for project rules and docs/PERF_TRACE.md for the instrumentation contract.

cmake_minimum_required(VERSION 3.21)

# ROCMINFER_ENABLE_HIP must be known before project() runs. enable_language(HIP)
# inside an if() block is a silent no-op for source classification: CMake fixes
# each source's language when the target is created, and by then the language is
# not registered, so .hip files fall back to CXX and no device code is emitted.
# Configure with -DROCMINFER_ENABLE_HIP=ON (default when a HIP toolchain exists).
option(ROCMINFER_ENABLE_HIP "Compile the HIP device backend" OFF)

if(ROCMINFER_ENABLE_HIP)
    project(rocminfer
        VERSION 0.1.0
        DESCRIPTION "AMD-first LLM inference engine"
        LANGUAGES CXX HIP)
else()
    project(rocminfer
        VERSION 0.1.0
        DESCRIPTION "AMD-first LLM inference engine"
        LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# HIP / ROCm
# ---------------------------------------------------------------------------
# Set ROCMINFER_ENABLE_HIP=ON to compile the device backend and the HIP event
# pool used for per-component device timing. When OFF the engine builds as a
# host-only binary whose reports carry no device column content.
if(ROCMINFER_ENABLE_HIP)
    find_package(hip REQUIRED)
    if(NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx1201" CACHE STRING
            "Comma-separated gfx targets, e.g. gfx1031;gfx1201")
    endif()
    add_compile_definitions(ROCMINFER_HAVE_HIP=1)
    message(STATUS "ROCMinfer: HIP enabled for ${CMAKE_HIP_ARCHITECTURES}")
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
# Sanitizers (host-only)
# ---------------------------------------------------------------------------
# HIP plus sanitizers is unproven on this toolchain: the device pass and the ROCm
# runtime have never been executed under AddressSanitizer here, and a runtime that
# aborts inside its own query path is worse than an uninstrumented one. The option
# therefore instruments only the translation units that carry no device code (the
# GGUF reader, sha256, the format registry, the ROCmFP4 CPU reference, the
# profiler host path) plus the negative/fuzz driver. Every HIP target, the device
# event pool, and the front end are deliberately left alone.
option(ROCMINFER_ENABLE_SANITIZERS
       "Instrument the host-only targets with AddressSanitizer + UndefinedBehaviorSanitizer"
       OFF)

# Targets that contain no device code. Appended as they are declared so the
# sanitizer flags have exactly one definition, applied once near the end.
set(ROCMINFER_HOST_ONLY_TARGETS "")

set(ROCMINFER_SANITIZER_FLAGS "")
set(ROCMINFER_ASAN_RUNTIME_DLL "")

if(ROCMINFER_ENABLE_SANITIZERS)
    set(ROCMINFER_SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)

    # On Windows AddressSanitizer ships only as a runtime DLL that the driver
    # links dynamically, so an instrumented executable cannot start without it on
    # the search path. Resolve it at configure time and stage it next to every
    # instrumented target; a missing runtime would otherwise look like a crash
    # with no report rather than a configuration error.
    if(CMAKE_HOST_WIN32)
        execute_process(
            COMMAND "${CMAKE_CXX_COMPILER}" -print-resource-dir
            OUTPUT_VARIABLE _rocminfer_resource_dir
            RESULT_VARIABLE _rocminfer_resource_status
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT _rocminfer_resource_status EQUAL 0 OR
           NOT EXISTS "${_rocminfer_resource_dir}")
            message(FATAL_ERROR
                "ROCMINFER_ENABLE_SANITIZERS: '${CMAKE_CXX_COMPILER} -print-resource-dir' "
                "failed, so the sanitizer runtime cannot be located. Configure with "
                "-DROCMINFER_ENABLE_SANITIZERS=OFF.")
        endif()

        set(ROCMINFER_ASAN_RUNTIME_DLL
            "${_rocminfer_resource_dir}/lib/windows/clang_rt.asan_dynamic-x86_64.dll")
        if(NOT EXISTS "${ROCMINFER_ASAN_RUNTIME_DLL}")
            message(FATAL_ERROR
                "ROCMINFER_ENABLE_SANITIZERS: AddressSanitizer runtime not found at "
                "${ROCMINFER_ASAN_RUNTIME_DLL}. Configure with -DROCMINFER_ENABLE_SANITIZERS=OFF.")
        endif()
        message(STATUS "ROCMinfer: sanitizers ON for host-only targets, "
                       "runtime ${ROCMINFER_ASAN_RUNTIME_DLL}")
    else()
        message(STATUS "ROCMinfer: sanitizers ON for host-only targets")
    endif()
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(rocminfer_core STATIC
    src/profiler.cpp
    src/profiler_device_null.cpp
    src/format_registry.cpp
    src/rocmfp4_reference.cpp
    src/gguf_reader.cpp
    src/sha256.cpp
    src/gpu_buffer.cpp
    src/model.cpp)

target_include_directories(rocminfer_core
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(ROCMINFER_ENABLE_HIP)
    target_sources(rocminfer_core PRIVATE src/profiler_hip.cpp)
    # Device kernels. The .hip extension makes CMake classify these as HIP, and
    # the explicit LANGUAGE property is a guard: if the HIP language ever stops
    # being enabled at configure time, this fails the build instead of silently
    # routing device code through the C++ compiler.
    #
    # Portable kernels build for every configured arch; they contain no device
    # feature that is absent on either gfx1031 or gfx1201 (sdot8, FP32 FMA).
    target_sources(rocminfer_core PRIVATE src/kernels/rocmfp4_dot.hip)
    set_source_files_properties(src/kernels/rocmfp4_dot.hip PROPERTIES LANGUAGE HIP)
    target_sources(rocminfer_core PRIVATE src/kernels/int4_dot.hip)
    set_source_files_properties(src/kernels/int4_dot.hip PROPERTIES LANGUAGE HIP)
    target_sources(rocminfer_core PRIVATE src/kernels/int4_matvec.hip)
    set_source_files_properties(src/kernels/int4_matvec.hip PROPERTIES LANGUAGE HIP)
    target_sources(rocminfer_core PRIVATE src/kernels/int4_wmma.hip)
    set_source_files_properties(src/kernels/int4_wmma.hip PROPERTIES LANGUAGE HIP)

    # WMMA kernels use rocWMMA, which static_asserts "Unsupported architecture"
    # for any target without a matrix core (gfx1031). rocWMMA's include cannot
    # therefore pass through a gfx1031 device compile pass. CMake does NOT honor
    # a per-source HIP_ARCHITECTURES property for HIP (the file still receives
    # every --offload-arch from CMAKE_HIP_ARCHITECTURES on a multi-arch build),
    # so the WMMA sources live in a dedicated OBJECT target whose target-level
    # HIP_ARCHITECTURES is pinned to gfx1201. A multi-arch (gfx1031;gfx1201) build
    # then compiles the rocFP4/int4 device code for both arches and the WMMA tile
    # only for gfx1201, never invoking rocWMMA under a gfx1031 target.
    list(FIND CMAKE_HIP_ARCHITECTURES "gfx1201" _rocminfer_wmma_idx)
    if(_rocminfer_wmma_idx GREATER_EQUAL 0)
        set(ROCMINFER_HAVE_WMMA ON CACHE INTERNAL "WMMA (gfx1201) backend present")
        add_compile_definitions(ROCMINFER_HAVE_WMMA=1)
        add_library(rocminfer_wmma_hip OBJECT
            src/kernels/matmul_wmma.hip
            src/kernels/rocmfp4_wmma_dot.hip)
        set_target_properties(rocminfer_wmma_hip PROPERTIES
            HIP_ARCHITECTURES "gfx1201")
        target_include_directories(rocminfer_wmma_hip
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(rocminfer_wmma_hip PUBLIC hip::host)
        target_link_libraries(rocminfer_core PUBLIC rocminfer_wmma_hip)
    endif()
    target_link_libraries(rocminfer_core PUBLIC hip::host)
endif()

# ---------------------------------------------------------------------------
# Host-only core (no device code)
# ---------------------------------------------------------------------------
# The subset of the engine that has no HIP dependency, built on its own so the
# negative/fuzz driver and any future host-only consumer link against exactly
# these translation units and nothing else. Under a HIP-enabled configure this
# library still compiles the null device-timing backend rather than the HIP one,
# which keeps it runnable where no GPU is present.
add_library(rocminfer_host_core OBJECT
    src/profiler.cpp
    src/profiler_device_null.cpp
    src/format_registry.cpp
    src/rocmfp4_reference.cpp
    src/gguf_reader.cpp
    src/sha256.cpp)
list(APPEND ROCMINFER_HOST_ONLY_TARGETS rocminfer_host_core)

target_include_directories(rocminfer_host_core
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# profiler_device_null.cpp is guarded only by !ROCMINFER_HAVE_HIP, so clearing the
# define for this target alone selects the null DeviceEventPool without touching
# the HIP backend in rocminfer_core. The remaining host sources do not read the
# macro, and -U follows the directory-level -D on the command line.
target_compile_options(rocminfer_host_core PRIVATE "-UROCMINFER_HAVE_HIP")

# ---------------------------------------------------------------------------
# Front end
# ---------------------------------------------------------------------------
add_executable(rocminfer src/main.cpp)
target_link_libraries(rocminfer PRIVATE rocminfer_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

# Doxygen validation gate: WARN_AS_ERROR=ON, zero warnings. The docs/Doxyfile
# is configured to fail (non-zero exit) on any documentation error.
find_program(DOXYGEN_EXECUTABLE doxygen)
if(DOXYGEN_EXECUTABLE)
    add_test(NAME doxy-validate
             COMMAND "${DOXYGEN_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile"
             WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
add_executable(rocminfer_tests tests/test_profiler.cpp)
target_link_libraries(rocminfer_tests PRIVATE rocminfer_core)
add_test(NAME profiler COMMAND rocminfer_tests)

add_executable(rocminfer_format_tests tests/test_rocmfp4.cpp)
target_link_libraries(rocminfer_format_tests PRIVATE rocminfer_core)
add_test(NAME rocmfp4_reference COMMAND rocminfer_format_tests)

add_executable(rocminfer_gguf_tests tests/test_gguf.cpp)
target_link_libraries(rocminfer_gguf_tests PRIVATE rocminfer_core)
add_test(NAME gguf_reader COMMAND rocminfer_gguf_tests)

add_executable(rocminfer_sha256_tests tests/test_sha256.cpp)
target_link_libraries(rocminfer_sha256_tests PRIVATE rocminfer_core)
add_test(NAME sha256 COMMAND rocminfer_sha256_tests)

# Negative and fuzz driver over the host-only paths. The corpus is deterministic
# and needs no model file or GPU: truncated GGUF streams cut at every byte
# offset, unknown metadata and tensor types, absurd lengths, counts and negative
# fields, all 256 scale bytes in both ROCmFP4 layouts, and every nibble code.
# It links rocminfer_host_core rather than rocminfer_core so it can run under the
# sanitizer build without any HIP translation unit entering the link.
add_executable(rocminfer_fuzz tests/test_fuzz.cpp)
list(APPEND ROCMINFER_HOST_ONLY_TARGETS rocminfer_fuzz)
target_link_libraries(rocminfer_fuzz PRIVATE rocminfer_host_core)
add_test(NAME fuzz_negative COMMAND rocminfer_fuzz)

# Requires real ROCmFPX/ROCmFP4 model files; skips cleanly when they are absent.
add_executable(rocminfer_payload_tests tests/test_rocmfp4_payload.cpp)
target_link_libraries(rocminfer_payload_tests PRIVATE rocminfer_core)
add_test(NAME rocmfp4_payload COMMAND rocminfer_payload_tests)

# Device kernel test: requires a working HIP runtime and a real GPU.
if(ROCMINFER_ENABLE_HIP)
    add_executable(rocminfer_kernel_tests tests/test_rocmfp4_dot.cpp)
    target_link_libraries(rocminfer_kernel_tests PRIVATE rocminfer_core)
    add_test(NAME rocmfp4_dot COMMAND rocminfer_kernel_tests)

    # End-to-end matvec: loads a real model tensor to GPU and runs the fused
    # dequantize-and-dot kernel, verifying correctness against the CPU oracle.
    add_executable(rocminfer_matvec_tests tests/test_matvec_e2e.cpp)
    target_link_libraries(rocminfer_matvec_tests PRIVATE rocminfer_core)
    add_test(NAME matvec_e2e COMMAND rocminfer_matvec_tests)

    # Integer-dot primitive: verified 4-bit signed int4 x int4 via sdot8 on the
    # GPU, validated against an exact integer reference. Available on both
    # consumer targets; not gated on WMMA.
    add_executable(rocminfer_int4_dot_tests tests/test_int4_dot.cpp)
    target_link_libraries(rocminfer_int4_dot_tests PRIVATE rocminfer_core)
    add_test(NAME int4_dot COMMAND rocminfer_int4_dot_tests)

    # Integer broadcast matvec (shared-x): genuine s4 rows x one shared s4
    # vector, sdot8 on gfx1031 / scalar elsewhere. Portable, not WMMA-gated.
    add_executable(rocminfer_int4_matvec_tests tests/test_int4_matvec.cpp)
    target_link_libraries(rocminfer_int4_matvec_tests PRIVATE rocminfer_core)
    add_test(NAME int4_matvec COMMAND rocminfer_int4_matvec_tests)

    # int4 x int4 WMMA tile: raw iu4 builtin on gfx1201, sdot8/scalar
    # fallback elsewhere. Arch-guarded, no rocWMMA include, so portable.
    add_executable(rocminfer_int4_wmma_tests tests/test_int4_wmma.cpp)
    target_link_libraries(rocminfer_int4_wmma_tests PRIVATE rocminfer_core)
    add_test(NAME int4_wmma COMMAND rocminfer_int4_wmma_tests)

    # Q4_0_INT4 registered format: a real GGUF on disk loaded through
    # GgufReader, then Model::matvec_int4x against an exact int64 oracle,
    # plus both dispatch branches and the negative gates.
    add_executable(rocminfer_int4_format_e2e_tests tests/test_int4_format_e2e.cpp)
    target_link_libraries(rocminfer_int4_format_e2e_tests PRIVATE rocminfer_core)
    add_test(NAME int4_format_e2e COMMAND rocminfer_int4_format_e2e_tests)

    add_executable(rocminfer_bench_matmul tools/bench_matmul.cpp)
    target_link_libraries(rocminfer_bench_matmul PRIVATE rocminfer_core)

    if(ROCMINFER_HAVE_WMMA)
        # Phase 5: WMMA correctness + benchmark vs register FMA on gfx1201.
        add_executable(rocminfer_wmma_tests tests/test_matmul_wmma.cpp)
        target_link_libraries(rocminfer_wmma_tests PRIVATE rocminfer_core)
        add_test(NAME matmul_wmma COMMAND rocminfer_wmma_tests)

        # Phase 5: ROCmFP4 dequantize-and-WMMA correctness vs CPU oracle + timing.
        add_executable(rocminfer_wmma_dot_tests tests/test_rocmfp4_wmma_dot.cpp)
        target_link_libraries(rocminfer_wmma_dot_tests PRIVATE rocminfer_core)
        add_test(NAME rocmfp4_wmma_dot COMMAND rocminfer_wmma_dot_tests)
    endif()
endif()

# Golden cross-check: the committed artifacts must exist. The comparison itself
# runs through Python, which is not a build dependency, so it is registered only
# when an interpreter is present at configure time.
find_program(ROCMINFER_PYTHON NAMES python3 python)
if(ROCMINFER_PYTHON AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/docs/goldens/artifacts/tiny-Q4_0_ROCMFP4.gguf")
    add_test(NAME golden_crosscheck_dual
             COMMAND "${ROCMINFER_PYTHON}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/crosscheck_decode.py"
                     "$<TARGET_FILE:rocminfer>"
                     "${CMAKE_CURRENT_SOURCE_DIR}/docs/goldens/artifacts/tiny-Q4_0_ROCMFP4.gguf")
    add_test(NAME golden_crosscheck_fast
             COMMAND "${ROCMINFER_PYTHON}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/crosscheck_decode.py"
                     "$<TARGET_FILE:rocminfer>"
                     "${CMAKE_CURRENT_SOURCE_DIR}/docs/goldens/artifacts/tiny-Q4_0_ROCMFP4_FAST.gguf")
    add_test(NAME golden_digest_verify
             COMMAND "${ROCMINFER_PYTHON}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_goldens.py" --verify
                     "${CMAKE_CURRENT_SOURCE_DIR}/docs/goldens/artifacts/tiny-src-f32.gguf"
                     "${CMAKE_CURRENT_SOURCE_DIR}/docs/goldens/artifacts/tiny-Q4_0_ROCMFP4.gguf"
                     "${CMAKE_CURRENT_SOURCE_DIR}/docs/goldens/artifacts/tiny-Q4_0_ROCMFP4_FAST.gguf")
endif()

# ---------------------------------------------------------------------------
# Sanitizer instrumentation (host-only targets only)
# ---------------------------------------------------------------------------
# Applied here, once, so the flag list has a single definition. HIP targets are
# excluded by construction: they never enter ROCMINFER_HOST_ONLY_TARGETS.
if(ROCMINFER_ENABLE_SANITIZERS)
    foreach(_rocminfer_target IN LISTS ROCMINFER_HOST_ONLY_TARGETS)
        target_compile_options(${_rocminfer_target} PRIVATE ${ROCMINFER_SANITIZER_FLAGS})
        target_link_options(${_rocminfer_target} PRIVATE ${ROCMINFER_SANITIZER_FLAGS})
        # The runtime is staged next to the executables only. CMake forbids
        # POST_BUILD commands on an OBJECT library because it has no link step,
        # and the DLL needs to sit beside the process image anyway: the Windows
        # loader searches the executable's directory, so staging it there covers
        # every instrumented translation unit the process links.
        if(ROCMINFER_ASAN_RUNTIME_DLL)
            get_target_property(_rocminfer_target_type ${_rocminfer_target} TYPE)
            if(NOT _rocminfer_target_type STREQUAL "OBJECT_LIBRARY")
                add_custom_command(TARGET ${_rocminfer_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${ROCMINFER_ASAN_RUNTIME_DLL}"
                            "$<TARGET_FILE_DIR:${_rocminfer_target}>"
                    COMMENT "Staging the AddressSanitizer runtime for ${_rocminfer_target}")
            endif()
        endif()
    endforeach()
endif()