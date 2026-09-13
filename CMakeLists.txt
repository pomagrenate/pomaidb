cmake_minimum_required(VERSION 3.20)
project(pomai VERSION 0.1.0 LANGUAGES C CXX)

# Fix C++20 attribute placement in vendored palloc headers if needed
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/palloc/include/palloc_vector.h")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/third_party/palloc/include/palloc_vector.h" PALLOC_VEC_HDR)
  if (PALLOC_VEC_HDR MATCHES "static inline pa_decl_nodiscard")
    string(REPLACE "static inline pa_decl_nodiscard" "pa_decl_nodiscard static inline" PALLOC_VEC_HDR "${PALLOC_VEC_HDR}")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/palloc/include/palloc_vector.h" "${PALLOC_VEC_HDR}")
  endif()
endif()

# =========================
# Global build settings
# =========================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(POMAI_BUILD_TESTS "Build tests" ON)
option(POMAI_BUILD_BENCH "Build benchmarks" OFF)
option(POMAI_REGISTER_BENCH_CTEST "Register benchmark executables as CTest (ctest -L bench)" ON)

# Prefer integer (SQ8/FP16) distance paths where data is quantized; reduces float use on embedded.
option(POMAI_PREFER_INTEGER_MATH "Prefer integer/SQ8/FP16 paths for distance (embedded)" ON)

# Edge-oriented build profile for constrained devices.
# When enabled, we bias towards smaller binaries and lower memory/debug overhead.
option(POMAI_EDGE_BUILD "Optimize PomaiDB build for edge devices (size/footprint)" OFF)

# TFLite build options: minimize dependencies
add_compile_definitions(TFLITE_WITH_RUY=0)
add_compile_definitions(TFLITE_WITH_XNNPACK=0)
add_compile_definitions(TFLITE_WITHOUT_XNNPACK=1)
add_compile_definitions(TFLITE_USE_APPLE_ACCELERATE=0)
add_compile_definitions(TFLITE_WITH_DELEGATES=0)

# Strict build mode: treat project warnings as errors (useful for CI/hardening).
# We keep a couple of warning categories non-fatal to tolerate intentional diagnostics
# from some vendored headers included by the core.
option(POMAI_STRICT "Treat warnings as errors for PomaiDB code" OFF)

# Ensure POMAI_PREFER_INTEGER_MATH is defined for source code (0 or 1)
add_compile_definitions(POMAI_PREFER_INTEGER_MATH=$<BOOL:${POMAI_PREFER_INTEGER_MATH}>)

if (POMAI_EDGE_BUILD)
  add_compile_definitions(POMAI_EDGE_BUILD=1)
  # Favor size over speed and strip most debug info in edge builds.
  add_compile_options(-Os -g0)
endif()

if (MINGW)
  add_link_options(-static-libgcc -static-libstdc++)
endif()


# =========================
# Native HNSW (Replaces FAISS HNSW)
# =========================
message(STATUS "[pomai] Native HNSW: enabled")

# Khronos Vulkan-Headers + Vulkan-Hpp (vendored under third_party/vulkan/include).
# Override with: cmake -DPOMAI_VULKAN_HEADERS_DIR=/path/to/include
set(POMAI_VULKAN_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan/include" CACHE PATH
    "Directory containing vulkan/vulkan.h (parent of the vulkan/ header folder)")
if(NOT EXISTS "${POMAI_VULKAN_HEADERS_DIR}/vulkan/vulkan.h")
  message(FATAL_ERROR
    "Vulkan headers not found at:\n  ${POMAI_VULKAN_HEADERS_DIR}/vulkan/vulkan.h\n"
    "Place the Khronos bundle under third_party/vulkan/include/ or set POMAI_VULKAN_HEADERS_DIR.")
endif()
message(STATUS "[pomai] Vulkan headers: ${POMAI_VULKAN_HEADERS_DIR}")

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/palloc/CMakeLists.txt")
  message(FATAL_ERROR "palloc is a mandatory core dependency of PomaiDB. Run git submodule update --init --recursive")
endif()

# Hardcoded First-Class Native Allocator: palloc
set(PA_VECTOR ON)
set(PA_BUILD_STATIC ON)
set(PA_BUILD_SHARED OFF)
set(PA_BUILD_TESTS OFF)
set(PA_BUILD_OBJECT OFF)
set(PA_OVERRIDE OFF)
add_subdirectory(third_party/palloc)

find_package(Threads REQUIRED)

# =========================
# Library: pomai
# =========================
add_library(pomai STATIC
  src/db.cc
  src/database.cc
  src/utils/palloc_override.cc
  src/utils/palloc_page_pool.cc
  src/vector_engine.cc
  src/simd_dispatch.cc
  src/semantic_lifecycle.cc
  src/aes_gcm.cc
  src/segment_manifest.cc
  src/membrane_manager.cc
  src/memtable.cc
  src/wal.cc
  src/vector_pod.cc
  src/manifest.cc
  src/storage_engine.cc
  src/segment.cc
  src/sync_provider.cc
  src/scheduler.cc
  src/volk.c
  src/vulkan_init.cc
  src/vulkan_device_context.cc
  src/vulkan_memory_bridge.cc
  src/vulkan_staging_pool.cc

  src/scalar_quantizer.cc
  src/half_float_quantizer.cc
  src/bit_quantizer.cc
  src/ivf_coarse.cc
  src/ivf_flat.cc
  src/distance.cc
  src/reference_distance.cc

  src/routing_table.cc
  src/kmeans_lite.cc
  src/routing_persist.cc

  # Phase 2: Product Quantizer
  src/pomai_pq.cc

  # Phase 3 (Milvus-Lite inspired): BitsetMask pre-computed filter bitset
  src/bitset_mask.cc

  src/hnsw_index.cc
  third_party/pomaidb_hnsw/hnsw.cc

  # Pomegranate Engine Architecture
  src/pulp.cc
  src/aril.cc
  src/locule.cc
  src/compass.cc
  src/fruit_map.cc
  src/rind.cc
  src/press.cc
  src/pomegranate_query.cc
  src/pomegranate_engine.cc

  # Utilities (linked into libpomai)
  src/utils/posix_file.cc
  src/utils/posix_env.cc
  src/utils/memory_env.cc
  src/utils/env.cc
  src/utils/crc32c.cc
  src/utils/logging.cc
)

target_include_directories(pomai
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/capi
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${POMAI_VULKAN_HEADERS_DIR}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

if (WIN32)
  target_compile_definitions(pomai PRIVATE VK_USE_PLATFORM_WIN32_KHR)
endif()

# Vendored code: keep strict warnings for PomaiDB, but avoid breaking builds on
# third_party sources when POMAI_STRICT is enabled.
if (POMAI_STRICT AND NOT MSVC)
  set_source_files_properties(third_party/pomaidb_hnsw/hnsw.cc PROPERTIES COMPILE_OPTIONS
    "-Wno-error;-Wno-shadow;-Wno-unused-parameter;-Wno-unused-but-set-variable")
endif()

# OpenMP for parallel builds if needed
find_package(OpenMP QUIET)
if (OpenMP_CXX_FOUND)
  target_link_libraries(pomai PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(OpenSSL QUIET)
# PUBLIC: static lib pomai leaves unresolved pa_* until consumers link palloc-static too.
target_link_libraries(pomai PUBLIC palloc-static Threads::Threads)
target_compile_definitions(pomai PUBLIC PA_VECTOR=1)
if (OPENSSL_FOUND)
  target_link_libraries(pomai PRIVATE OpenSSL::Crypto)
  target_compile_definitions(pomai PRIVATE POMAI_HAS_OPENSSL=1)
endif()
if (UNIX AND NOT APPLE)
  target_link_libraries(pomai PRIVATE ${CMAKE_DL_LIBS})
endif()

if (MSVC)
  target_compile_options(pomai PRIVATE /W4 /permissive-)
else()
  # Strict warnings for pomaidb code; then suppress vendored third_party (simd) warnings.
  # We do not use / -Werror globally because some third_party headers intentionally emit #warning.
  target_compile_options(pomai PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    -Wunused-parameter -Wunused-variable
    -Wno-cpp -Wno-unknown-pragmas -Wno-conversion -Wno-float-conversion -Wno-unused-function
  )
endif()

if (POMAI_STRICT)
  if (MSVC)
    target_compile_options(pomai PRIVATE /WX)
  else()
    target_compile_options(pomai PRIVATE
      -Werror
      # Some vendored headers use extensions that trigger pedantic/cpp warnings.
      -Wno-error=pedantic
      -Wno-error=cpp
      # third_party/simd uses #warning; pre-C++23 this triggers -Wc++23-extensions.
      # We disable that warning category entirely to keep strict builds usable.
      -Wno-c++23-extensions
    )
  endif()
endif()


# =========================
# Library: pomai_c (C ABI)
# =========================
set(POMAI_C_API_SOURCES
  src/capi/capi_status.cc
  src/capi/capi_db.cc
  src/capi/capi_scan.cc
)

add_library(pomai_c SHARED ${POMAI_C_API_SOURCES})
add_library(pomai_c_static STATIC ${POMAI_C_API_SOURCES})

set_target_properties(pomai_c_static PROPERTIES OUTPUT_NAME pomai_c)

target_link_libraries(pomai_c PRIVATE pomai)
# Static C API archive must re-export pomai usage requirements (e.g. palloc headers)
# so targets that include C++ headers such as pomai/database.h can compile.
target_link_libraries(pomai_c_static PUBLIC pomai)

# Export C API symbols when building shared library on Windows.
target_compile_definitions(pomai_c PRIVATE POMAI_C_BUILD_DLL)

target_include_directories(pomai_c
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/capi
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(pomai_c_static
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/capi
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if (MSVC)
  target_compile_options(pomai_c PRIVATE /W4 /permissive-)
  target_compile_options(pomai_c_static PRIVATE /W4 /permissive-)
else()
  target_compile_options(pomai_c PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    -Wunused-parameter -Wunused-variable
  )
  target_compile_options(pomai_c_static PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    -Wunused-parameter -Wunused-variable
  )
endif()

if (POMAI_STRICT)
  if (MSVC)
    target_compile_options(pomai_c PRIVATE /WX)
    target_compile_options(pomai_c_static PRIVATE /WX)
  else()
    target_compile_options(pomai_c PRIVATE -Werror -Wno-error=pedantic -Wno-error=cpp -Wno-c++23-extensions)
    target_compile_options(pomai_c_static PRIVATE -Werror -Wno-error=pedantic -Wno-error=cpp -Wno-c++23-extensions)
  endif()
endif()

# =========================
# Tests
# =========================
if (POMAI_BUILD_TESTS)
  enable_testing()

  # Detect TSAN builds (best-effort).
  set(POMAI_IS_TSAN OFF)
  if (CMAKE_CXX_FLAGS MATCHES "fsanitize=thread" OR CMAKE_EXE_LINKER_FLAGS MATCHES "fsanitize=thread")
    set(POMAI_IS_TSAN ON)
  endif()

  option(POMAI_ENABLE_CRASH_TESTS "Enable crash-style tests (may be incompatible with TSAN)" ON)
  if (POMAI_IS_TSAN OR WIN32)
    set(POMAI_ENABLE_CRASH_TESTS OFF)
  endif()

  function(pomai_setup_test target)
    # Link against main C++ library and C API shim
    target_link_libraries(${target} PRIVATE pomai pomai_c)

    # One main per test binary (your harness)
    target_sources(${target} PRIVATE tests/common/test_runner.cc)

    # Warnings
    if (MSVC)
      target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
      target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()

    # Allow tests to include all engine, capi, and utils headers.
    target_include_directories(${target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}              # "tests/common/..."
      ${CMAKE_CURRENT_SOURCE_DIR}/src          # all engine headers
      ${CMAKE_CURRENT_SOURCE_DIR}/src/capi     # C API headers
      ${CMAKE_CURRENT_SOURCE_DIR}/src/utils    # utils headers
    )

    # Make TSAN builds more stable on some distros (PIE can trigger weird mmap layouts).
    if (POMAI_IS_TSAN)
      target_compile_options(${target} PRIVATE -fno-pie)
      target_link_options(${target} PRIVATE -no-pie)
    endif()
  endfunction()

  function(pomai_add_labeled_test target label)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "${label}")
  endfunction()

  # ---- Unit ----
  add_executable(crc32c_test tests/unit/crc32c_test.cc)
  pomai_setup_test(crc32c_test)
  pomai_add_labeled_test(crc32c_test "unit")

  add_executable(hnsw_pomaidb_test tests/unit/hnsw_pomaidb_test.cc)
  pomai_setup_test(hnsw_pomaidb_test)
  pomai_add_labeled_test(hnsw_pomaidb_test "unit")

  add_executable(memtable_test tests/unit/memtable_test.cc)
  pomai_setup_test(memtable_test)
  pomai_add_labeled_test(memtable_test "unit")

  add_executable(palloc_page_pool_test tests/unit/palloc_page_pool_test.cc)
  pomai_setup_test(palloc_page_pool_test)
  pomai_add_labeled_test(palloc_page_pool_test "unit")

  add_executable(storage_engine_test tests/unit/storage_engine_test.cc)
  pomai_setup_test(storage_engine_test)
  pomai_add_labeled_test(storage_engine_test "unit")

  add_executable(palloc_global_test tests/unit/palloc_global_test.cc)
  pomai_setup_test(palloc_global_test)
  pomai_add_labeled_test(palloc_global_test "unit")

  add_executable(env_test tests/unit/env_test.cc)
  pomai_setup_test(env_test)
  pomai_add_labeled_test(env_test "unit")

  add_executable(mailbox_test tests/unit/mailbox_test.cc)
  pomai_setup_test(mailbox_test)
  pomai_add_labeled_test(mailbox_test "unit")

  add_executable(logging_test tests/unit/logging_test.cc)
  pomai_setup_test(logging_test)
  pomai_add_labeled_test(logging_test "unit")

  add_executable(wal_test tests/unit/wal_test.cc)
  pomai_setup_test(wal_test)
  pomai_add_labeled_test(wal_test "unit")

  add_executable(segment_test tests/unit/segment_test.cc)
  pomai_setup_test(segment_test)
  pomai_add_labeled_test(segment_test "unit")

  add_executable(fp16_test tests/unit/fp16_test.cc)
  pomai_setup_test(fp16_test)
  pomai_add_labeled_test(fp16_test "unit")
  
  add_executable(shard_manifest_test tests/unit/shard_manifest_test.cc)
  pomai_setup_test(shard_manifest_test)
  pomai_add_labeled_test(shard_manifest_test "unit")

  add_executable(oracle_test tests/unit/oracle_test.cc tests/common/bruteforce_oracle.cc)
  pomai_setup_test(oracle_test)
  pomai_add_labeled_test(oracle_test "unit")

  add_executable(routing_table_test tests/unit/routing_table_test.cc)
  pomai_setup_test(routing_table_test)
  pomai_add_labeled_test(routing_table_test "unit")

  add_executable(streaming_index_test tests/unit/streaming_index_test.cc)
  pomai_setup_test(streaming_index_test)
  pomai_add_labeled_test(streaming_index_test "unit")

  add_executable(edge_platform_test tests/unit/edge_platform_test.cc)
  pomai_setup_test(edge_platform_test)
  pomai_add_labeled_test(edge_platform_test "unit")

  add_executable(kernel_hardening_test tests/unit/kernel_hardening_test.cc)
  pomai_setup_test(kernel_hardening_test)
  pomai_add_labeled_test(kernel_hardening_test "unit")

  add_executable(kernel_differential_test tests/unit/kernel_differential_test.cc)
  pomai_setup_test(kernel_differential_test)
  pomai_add_labeled_test(kernel_differential_test "unit")

  add_executable(low_ram_limits_test tests/unit/low_ram_limits_test.cc)
  pomai_setup_test(low_ram_limits_test)
  pomai_add_labeled_test(low_ram_limits_test "unit")

  add_executable(edge_profile_options_test tests/unit/edge_profile_options_test.cc)
  pomai_setup_test(edge_profile_options_test)
  pomai_add_labeled_test(edge_profile_options_test "unit")

  add_executable(vulkan_memory_bridge_test tests/unit/vulkan_memory_bridge_test.cc)
  pomai_setup_test(vulkan_memory_bridge_test)
  pomai_add_labeled_test(vulkan_memory_bridge_test "unit")

  add_executable(rind_concurrency_test tests/unit/rind_concurrency_test.cc)
  pomai_setup_test(rind_concurrency_test)
  pomai_add_labeled_test(rind_concurrency_test "unit")

  add_executable(spatial_press_test tests/unit/spatial_press_test.cc)
  pomai_setup_test(spatial_press_test)
  pomai_add_labeled_test(spatial_press_test "unit")

  add_executable(hnsw_upstream_test tests/unit/hnsw_upstream_test.cc)
  pomai_setup_test(hnsw_upstream_test)
  pomai_add_labeled_test(hnsw_upstream_test "unit")

  add_executable(recall_test tests/recall/recall_test.cc tests/recall/recall_dataset.cc tests/common/bruteforce_oracle.cc)
  pomai_setup_test(recall_test)
  pomai_add_labeled_test(recall_test "recall")

  # ---- Integration (embedded DB API) ----
  add_executable(db_basic_test tests/integ/db_basic_test.cc)
  pomai_setup_test(db_basic_test)
  pomai_add_labeled_test(db_basic_test "integ")

  add_executable(db_batch_test tests/integ/db_batch_test.cc)
  pomai_setup_test(db_batch_test)
  pomai_add_labeled_test(db_batch_test "integ")

  add_executable(db_persistence_test tests/integ/db_persistence_test.cc)
  pomai_setup_test(db_persistence_test)
  pomai_add_labeled_test(db_persistence_test "integ")

  add_executable(db_segment_test tests/integ/db_segment_test.cc)
  pomai_setup_test(db_segment_test)
  pomai_add_labeled_test(db_segment_test "integ")

  add_executable(db_open_test tests/integ/db_open_test.cc)
  pomai_setup_test(db_open_test)
  pomai_add_labeled_test(db_open_test "integ")
  
  add_executable(db_consistency_test tests/integ/db_consistency_test.cc)
  pomai_setup_test(db_consistency_test)
  pomai_add_labeled_test(db_consistency_test "integ")

  add_executable(pomegranate_hnsw_e2e_test tests/integ/pomegranate_hnsw_e2e_test.cc)
  pomai_setup_test(pomegranate_hnsw_e2e_test)
  pomai_add_labeled_test(pomegranate_hnsw_e2e_test "integ")

  add_executable(db_partial_search_test tests/integ/db_partial_search_test.cc)
  pomai_setup_test(db_partial_search_test)
  pomai_add_labeled_test(db_partial_search_test "integ")

  add_executable(db_backpressure_test tests/integ/db_backpressure_test.cc)
  pomai_setup_test(db_backpressure_test)
  pomai_add_labeled_test(db_backpressure_test "integ")

  add_executable(db_edge_workload_test tests/integ/db_edge_workload_test.cc)
  pomai_setup_test(db_edge_workload_test)
  pomai_add_labeled_test(db_edge_workload_test "integ")

  add_executable(db_soak_stress_test tests/integ/db_soak_stress_test.cc)
  pomai_setup_test(db_soak_stress_test)
  pomai_add_labeled_test(db_soak_stress_test "stress")
  set_tests_properties(db_soak_stress_test PROPERTIES TIMEOUT 180)

  add_executable(db_error_paths_test tests/integ/db_error_paths_test.cc)
  pomai_setup_test(db_error_paths_test)
  pomai_add_labeled_test(db_error_paths_test "integ")

  add_executable(routing_engine_test tests/integ/routing_engine_test.cc)
  pomai_setup_test(routing_engine_test)
  pomai_add_labeled_test(routing_engine_test "integ")

  add_executable(manifest_test tests/unit/manifest_test.cc)
  pomai_setup_test(manifest_test)
  pomai_add_labeled_test(manifest_test "unit")

  add_executable(membrane_test tests/integ/membrane_test.cc)
  pomai_setup_test(membrane_test)
  pomai_add_labeled_test(membrane_test "integ")

  add_executable(membrane_persistence_test tests/integ/membrane_persistence_test.cc)
  pomai_setup_test(membrane_persistence_test)
  pomai_add_labeled_test(membrane_persistence_test "integ")

  add_executable(shard_concurrency_test tests/integ/shard_concurrency_test.cc)
  pomai_setup_test(shard_concurrency_test)
  pomai_add_labeled_test(shard_concurrency_test "integ")

  add_executable(iterator_test tests/integ/iterator_test.cc)
  pomai_setup_test(iterator_test)
  pomai_add_labeled_test(iterator_test "integ")

  add_executable(metadata_test tests/integ/metadata_test.cc)
  pomai_setup_test(metadata_test)
  pomai_add_labeled_test(metadata_test "integ")

  add_executable(search_filter_test tests/integ/search_filter_test.cc)
  pomai_setup_test(search_filter_test)
  pomai_add_labeled_test(search_filter_test "integ")

  add_executable(search_newest_wins_test tests/integ/search_newest_wins_test.cc)
  pomai_setup_test(search_newest_wins_test)
  pomai_add_labeled_test(search_newest_wins_test "integ")

  add_executable(manifest_corruption_test tests/crash/manifest_corruption_test.cc)
  pomai_setup_test(manifest_corruption_test)
  pomai_add_labeled_test(manifest_corruption_test "crash")

  add_executable(sd_fault_injection_test tests/crash/sd_fault_injection_test.cc)
  pomai_setup_test(sd_fault_injection_test)
  pomai_add_labeled_test(sd_fault_injection_test "crash")

  add_executable(recovery_test tests/crash/recovery_test.cc)
  pomai_setup_test(recovery_test)
  pomai_add_labeled_test(recovery_test "crash")
  set_tests_properties(recovery_test PROPERTIES TIMEOUT 90)

  # ---- Crash (Gate #2) ----
  if (POMAI_ENABLE_CRASH_TESTS)
    add_executable(pomai_crash_test tests/crash/crash_replay_test.cc)
    
    # Manual setup loosely based on pomai_setup_test but without test_runner.cc
    target_link_libraries(pomai_crash_test PRIVATE pomai)
    if (MSVC)
      target_compile_options(pomai_crash_test PRIVATE /W4 /permissive-)
    else()
      target_compile_options(pomai_crash_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
    target_include_directories(pomai_crash_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/src/capi
      ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    )
    if (POMAI_IS_TSAN)
      target_compile_options(pomai_crash_test PRIVATE -fno-pie)
      target_link_options(pomai_crash_test PRIVATE -no-pie)
    endif()

    add_test(NAME pomai_crash_replay COMMAND pomai_crash_test)
    set_tests_properties(pomai_crash_replay PROPERTIES LABELS "crash" TIMEOUT 60)

    add_executable(power_loss_stress_test tests/crash/power_loss_stress_test.cc)
    target_link_libraries(power_loss_stress_test PRIVATE pomai)
    if (MSVC)
      target_compile_options(power_loss_stress_test PRIVATE /W4 /permissive-)
    else()
      target_compile_options(power_loss_stress_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
    target_include_directories(power_loss_stress_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/src/capi
      ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    )
    if (POMAI_IS_TSAN)
      target_compile_options(power_loss_stress_test PRIVATE -fno-pie)
      target_link_options(power_loss_stress_test PRIVATE -no-pie)
    endif()

    add_test(NAME power_loss_stress COMMAND power_loss_stress_test)
    set_tests_properties(power_loss_stress PROPERTIES LABELS "crash;stress" TIMEOUT 120)
  endif()

  # ---- TSAN (Gate #3) ----
  add_executable(db_concurrency_tsan_test tests/tsan/db_concurrency_tsan_test.cc)
  pomai_setup_test(db_concurrency_tsan_test)
  pomai_add_labeled_test(db_concurrency_tsan_test "tsan")

  add_executable(shard_runtime_tsan_test tests/tsan/shard_runtime_tsan_test.cc)
  pomai_setup_test(shard_runtime_tsan_test)
  pomai_add_labeled_test(shard_runtime_tsan_test "tsan")

  add_executable(backpressure_tsan_test tests/tsan/backpressure_tsan_test.cc)
  pomai_setup_test(backpressure_tsan_test)
  pomai_add_labeled_test(backpressure_tsan_test "tsan")

  add_executable(basic_workload_tsan_test tests/tsan/basic_workload_tsan_test.cc)
  pomai_setup_test(basic_workload_tsan_test)
  pomai_add_labeled_test(basic_workload_tsan_test "tsan")

  add_subdirectory(tests/fuzz)

  # Policy check: owned PomaiDB code must avoid raw malloc/new.
  # This runs as a normal ctest so CI catches allocator regressions.
  if (UNIX)
    add_test(
      NAME allocator_policy_test
      COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_no_malloc_new.sh" "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(allocator_policy_test PROPERTIES LABELS "unit;policy")
  endif()
endif()

# =========================
# Benchmarks
# =========================
if (POMAI_BUILD_BENCH)
  # Add benchmarks here later, use a similar helper to tests (but WITHOUT test_runner.cc)
  # Example:
  # add_executable(my_bench benchmarks/my_bench.cc)
  # target_link_libraries(my_bench PRIVATE pomai)
  # target_include_directories(my_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()

set(POMAI_EXE_DEPS pomai)

# Comprehensive Benchmark (Industry-standard metrics)
add_executable(comprehensive_bench benchmarks/comprehensive_bench.cc)
target_link_libraries(comprehensive_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(comprehensive_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

# Ingestion Throughput Benchmark
add_executable(ingestion_bench benchmarks/ingestion_bench.cc)
target_link_libraries(ingestion_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(ingestion_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

# CBR-S benchmark suite
add_executable(bench_cbrs benchmarks/bench_cbrs.cpp)
target_link_libraries(bench_cbrs PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(bench_cbrs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)
set_target_properties(bench_cbrs PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Deterministic CI benchmark for perf gate
add_executable(ci_perf_bench benchmarks/ci_perf_bench.cc)
target_link_libraries(ci_perf_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(ci_perf_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

# Quantization Comparison Benchmark
add_executable(quantization_bench benchmarks/quantization_comp_bench.cc)
target_link_libraries(quantization_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(quantization_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

# benchmark_a: Multi-Environment Stress Test (IoT / Edge / Cloud validation)
add_executable(benchmark_a benchmarks/palloc_env_stress.cc)
target_link_libraries(benchmark_a PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(benchmark_a PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

# Encryption overhead benchmark harness
add_executable(encryption_perf_bench benchmarks/encryption_perf_bench.cc)
target_link_libraries(encryption_perf_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(encryption_perf_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

add_executable(low_ram_profile_bench benchmarks/low_ram_profile_bench.cc)
target_link_libraries(low_ram_profile_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(low_ram_profile_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

add_executable(vulkan_transfer_bench benchmarks/vulkan_transfer_bench.cc)
target_link_libraries(vulkan_transfer_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(vulkan_transfer_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

add_executable(kernel_dispatch_bench benchmarks/kernel_dispatch_bench.cc)
target_link_libraries(kernel_dispatch_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(kernel_dispatch_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

add_executable(kernel_micro_bench benchmarks/kernel_micro_bench.cc)
target_link_libraries(kernel_micro_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(kernel_micro_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

add_executable(crossover_bench benchmarks/crossover_bench.cc)
target_link_libraries(crossover_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(crossover_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

add_executable(production_hnsw_bench benchmarks/production_hnsw_bench.cc)
target_link_libraries(production_hnsw_bench PRIVATE ${POMAI_EXE_DEPS} Threads::Threads)
target_include_directories(production_hnsw_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

# Round 2 Benchmark Targets
add_executable(round2_cost_model benchmarks/round2_cost_model.cc)
target_link_libraries(round2_cost_model PRIVATE ${POMAI_EXE_DEPS} Threads::Threads)
target_include_directories(round2_cost_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

add_executable(round2_concurrency benchmarks/round2_concurrency.cc)
target_link_libraries(round2_concurrency PRIVATE ${POMAI_EXE_DEPS} Threads::Threads)
target_include_directories(round2_concurrency PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

add_executable(round2_compass_kmeans benchmarks/round2_compass_kmeans.cc)
target_link_libraries(round2_compass_kmeans PRIVATE ${POMAI_EXE_DEPS} Threads::Threads)
target_include_directories(round2_compass_kmeans PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

add_executable(round2_hnsw_audit benchmarks/round2_hnsw_audit.cc)
target_link_libraries(round2_hnsw_audit PRIVATE ${POMAI_EXE_DEPS} Threads::Threads)
target_include_directories(round2_hnsw_audit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

add_executable(round2_hybrid benchmarks/round2_hybrid.cc)
target_link_libraries(round2_hybrid PRIVATE ${POMAI_EXE_DEPS} Threads::Threads)
target_include_directories(round2_hybrid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

# Round 3 Benchmark Targets
add_executable(round3_hnsw_audit_suite benchmarks/round3_hnsw_audit_suite.cc)
target_link_libraries(round3_hnsw_audit_suite PRIVATE ${POMAI_EXE_DEPS} Threads::Threads)
target_include_directories(round3_hnsw_audit_suite PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

add_executable(round3_reproduce_198x benchmarks/round3_reproduce_198x.cc)
target_link_libraries(round3_reproduce_198x PRIVATE ${POMAI_EXE_DEPS} Threads::Threads)
target_include_directories(round3_reproduce_198x PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

add_executable(round3_rind_concurrency_fix benchmarks/round3_rind_concurrency_fix.cc)
target_link_libraries(round3_rind_concurrency_fix PRIVATE ${POMAI_EXE_DEPS} Threads::Threads)
target_include_directories(round3_rind_concurrency_fix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/capi ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

# Bounded benchmark smoke tests (avoid ctest default 1500s timeout on heavy workloads).
if (POMAI_REGISTER_BENCH_CTEST)
  set(POMAI_BENCH_TIMEOUT_LONG 600)
  set(POMAI_BENCH_TIMEOUT_MED 300)
  set(POMAI_BENCH_TIMEOUT_SHORT 180)

  add_test(NAME bench_comprehensive COMMAND $<TARGET_FILE:comprehensive_bench> --dataset small)
  set_tests_properties(bench_comprehensive PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_LONG})

  add_test(NAME bench_ingestion COMMAND $<TARGET_FILE:ingestion_bench> 10000 128)
  set_tests_properties(bench_ingestion PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_MED})

  add_test(NAME bench_ci_perf COMMAND $<TARGET_FILE:ci_perf_bench>)
  set_tests_properties(bench_ci_perf PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})

  add_test(NAME bench_quantization COMMAND $<TARGET_FILE:quantization_bench>)
  set_tests_properties(bench_quantization PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})

  add_test(NAME bench_cbrs_smoke COMMAND $<TARGET_FILE:bench_cbrs> --n 8000 --queries 400 --topk 10)
  set_tests_properties(bench_cbrs_smoke PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_LONG})

  add_test(NAME bench_benchmark_a COMMAND ${CMAKE_COMMAND} -E env POMAI_BENCH_LOW_MEMORY=1 $<TARGET_FILE:benchmark_a>)
  set_tests_properties(bench_benchmark_a PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_LONG})

  if (OPENSSL_FOUND)
    add_test(NAME bench_encryption_perf COMMAND $<TARGET_FILE:encryption_perf_bench>)
    set_tests_properties(bench_encryption_perf PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})
  endif()

  add_test(NAME bench_low_ram_profile COMMAND $<TARGET_FILE:low_ram_profile_bench>)
  set_tests_properties(bench_low_ram_profile PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})

  add_test(NAME bench_vulkan_transfer COMMAND $<TARGET_FILE:vulkan_transfer_bench>)
  set_tests_properties(bench_vulkan_transfer PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})

  add_test(NAME bench_kernel_dispatch COMMAND $<TARGET_FILE:kernel_dispatch_bench> 100000)
  set_tests_properties(bench_kernel_dispatch PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})
endif()

# =========================
# Tools
# =========================
add_executable(pomai_inspect src/inspect.cc)
target_link_libraries(pomai_inspect PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(pomai_inspect PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/capi
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)

# =========================
# C API Tests
# =========================
if (POMAI_BUILD_TESTS)
  add_executable(c_api_test tests/test_c_api_basic.cpp)
  target_link_libraries(c_api_test PRIVATE pomai_c_static)
  target_sources(c_api_test PRIVATE tests/common/test_runner.cc)
  target_include_directories(c_api_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/capi
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
  )
  add_test(NAME c_api_test COMMAND c_api_test)

  add_executable(accuracy_test tests/accuracy_test.cc)
  target_link_libraries(accuracy_test PRIVATE pomai)
  target_sources(accuracy_test PRIVATE tests/common/test_runner.cc)
  target_include_directories(accuracy_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/capi
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
  )
  add_test(NAME accuracy_test COMMAND accuracy_test)
endif()

install(TARGETS pomai pomai_c pomai_c_static
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(FILES
  src/capi/c_api.h
  src/capi/c_status.h
  src/capi/c_types.h
  src/capi/c_version.h
  src/database.h
  DESTINATION include/pomai)
