cmake_minimum_required(VERSION 3.20)
project(chaos_anomaly_detector CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find the parent pomaidb project
set(POMAIDB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# Include directories for pomaidb headers
include_directories(
    ${POMAIDB_ROOT}/include
    ${POMAIDB_ROOT}/src
    ${POMAIDB_ROOT}/src/capi
    ${POMAIDB_ROOT}/src/utils
    ${POMAIDB_ROOT}/third_party/palloc/include
)

# Add the executable
add_executable(chaos_anomaly_detector chaos_anomaly_detector.cpp)

# Link against pomaidb libraries - we need to import the targets
# Try to find the built pomaidb libraries
find_library(POMAI_LIB pomai PATHS ${POMAIDB_ROOT}/build ${POMAIDB_ROOT}/build/src NO_DEFAULT_PATH)
find_library(POMAI_C_LIB pomai_c PATHS ${POMAIDB_ROOT}/build ${POMAIDB_ROOT}/build/src/capi NO_DEFAULT_PATH)

# Try different palloc library names
find_library(PALLOC_LIB palloc-static PATHS ${POMAIDB_ROOT}/build/third_party/palloc NO_DEFAULT_PATH)
find_library(PALLOC_LIB_ALT palloc PATHS ${POMAIDB_ROOT}/build/third_party/palloc NO_DEFAULT_PATH)
find_library(PALLOC_LIB_MAIN palloc PATHS ${POMAIDB_ROOT}/build NO_DEFAULT_PATH)
find_library(PALLOC_LIB_STATIC libpalloc.a PATHS ${POMAIDB_ROOT}/build/third_party/palloc NO_DEFAULT_PATH)

if(NOT POMAI_LIB OR NOT POMAI_C_LIB)
    message(WARNING "pomaidb libraries not found in default build directory.")
    message(WARNING "Please build pomaidb first: cd ${POMAIDB_ROOT} && mkdir build && cd build && cmake .. && cmake --build .")
    message(WARNING "Or specify POMAIDB_BUILD_DIR: cmake -DPOMAIDB_BUILD_DIR=/path/to/pomaidb/build ..")
endif()

# If libraries found, link them
if(POMAI_LIB AND POMAI_C_LIB)
    target_link_libraries(chaos_anomaly_detector ${POMAI_LIB} ${POMAI_C_LIB})
    
    # Try to link palloc from any of the possible locations/names
    if(PALLOC_LIB)
        target_link_libraries(chaos_anomaly_detector ${PALLOC_LIB})
        message(STATUS "Found palloc library: ${PALLOC_LIB}")
    elseif(PALLOC_LIB_ALT)
        target_link_libraries(chaos_anomaly_detector ${PALLOC_LIB_ALT})
        message(STATUS "Found palloc library: ${PALLOC_LIB_ALT}")
    elseif(PALLOC_LIB_MAIN)
        target_link_libraries(chaos_anomaly_detector ${PALLOC_LIB_MAIN})
        message(STATUS "Found palloc library: ${PALLOC_LIB_MAIN}")
    elseif(PALLOC_LIB_STATIC)
        target_link_libraries(chaos_anomaly_detector ${PALLOC_LIB_STATIC})
        message(STATUS "Found palloc library: ${PALLOC_LIB_STATIC}")
    else()
        message(WARNING "palloc library not found, linking may fail")
    endif()
    
    message(STATUS "Found pomai library: ${POMAI_LIB}")
    message(STATUS "Found pomai_c library: ${POMAI_C_LIB}")
endif()

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(chaos_anomaly_detector Threads::Threads)

# Windows-specific settings
if(WIN32)
    target_link_libraries(chaos_anomaly_detector ws2_32)
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(chaos_anomaly_detector PRIVATE /W4)
else()
    target_compile_options(chaos_anomaly_detector PRIVATE -Wall -Wextra)
endif()

# Print build information
message(STATUS "Building chaos_anomaly_detector")
message(STATUS "POMAIDB_ROOT: ${POMAIDB_ROOT}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
