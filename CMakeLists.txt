cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(HUNTER_BUILD_SHARED_LIBS "Value for BUILD_SHARED_LIBS for 3rd party packages" ON)

include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://github.com/cpp-pm/hunter/archive/v0.26.6.tar.gz"
        SHA1 "e70c29f878f5d5f5cdf1b9ccd628fb872e8624a8"
        LOCAL
)

project(DFL2 C CXX)

include("cmake/CompileTimeContent.cmake")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(DFL_LINKER_LIBS)

# _LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_compile_definitions(_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
endif()

# options
option(USE_HUNTER_BOOST "USE the Boost library in Hunter" ON)
option(USE_ROCKSDB "USE RocksDB to store block/transactions" OFF)
option(USE_BACKTRACE "USE backtrace library to provide stack trace information" OFF)
option(ENABLE_BLOCK_GENERATION "Enable generating block, can be disabled to save CPU & disk & memory resources" ON)

# Build type: RelWithDebInfo
set(CMAKE_CXX_FLAGS_RELEASEWITHDEBINFO "-O2 -g")
set(CMAKE_C_FLAGS_RELEASEWITHDEBINFO "-O2 -g")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "Available build types" FORCE)

if (ENABLE_BLOCK_GENERATION)
    add_compile_definitions(BLOCK_GENERATION=1)
else()
    add_compile_definitions(BLOCK_GENERATION=0)
endif ()

# OpenSSL
#find_package(OpenSSL)
if (OPENSSL_FOUND)
    message(STATUS "Find OpenSSL ${OPENSSL_VERSION} in system")
    include_directories(${OPENSSL_INCLUDE_DIRS})
    list(APPEND DFL_LINKER_LIBS ${OPENSSL_CRYPTO_LIBRARIES})
else()
    message(STATUS "OpenSSL not found, use hunter")
    hunter_add_package(OpenSSL)
    find_package(OpenSSL REQUIRED)
    list(APPEND DFL_LINKER_LIBS OpenSSL::Crypto)
endif ()

# Boost
if (USE_HUNTER_BOOST)
    hunter_add_package(Boost COMPONENTS system filesystem serialization thread program_options)
    find_package(Boost CONFIG REQUIRED COMPONENTS system filesystem serialization thread program_options)
    include_directories(${Boost_INCLUDE_DIRS})
    list(APPEND DFL_LINKER_LIBS Boost::system Boost::filesystem Boost::serialization Boost::thread Boost::program_options)
else()
    find_package(Boost REQUIRED COMPONENTS system filesystem serialization thread program_options)
    include_directories(${Boost_INCLUDE_DIRS})
    list(APPEND DFL_LINKER_LIBS ${Boost_LIBRARIES})
endif ()
if (Boost_VERSION VERSION_GREATER_EQUAL 1.76)
    add_compile_definitions(Boost_dll_import_symbol)
endif ()

# Backtrace
if (USE_BACKTRACE)
    if(APPLE)
        message(FATAL_ERROR "backtrace is not supported on macOS")
    endif()
    list(APPEND DFL_LINKER_LIBS backtrace)
    message(STATUS "backtrace is enabled")
    add_compile_definitions(Backtrace_enable=1)
else()
    message(STATUS "backtrace is disabled")
    add_compile_definitions(Backtrace_enable=0)
endif()

## hunter Caffe
#include(hunter_add_version)
#set(HUNTER_Caffe_VERSION 0.0.5)
#hunter_add_version(
#        PACKAGE_NAME Caffe
#        VERSION "0.0.5"
#        URL "https://github.com/twoentartian/caffe/archive/refs/tags/0.0.5.tar.gz"
#        SHA1 a7f1c15416120bcd9c32873717e950290c522404
#)
#hunter_add_package(Caffe)
#find_package(Caffe CONFIG REQUIRED)

# Nlohmann Json
hunter_add_package(nlohmann_json)
find_package(nlohmann_json CONFIG REQUIRED)

# glog
find_package(GLOG)
if (GLOG_FOUND)
    message(STATUS "Find glog ${glog_VERSION} in system")
    include_directories(${glog_INCLUDE_DIRS})
    list(APPEND DFL_LINKER_LIBS ${glog_LIBRARIES})
else()
    hunter_add_package(glog)
    find_package(glog CONFIG REQUIRED)
    list(APPEND DFL_LINKER_LIBS glog::glog)
endif ()
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    list(APPEND DFL_LINKER_LIBS unwind)
endif ()

# RocksDB
if (USE_ROCKSDB)
    add_compile_definitions(USE_ROCKSDB=1)
    find_package(RocksDB)
    if (RocksDB_FOUND)
        message(STATUS "Find RocksDB in system")
        include_directories(${ROCKSDB_INCLUDE_DIR})
        list(APPEND DFL_LINKER_LIBS ${ROCKSDB_LIBRARIES})
    else()
        hunter_add_package(rocksdb)
        find_package(RocksDB CONFIG REQUIRED)
        list(APPEND DFL_LINKER_LIBS RocksDB::rocksdb)
    endif ()
else()
    add_compile_definitions(USE_ROCKSDB=0)
endif ()

# LZ4
find_package(LZ4)
if (LZ4_FOUND)
    message(STATUS "Find LZ4 in system")
    include_directories(${LZ4_INCLUDE_DIRS})
    list(APPEND DFL_LINKER_LIBS ${LZ4_LIBRARIES})
else()
    hunter_add_package(lz4)
    find_package(LZ4 CONFIG REQUIRED)
    list(APPEND DFL_LINKER_LIBS lz4::lz4)
endif ()

# BLAS
# ---[ BLAS
if(NOT APPLE)
    set(BLAS "Open" CACHE STRING "Selected BLAS library")
    set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

    if(BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")
        find_package(Atlas REQUIRED)
        include_directories(SYSTEM ${Atlas_INCLUDE_DIR})
        list(APPEND DFL_LINKER_LIBS ${Atlas_LIBRARIES})
    elseif(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
        find_package(OpenBLAS)
        if (OpenBLAS_FOUND)
            message("Find OpenBLAS in system, include_dir: " ${OpenBLAS_INCLUDE_DIR})
            include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
            list(APPEND DFL_LINKER_LIBS ${OpenBLAS_LIB})
        else()
            hunter_add_package(OpenBLAS)
            find_package(OpenBLAS CONFIG REQUIRED)
            list(APPEND DFL_LINKER_LIBS OpenBLAS::OpenBLAS)
        endif ()
    elseif(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
        find_package(MKL REQUIRED)
        include_directories(SYSTEM ${MKL_INCLUDE_DIR})
        list(APPEND DFL_LINKER_LIBS ${MKL_LIBRARIES})
    endif()
elseif(APPLE)
    find_package(vecLib REQUIRED)
    include_directories(SYSTEM ${vecLib_INCLUDE_DIR})
    list(APPEND DFL_LINKER_LIBS ${vecLib_LINKER_LIBS})
endif()

# Caffe cmake
add_subdirectory(3rd/caffe)
include_directories(3rd/caffe/include)
include_directories(${CMAKE_BINARY_DIR}/3rd/caffe/include)
include(cmake/CaffeHeader.cmake)
list(APPEND DFL_LINKER_LIBS caffe)

# AVX
include(CheckCXXSourceRuns)

set(CMAKE_REQUIRED_FLAGS "-mavx")
CHECK_CXX_SOURCE_RUNS("
    #include <immintrin.h>
    int main() {
        __m256 a;
        a = _mm256_set1_ps(0);
        return 0;
    }"
        HAS_AVX)

if (HAS_AVX)
    message(STATUS "AVX supported")
    add_definitions("-mavx")
    add_compile_definitions(USE_AVX=1)
else()
    message(STATUS "AVX not supported")
endif()

include_directories(lib)

add_subdirectory(tests)
add_subdirectory(bin)
