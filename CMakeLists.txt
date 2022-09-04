cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(HUNTER_BUILD_SHARED_LIBS "Value for BUILD_SHARED_LIBS for 3rd party packages" OFF)

include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://github.com/cpp-pm/hunter/archive/v0.24.3.tar.gz"
        SHA1 "10738b59e539818a01090e64c2d09896247530c7"
        LOCAL
)

project(DFL2 C CXX)

# Caffe cmake
#add_subdirectory(3rd/caffe)
#include_directories(3rd/caffe/include)
#include_directories(${CMAKE_BINARY_DIR}/3rd/caffe/include)

include(hunter_add_version)
set(HUNTER_Caffe_VERSION 0.0.2)
hunter_add_version(
        PACKAGE_NAME Caffe
        VERSION "0.0.2"
        URL "https://github.com/twoentartian/caffe/archive/refs/tags/0.0.2.tar.gz"
        SHA1 f174466e2b94eeae6620c112bc521e4746446552
)

hunter_add_package(Caffe)
find_package(Caffe CONFIG REQUIRED)

# Nlohmann Json
hunter_add_package(nlohmann_json)
find_package(nlohmann_json CONFIG REQUIRED)

# Boost
hunter_add_package(Boost COMPONENTS system filesystem serialization)
find_package(Boost CONFIG REQUIRED system filesystem serialization)
include_directories(${Boost_INCLUDE_DIRS})

# OpenSSL
hunter_add_package(OpenSSL)
find_package(OpenSSL REQUIRED)

# RocksDB
hunter_add_package(rocksdb)
find_package(RocksDB CONFIG REQUIRED)

# LZ4
hunter_add_package(lz4)
find_package(LZ4 CONFIG REQUIRED)

include_directories(lib)

add_subdirectory(tests)
add_subdirectory(bin)
