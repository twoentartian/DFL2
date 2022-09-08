cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(HUNTER_BUILD_SHARED_LIBS "Value for BUILD_SHARED_LIBS for 3rd party packages" OFF)

include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://github.com/twoentartian/hunter/archive/refs/tags/test.tar.gz"
        SHA1 "05269beb951f506ecc0434e2efbb92e34c4d05b9"
        LOCAL
)

project(DFL2 C CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(DFL_LINKER_LIBS)

# Caffe cmake
add_subdirectory(3rd/caffe)
include_directories(3rd/caffe/include)
include_directories(${CMAKE_BINARY_DIR}/3rd/caffe/include)
include(cmake/CaffeHeader.cmake)
list(APPEND DFL_LINKER_LIBS caffe)

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

# Boost
find_package(Boost COMPONENTS system filesystem serialization)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    list(APPEND DFL_LINKER_LIBS ${Boost_LIBRARIES})
else()
    hunter_add_package(Boost COMPONENTS system filesystem serialization)
    find_package(Boost CONFIG REQUIRED system filesystem serialization)
    include_directories(${Boost_INCLUDE_DIRS})
    list(APPEND DFL_LINKER_LIBS Boost::system Boost::filesystem Boost::serialization)
endif ()
if (Boost_VERSION VERSION_GREATER_EQUAL 1.76)
    add_compile_definitions(Boost_dll_import_symbol)
endif ()


# OpenSSL
find_package(OpenSSL)
if (OPENSSL_FOUND)
    message(STATUS "Find OpenSSL ${OPENSSL_VERSION} in system")
    include_directories(${OPENSSL_INCLUDE_DIRS})
    list(APPEND DFL_LINKER_LIBS ${OPENSSL_LIBRARIES})
else()
    message(STATUS "OpenSSL not found, use hunter")
    hunter_add_package(OpenSSL)
    find_package(OpenSSL REQUIRED)
    list(APPEND DFL_LINKER_LIBS OpenSSL::Crypto)
endif ()

# glog
find_package(glog)
if (glog_FOUND)
    message(STATUS "Find glog ${glog_VERSION} in system")
    include_directories(${glog_INCLUDE_DIRS})
    list(APPEND DFL_LINKER_LIBS ${glog_LIBRARIES})
else()
    hunter_add_package(glog)
    find_package(glog CONFIG REQUIRED)
    list(APPEND DFL_LINKER_LIBS glog::glog)
endif ()

# RocksDB
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

# LZ4
find_package(LZ4)
if (LZ4_FOUND)
    message(STATUS "Find LZ4 in system")
    include_directories(${LZ4_INCLUDE_DIRS})
    list(APPEND DFL_LINKER_LIBS ${LZ4_LIBRARIES})
else()
    hunter_add_package(lz4)
    find_package(LZ4 CONFIG REQUIRED)
    list(APPEND DFL_LINKER_LIBS LZ4::lz4_static)
endif ()


include_directories(lib)

add_subdirectory(tests)
add_subdirectory(bin)
