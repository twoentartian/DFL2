cmake_minimum_required(VERSION 3.16)
project(DFL C CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://github.com/cpp-pm/hunter/archive/v0.24.3.tar.gz"
        SHA1 "10738b59e539818a01090e64c2d09896247530c7"
)



# Caffe cmake
set(CAFFE_HOME 3rd/caffe)
add_subdirectory(${CAFFE_HOME})
include_directories(${CAFFE_HOME}/include)

# Nlohmann Json
set(JSON_HOME 3rd/json)
add_subdirectory(${JSON_HOME})
include_directories(${JSON_HOME}/include)





# Boost
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.76.0 REQUIRED COMPONENTS system filesystem serialization unit_test_framework )
include_directories(${Boost_INCLUDE_DIRS})

# find glog
find_package(GLOG REQUIRED)

# find protobuf
find_package(Protobuf REQUIRED)

# find levelDB
find_package(LevelDB REQUIRED)

# find snappy
find_package(Snappy REQUIRED)

# find LMDB
find_package(LMDB REQUIRED)

# find OpenCV
find_package(OpenCV REQUIRED)

# OpenSSL
find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})

# RocksDB
find_package(RocksDB REQUIRED)
include_directories(${RocksDB_INCLUDE_DIR})

# LZ4
find_package(LZ4 REQUIRED)
include_directories(${LZ4_INCLUDE_DIR})

list(APPEND Caffe_libs "${CAFFE_LIB}" "${GLOG_LIBRARY}" "${Protobuf_LIBRARIES}" "${snappy_LIBRARIES}" "${LevelDB_LIBRARIES}" "${LMDB_LIBRARIES}" "${OpenCV_LIBS}" "${Boost_LIBRARIES}" "${LZ4_LIBRARIES}")

# TensorFlow
#set(TENSORFLOW_HOME 3rd/tensorflow)
#include_directories(${TENSORFLOW_HOME}/build/include)
#link_directories(${TENSORFLOW_HOME}/build/link)
#set(TENSORFLOW_LIB libtensorflow_cc.so libtensorflow_framework.so libprotobuf.so)

include_directories(lib)

add_subdirectory(tests)
add_subdirectory(bin)
