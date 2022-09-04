add_executable(DFL DFL.cpp)
target_link_libraries(DFL caffe Boost::serialization LZ4::lz4_static OpenSSL::Crypto RocksDB::rocksdb)
